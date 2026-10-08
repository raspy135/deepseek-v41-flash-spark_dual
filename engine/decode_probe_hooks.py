"""Temporary, capture-time decode instrumentation; never patch normal serving.

Operations retain their production implementation and ordering. Local pure leaves
can replay against private inputs; model-state writes and TP wrappers are timed
only. The controller owns GPU events/snapshots and enforces their byte budget.
"""
from contextlib import contextmanager
import functools
import sys

import torch


@contextmanager
def capture_hooks(fd, probe, key):
    import engine.fastdecode as D
    import v41_ref as R
    from engine import comm
    import fp8_linear as FP8

    saved = []
    path = ["verify"]
    parents = []
    serial = {}
    weights = {}
    weight_info = {}

    def register(value, label):
        if value is None or isinstance(value, (bool, int, float, str)):
            return
        weights[id(value)] = label
        tensors = ([value] if isinstance(value, torch.Tensor) else
                   [v for v in vars(value).values() if isinstance(v, torch.Tensor)]
                   if hasattr(value, "__dict__") else [])
        unique = {t.data_ptr(): t for t in tensors}
        weight_info[id(value)] = {"weight_bytes": sum(t.numel() * t.element_size() for t in unique.values()),
                                 "shape": list(value.shape) if hasattr(value, "shape") else []}
        local = getattr(value, "local", None)
        if local is not None:
            register(local, label + ".local")

    for layer, w in enumerate(list(fd.W.layers) + list(fd.W.mtp)):
        for attr, value in vars(w).items():
            register(value, f"L{layer}.{attr}")
    for attr in ("head", "draft_head", "markov_head_bf16"):
        register(getattr(fd, attr), attr)

    def replace(obj, attr, value):
        local = attr in vars(obj)
        old = vars(obj).get(attr) if local else None
        saved.append((obj, attr, local, old))
        setattr(obj, attr, value)

    def name(kind, weight=None):
        tag = weights.get(id(weight), type(weight).__name__) if weight is not None else ""
        base = "/".join(path) + "/" + kind + ("/" + tag if tag else "")
        serial[base] = serial.get(base, 0) + 1
        return base + f"#{serial[base]}"

    def metadata(kind, weight=None):
        return {"phase": path[0] if path[0].startswith("draft") else "verify",
                "scope": "/".join(path), "kind": kind,
                "parent": parents[-1] if parents else None,
                "weight": weights.get(id(weight)) if weight is not None else None,
                **weight_info.get(id(weight), {})}

    def operation(kind, call, weight=None, **kwargs):
        # Scope paths can reset at layer/draft/head boundaries, so path prefixes
        # do not establish nesting. Retain the actual dispatch stack instead.
        op_name = name(kind, weight)
        meta = metadata(kind, weight)
        parents.append(op_name)
        try:
            return probe.operation(op_name, call, metadata=meta, **kwargs)
        finally:
            parents.pop()

    def wrap(obj, attr, kind, mode="timing"):
        if not hasattr(obj, attr):
            return
        original = getattr(obj, attr)

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            weight = args[1] if mode in ("weight", "tensor_weight", "head") and len(args) > 1 else None
            if weight is not None and id(weight) not in weight_info:
                register(weight, type(weight).__name__)
            pure = mode in ("weight", "tensor_weight", "head", "tensors", "einsum")
            collective = (weight is not None and
                          (hasattr(weight, "tp_linear") or hasattr(weight, "tp_logits")))
            if mode == "tensor_weight" and not isinstance(weight, torch.Tensor):
                pure = False
            if kwargs.get("out") is not None:
                pure = False
            positions = []
            if pure and not collective:
                if mode in ("weight", "tensor_weight", "head"):
                    positions = [0] if args and isinstance(args[0], torch.Tensor) else []
                else:
                    positions = [i for i, value in enumerate(args) if isinstance(value, torch.Tensor)
                                 and id(value) not in weights]
                # Only positional tensor arguments are replayed by this adapter.
                if any(isinstance(v, torch.Tensor) for v in kwargs.values()):
                    pure = False
            inputs = tuple(args[i] for i in positions) if pure and not collective else ()

            # Retain only immutable weights/scalars, never the graph's live
            # activation storage owners through the isolation callback.
            template = [None if i in positions else value for i, value in enumerate(args)]

            def replay(*private):
                replay_args = list(template)
                for i, value in zip(positions, private):
                    replay_args[i] = value
                return original(*replay_args, **kwargs)

            extra = {"record_stream": args[0]} if mode == "stream" else {}
            return operation(kind, lambda: original(*args, **kwargs), weight,
                             inputs=inputs, replay=replay if pure and not collective else None,
                             weight_refs=(weight,) if weight is not None else (), pure=pure,
                             stateful=not pure and not collective,
                             collective=collective or mode == "collective", **extra)

        replace(obj, attr, wrapped)

    def scope(attr, label):
        original = getattr(fd, attr)

        @functools.wraps(original)
        def scoped(*args, **kwargs):
            previous = path[:]
            if attr in ("_layer_a", "_layer_b"):
                path[:] = ["verify", f"L{args[0]}", label]
            elif attr == "_draft":
                path[:] = ["draft.greedy" if args[0] else "draft.sampled"]
            elif attr == "_final":
                path[:] = ["verify", "head"]
            else:
                path.append(label)
            try:
                return operation("span", lambda: original(*args, **kwargs), stateful=True)
            finally:
                path[:] = previous

        replace(fd, attr, scoped)

    probe.begin_capture(key)
    success = False
    try:
        for attr, label in (("_layer_a", "attention_router"), ("_layer_b", "experts"),
                            ("_attention", "attention"), ("_compressed", "compression"),
                            ("_indexer", "indexer"), ("_hc_mixes", "hc"),
                            ("_hc_pre_rn", "hc_norm"), ("_moe_merged", "merged_moe"),
                            ("_shared_ffn", "shared"), ("_routed_experts", "routed"),
                            ("_final", "head"), ("_draft", "draft")):
            scope(attr, label)
        for obj in (R, FP8):
            wrap(obj, "fp8_linear", "dense.fp8", "weight")
            wrap(obj, "fp8_grouped_linear", "dense.grouped_fp8", "weight")
        wrap(R, "qlinear", "projection", "timing")
        wrap(R, "mm", "dense.mm", "tensor_weight")
        wrap(R, "wo_a_proj", "projection.grouped")
        wrap(R, "head_logits", "head", "head")
        # R.rmsnorm can dispatch to LeanOps' persistent padded scratch.
        wrap(R, "rmsnorm", "norm")
        wrap(R, "engram_forward", "engram")
        wrap(D, "_lin", "markov", "tensor_weight")
        wrap(D, "_hc_post_fused", "residual")
        wrap(torch, "einsum", "attention.product", "einsum")
        wrap(comm, "all_gather_fast", "communication", "collective")
        wrap(D.l2pf, "touch", "prefetch", "stream")
        if fd.lean is not None:
            for attr in ("rmsnorm", "hc_pre_rn", "hc_mixes", "router", "swiglu",
                         "route_prep", "block_null", "keys_f32"):
                wrap(fd.lean, attr, "lean." + attr)
            wrap(fd.lean, "rope", "rope", "tensors")
            wrap(fd.lean, "attn_probs", "attention.softmax", "tensors")
        wrap(fd.m, "moe_fn", "experts")
        cuda = sys.modules.get("fp4_moe_cuda")
        if cuda is not None:
            for attr in ("up", "down", "build_routing_small"):
                wrap(cuda, attr, "experts." + attr)
        yield
        success = True
    finally:
        for obj, attr, local, old in reversed(saved):
            if local:
                setattr(obj, attr, old)
            else:
                delattr(obj, attr)
        probe.end_capture(success=success)
