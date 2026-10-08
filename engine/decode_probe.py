"""Opt-in, loaded-process decode timing and private-input isolation.

The caller owns idleness, TP command coordination, and the CUDA graphs containing
the hooks.  In particular, evict those graphs before releasing this controller:
disarming Python cannot remove copy/event nodes already captured in a graph.

Live timings are the *last replay* of each capture key/phase, not a generation
average.  Isolation is deliberately restricted to explicitly pure local leaves.
Every mutable tensor argument must be supplied in ``inputs``; ``replay`` may close
over immutable live model weights, but never a mutable cache, arena, collective,
output buffer, or generation state.  Weights are retained, never copied.  Results
contain layouts and latency only, with no prompts or activation values.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from fnmatch import fnmatchcase
from statistics import median
from typing import Any, Callable


_MIB = 1 << 20
_MAX_TIMING_RECORDS = 16384


@dataclass(frozen=True)
class ProbeConfig:
    action: str = "arm"
    names: tuple[str, ...] = ("*dense*", "*head*", "*markov*")
    max_cases: int = 256
    snapshot_budget_mb: int = 64
    warmup: int = 3
    repeats: int = 6
    calls: int = 16
    flush_mb: int = 64

    def __post_init__(self):
        if self.action not in ("arm", "run", "stop"):
            raise ValueError("action must be arm, run, or stop")
        if not isinstance(self.names, (tuple, list)) or not 1 <= len(self.names) <= 64:
            raise ValueError("names must be a nonempty list of at most 64 patterns")
        if any(not isinstance(s, str) or not s or len(s) > 128 for s in self.names):
            raise ValueError("each name pattern must contain 1..128 characters")
        object.__setattr__(self, "names", tuple(self.names))
        for name, lo, hi in (("max_cases", 1, 1024), ("snapshot_budget_mb", 1, 256),
                             ("warmup", 0, 16), ("repeats", 1, 16),
                             ("calls", 1, 64), ("flush_mb", 0, 256)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
                raise ValueError(f"{name} must be an integer in {lo}..{hi}")

    @classmethod
    def from_dict(cls, body):
        if not isinstance(body, dict):
            raise TypeError("probe options must be a JSON object")
        extra = set(body) - set(cls.__dataclass_fields__)
        if extra:
            raise ValueError("unknown probe options: " + ", ".join(sorted(extra)))
        return cls(**body)


@dataclass
class _Case:
    name: str
    key: Any
    phase: str
    metadata: dict
    start: Any
    end: Any
    inputs_meta: list = field(default_factory=list)
    outputs_meta: list = field(default_factory=list)
    private_inputs: tuple | None = None
    expected: Any = None
    replay: Callable | None = None
    weight_refs: tuple = ()
    snapshot_bytes: int = 0
    replay_eligible: bool = False
    isolation_reserved: bool = False
    skip_reason: str | None = None
    isolation: dict | None = None


class _SnapshotError(Exception):
    pass


class DecodeProbe:
    """A controller whose hooks do no work unless armed inside a capture session.

    ``event_factory`` receives ``enable_timing=True, external=True``.  The latter
    is essential: internal capture events need not retain replay timestamps.
    ``torch_module`` and ``synchronize`` allow CPU tests without a CUDA context.
    They are not alternate production timing backends.
    """

    def __init__(self, config: ProbeConfig | None = None, device="cuda", *,
                 event_factory=None, synchronize=None, torch_module=None,
                 isolation_stream=None):
        if torch_module is None:
            import torch as torch_module
        self.torch = torch_module
        self.config = config or ProbeConfig()
        if not isinstance(self.config, ProbeConfig):
            raise TypeError("config must be ProbeConfig")
        self.device = self.torch.device(device)
        self._event_factory = event_factory or self.torch.cuda.Event
        self._synchronize = synchronize or (lambda: self.torch.cuda.synchronize(self.device))
        self._armed = False
        self._capture_key = None
        self._capturing = False
        self._records: dict[Any, list[_Case]] = {}
        self._replays: dict[tuple[Any, str], int] = {}
        self._snapshot_bytes = 0
        self._snapshot_cases = 0
        self._timing_records = 0
        self._dropped_timings = 0
        self._profile_running = False
        self._last_report = None
        # Allocating diagnostic snapshots inside a shared CUDA graph pool is
        # unsafe: later graph variants can reuse that pool's address ranges and
        # overwrite an older phase's retained snapshots. Python tensor refs do
        # not establish external ownership for capture-time allocations.
        # One normal-allocator slab is created before entering capture; captured
        # operations only form views and copy into disjoint persistent slices.
        self._snapshot_arena = None
        self._arena_cursor = 0
        # cuBLAS retains a large workspace per stream in this build. Creating one
        # stream per leaf would turn a bounded diagnostic into hundreds of MiB
        # (or GiB) of retained library workspace. Serving passes its existing
        # capture warmup stream; standalone probes create at most one lazily.
        self._isolation_stream = isolation_stream

    @property
    def armed(self):
        return self._armed

    @property
    def active(self):
        return self._armed

    def arm(self):
        if self._profile_running or self._capturing:
            raise RuntimeError("cannot arm during capture or isolation")
        self._armed = True
        return self.status()

    def _remove_key(self, key):
        records = self._records.pop(key, ())
        self._snapshot_bytes -= sum(c.snapshot_bytes for c in records)
        self._snapshot_cases -= sum(c.isolation_reserved for c in records)
        self._timing_records -= len(records)
        for marker in [m for m in self._replays if m[0] == key]:
            del self._replays[marker]

    def begin_capture(self, key):
        if self._capturing or self._profile_running:
            raise RuntimeError("probe capture already active or isolation running")
        if not self._armed:
            return False
        if self._snapshot_arena is None:
            if self.device.type == "cuda" and self.torch.cuda.is_current_stream_capturing():
                raise RuntimeError("begin_capture must allocate snapshots outside CUDA capture")
            self._snapshot_arena = self.torch.empty(self.config.snapshot_budget_mb * _MIB,
                                                   dtype=self.torch.uint8, device=self.device)
        # Caller must have evicted any older graph for this key first. Old CUDA
        # graphs retain their allocations independently of our Python records.
        self._remove_key(key)
        self._records[key] = []
        self._capture_key = key
        self._capturing = True
        self._last_report = None
        return True

    def end_capture(self, success=True):
        if not self._capturing:
            return
        if not success:
            self._remove_key(self._capture_key)
        self._capture_key = None
        self._capturing = False

    def abort_capture(self):
        self.end_capture(success=False)

    def forget_capture(self, key):
        """Release an already-evicted graph key's private buffers and event refs."""
        if self._capturing and key == self._capture_key:
            raise RuntimeError("cannot forget the active capture")
        self._remove_key(key)

    def clear_captures(self):
        """Drop records after the caller synchronizes and evicts hooked graphs.

        Keep the arm state/configuration so the next pool capture can be probed.
        """
        if self._capturing or self._profile_running:
            raise RuntimeError("cannot clear during capture or isolation")
        self._records.clear()
        self._replays.clear()
        self._snapshot_bytes = self._snapshot_cases = self._timing_records = 0
        self._arena_cursor = 0
        self._dropped_timings = 0
        self._last_report = None

    def disarm(self, clear=True):
        if self._profile_running:
            raise RuntimeError("cannot disarm during isolation")
        self.end_capture(success=False)
        self._armed = False
        if clear:
            self.clear_captures()
            self._snapshot_arena = None
        return self.status()

    def _selected(self, name, config=None):
        return any(fnmatchcase(name, pattern) for pattern in (config or self.config).names)

    def _tensor_metadata(self, tensor):
        return dict(shape=list(tensor.shape), stride=list(tensor.stride()),
                    dtype=str(tensor.dtype), device=str(tensor.device))

    def _tensor_leaves(self, tree):
        if isinstance(tree, self.torch.Tensor):
            return [tree]
        if isinstance(tree, (tuple, list)):
            return [t for value in tree for t in self._tensor_leaves(value)]
        if isinstance(tree, dict):
            return [t for value in tree.values() for t in self._tensor_leaves(value)]
        return []

    def _layouts(self, tree):
        return [self._tensor_metadata(t) for t in self._tensor_leaves(tree)]

    @staticmethod
    def _safe_metadata(metadata):
        # Do not accidentally return arbitrary call arguments, prompt strings, or
        # tensor reprs. All text below describes operator identity/implementation.
        allowed = {"family", "layer", "width", "rank", "k", "n", "m", "kernel",
                   "dtype", "phase", "bytes", "category", "strides", "shape",
                   "weight_bytes", "implementation", "parent", "path", "fieldpath",
                   "scope", "kind", "weight"}
        def safe(value):
            if value is None or isinstance(value, (bool, int, float)):
                return value
            if isinstance(value, str):
                return value[:128]
            if isinstance(value, (tuple, list)) and len(value) <= 32:
                values = [safe(v) for v in value]
                if all(v is not None for v in values):
                    return values
            return None
        # An explicit null parent identifies a recorded root. Omitting it would
        # make dashboard readers fall back to legacy scope/dispatch inference.
        return {k: safe(v) for k, v in (metadata or {}).items()
                if k in allowed and (safe(v) is not None or (k == "parent" and v is None))}

    def _storage_bytes(self, tensor):
        if tensor.layout != self.torch.strided:
            raise _SnapshotError("unsupported_tensor_layout")
        if tensor.numel() == 0:
            return 0
        dimensions = [(int(s), int(n)) for s, n in zip(tensor.stride(), tensor.shape) if n > 1]
        # Conservative, exact check for regular nonoverlapping strided layouts.
        # Expanded/overlapping views require an alias-aware replay API; flattening
        # them would change layout-specialized dispatch and is not a valid test.
        extent = 1
        for stride, size in sorted(dimensions):
            if stride <= 0 or stride < extent:
                raise _SnapshotError("overlapping_or_negative_stride")
            extent += (size - 1) * stride
        span = 1 + sum((int(n) - 1) * int(s) for n, s in zip(tensor.shape, tensor.stride()))
        return span * tensor.element_size()

    def _snapshot(self, tree, limit, *, persistent=False):
        def validate(value):
            if isinstance(value, self.torch.Tensor):
                return
            if isinstance(value, (tuple, list)):
                for v in value:
                    validate(v)
                return
            if isinstance(value, dict) and all(isinstance(k, (str, int)) for k in value):
                for v in value.values():
                    validate(v)
                return
            if value is None or isinstance(value, (bool, int, float, str)):
                return
            raise _SnapshotError("unsupported_mutable_argument")
        validate(tree)
        tensors = self._tensor_leaves(tree)
        unique = {id(t): t for t in tensors}
        spans = {identity: self._storage_bytes(t) for identity, t in unique.items()}
        total = sum(spans.values())
        if total > limit:
            raise _SnapshotError("snapshot_budget")
        # Distinct views of one mutable storage cannot be cloned independently:
        # doing so removes aliasing. Identical tensor arguments retain aliasing.
        intervals = []
        for identity, t in unique.items():
            if t.numel():
                start = t.data_ptr()
                end = start + spans[identity]
                if any(device == str(t.device) and start < old_end and old_start < end
                       for device, old_start, old_end in intervals):
                    raise _SnapshotError("aliased_mutable_views")
                intervals.append((str(t.device), start, end))
        plans = {}
        cursor = self._arena_cursor
        if persistent:
            if self._snapshot_arena is None:
                raise RuntimeError("persistent snapshots require begin_capture outside CUDA capture")
            for identity, t in unique.items():
                if (t.device.type != self.device.type or
                        self.device.index is not None and t.device.index != self.device.index):
                    raise _SnapshotError("snapshot_device_mismatch")
                cursor = (cursor + 255) & ~255
                plans[identity] = cursor
                cursor += spans[identity]
            if cursor > self._snapshot_arena.numel():
                raise _SnapshotError("snapshot_arena_budget")
        copies = {}
        for identity, t in unique.items():
            if persistent:
                private = self._snapshot_arena.view(t.dtype).as_strided(
                    tuple(t.shape), tuple(t.stride()), plans[identity] // t.element_size())
            else:
                # Isolation's working copies are made at idle outside graph
                # capture and have ordinary allocator ownership. They must not
                # consume or overwrite the persistent baseline slab.
                private = self.torch.empty_strided(tuple(t.shape), tuple(t.stride()),
                                                   dtype=t.dtype, device=t.device)
            # A diagnostic must not retain a source activation through an
            # autograd CopyBackwards edge if a caller forgot inference_mode.
            with self.torch.no_grad():
                private.copy_(t)
            copies[identity] = private
        if persistent:
            self._arena_cursor = cursor
        def rebuild(value):
            if isinstance(value, self.torch.Tensor):
                return copies[id(value)]
            if isinstance(value, tuple):
                return tuple(rebuild(v) for v in value)
            if isinstance(value, list):
                return [rebuild(v) for v in value]
            if isinstance(value, dict):
                if not all(isinstance(k, (str, int)) for k in value):
                    raise _SnapshotError("unsupported_input_mapping")
                return {k: rebuild(v) for k, v in value.items()}
            if value is None or isinstance(value, (bool, int, float, str)):
                return value
            raise _SnapshotError("unsupported_mutable_argument")
        return rebuild(tree), total

    def operation(self, name, call, *, inputs=(), replay=None, weight_refs=(),
                  metadata=None, pure=False, stateful=False, collective=False,
                  record_stream=None):
        """Execute ``call`` exactly once, optionally capturing diagnostic nodes.

        Snapshots are device-to-device copies immediately outside the timed span.
        They are graph nodes, so they see actual replay inputs/outputs, not stale
        capture-time values. Nested timing spans overlap and must not be summed.
        """
        if not self._armed or not self._capturing:
            return call()
        if self._timing_records >= _MAX_TIMING_RECORDS:
            self._dropped_timings += 1
            return call()
        metadata = self._safe_metadata(metadata)
        phase = metadata.get("phase", "verify")
        case = _Case(str(name), self._capture_key, phase, metadata,
                     self._event_factory(enable_timing=True, external=True),
                     self._event_factory(enable_timing=True, external=True))
        case.inputs_meta = self._layouts(inputs)
        if collective:
            case.skip_reason = "collective_timing_only"
        elif stateful:
            case.skip_reason = "stateful_timing_only"
        elif not pure:
            case.skip_reason = "not_explicitly_pure"
        elif replay is None:
            case.skip_reason = "no_private_replay_callback"
        elif not self._selected(case.name):
            case.skip_reason = "not_selected"
        elif self._snapshot_cases >= self.config.max_cases:
            case.skip_reason = "case_limit"
        else:
            try:
                case.private_inputs, case.snapshot_bytes = self._snapshot(
                    tuple(inputs), self.config.snapshot_budget_mb * _MIB - self._snapshot_bytes,
                    persistent=True)
                # Reserve before calling a possibly nested operation. A later
                # output rejection cannot erase already-captured copy nodes.
                self._snapshot_bytes += case.snapshot_bytes
                self._snapshot_cases += 1
                case.isolation_reserved = True
            except _SnapshotError as exc:
                case.skip_reason = str(exc)
        self._records[self._capture_key].append(case)
        self._timing_records += 1
        try:
            if record_stream is None:
                case.start.record()
            else:
                case.start.record(record_stream)
            result = call()
            if record_stream is None:
                case.end.record()
            else:
                case.end.record(record_stream)
            case.outputs_meta = self._layouts(result)
            if case.skip_reason is None:
                if not case.outputs_meta:
                    case.skip_reason = "no_tensor_output_for_exactness"
                else:
                    try:
                        case.expected, output_bytes = self._snapshot(
                            result, self.config.snapshot_budget_mb * _MIB -
                            self._snapshot_bytes, persistent=True)
                        case.snapshot_bytes += output_bytes
                        self._snapshot_bytes += output_bytes
                    except _SnapshotError as exc:
                        case.skip_reason = str(exc)
            if case.skip_reason is None:
                case.replay = replay
                case.weight_refs = tuple(weight_refs)
                case.replay_eligible = True
            else:
                if case.isolation_reserved:
                    self._snapshot_cases -= 1
                    case.isolation_reserved = False
                # Retain/charge any input-only snapshot until its graph is gone.
                # The graph allocator may otherwise keep more live storage than
                # our advertised snapshot budget after repeated rejections.
                case.expected = None
            return result
        except BaseException:
            # Preserve the production exception, and never report a partial span.
            self.abort_capture()
            raise

    def mark_replay(self, key, phase="verify"):
        """Mark only the phase actually replayed, after its last segment queues.

        A shared draft graph can belong to the first captured verify key, and
        greedy/sampled alternatives must use distinct exact phase names.
        """
        if key in self._records:
            marker = (key, str(phase))
            self._replays[marker] = self._replays.get(marker, 0) + 1

    @staticmethod
    def _report_key(key):
        if isinstance(key, (tuple, list)):
            return [DecodeProbe._report_key(v) for v in key]
        if key is None or isinstance(key, (bool, int, float)):
            return key
        if isinstance(key, str):
            return key[:96]
        return type(key).__name__

    def status(self):
        return dict(status=("isolating" if self._profile_running else "capturing" if self._capturing
                            else "armed" if self._armed else "stopped"),
                    config=asdict(self.config),
                    coverage=dict(timing_cases=self._timing_records,
                                  isolation_cases=self._snapshot_cases,
                                  snapshot_bytes=self._snapshot_bytes,
                                  snapshot_budget_bytes=self.config.snapshot_budget_mb * _MIB,
                                  snapshot_reserved_bytes=(self._snapshot_arena.numel()
                                                           if self._snapshot_arena is not None else 0),
                                  snapshot_consumed_bytes=self._arena_cursor,
                                  dropped_timing_cases=self._dropped_timings),
                    cases=[])

    def collect_live(self):
        """Synchronize at an idle boundary and read last-replay event timestamps."""
        if self._capturing or self._profile_running:
            raise RuntimeError("live report requires idle outside capture/isolation")
        replayed = any(self._replays.values())
        if replayed:
            self._synchronize()
        report = self.status()
        report["measurement"] = "last replay per capture key and phase; nested spans overlap"
        report["instrumentation"] = ("event/copy nodes alter live graph scheduling and cache pressure; "
                                     "leaf copies excluded, child diagnostic work remains in enclosing spans")
        report["isolation_contract"] = "pure local operations; immutable retained weights; private mutable inputs"
        report["snapshot_allocation"] = "external persistent slab; disjoint aligned views; no shared-graph-pool allocation"
        report["coverage"]["replayed_timing_cases"] = 0
        report["coverage"]["unreplayed_timing_cases"] = 0
        for records in self._records.values():
            for case in records:
                count = self._replays.get((case.key, case.phase), 0)
                live_ms = None
                if count:
                    live_ms = float(case.start.elapsed_time(case.end))
                    report["coverage"]["replayed_timing_cases"] += 1
                else:
                    report["coverage"]["unreplayed_timing_cases"] += 1
                row = dict(name=case.name, key=self._report_key(case.key), phase=case.phase,
                           metadata=dict(case.metadata), inputs=case.inputs_meta,
                           outputs=case.outputs_meta, replay_eligible=case.replay_eligible,
                           isolation_ready=bool(case.replay_eligible and count),
                           skip_reason=case.skip_reason, snapshot_bytes=case.snapshot_bytes,
                           retained_weight_refs=len(case.weight_refs), live_ms=live_ms,
                           replay_count=count)
                if case.isolation is not None:
                    row["isolation"] = dict(case.isolation)
                    row["isolation"]["stale"] = (
                        case.isolation.get("measured_replay_count") != count)
                report["cases"].append(row)
        self._last_report = report
        return report

    def _bit_exact(self, expected, actual):
        if isinstance(expected, self.torch.Tensor):
            if not isinstance(actual, self.torch.Tensor):
                return False
            if expected.shape != actual.shape or expected.dtype != actual.dtype:
                return False
            # Byte equality catches signed zero and NaN payload differences. No
            # tensor values ever leave the device/process or enter the report.
            a = expected.contiguous().reshape(-1).view(self.torch.uint8)
            b = actual.contiguous().reshape(-1).view(self.torch.uint8)
            return bool(self.torch.equal(a, b))
        if isinstance(expected, (tuple, list)):
            return (type(expected) is type(actual) and len(expected) == len(actual) and
                    all(self._bit_exact(e, a) for e, a in zip(expected, actual)))
        if isinstance(expected, dict):
            return (isinstance(actual, dict) and expected.keys() == actual.keys() and
                    all(self._bit_exact(expected[k], actual[k]) for k in expected))
        return type(expected) is type(actual) and expected == actual

    def _isolate_case(self, case, config, flush):
        torch = self.torch
        # Second private copy means a replay with a private out= argument cannot
        # destroy the captured baseline. Never clone model weights.
        inputs, private_bytes = self._snapshot(case.private_inputs,
                                               self.config.snapshot_budget_mb * _MIB)
        if self._isolation_stream is None:
            self._isolation_stream = torch.cuda.Stream(device=self.device)
        stream = self._isolation_stream
        # Private copy_ kernels queued on the caller's current stream must finish
        # before a new isolation stream reads them. Stream creation is not a fence.
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream), torch.no_grad():
            result = None
            for _ in range(max(1, config.warmup)):
                result = case.replay(*inputs)
        self._synchronize()
        if not self._bit_exact(case.expected, result):
            return dict(exact=False, error="private_replay_output_mismatch", warm_ms=None,
                        cold_ms=None, private_input_bytes=private_bytes)

        def capture(cold):
            graph = torch.cuda.CUDAGraph()
            pairs = []
            last = None
            with torch.no_grad(), torch.cuda.graph(graph, stream=stream):
                for _ in range(config.calls):
                    if cold and flush is not None:
                        flush.add_(1)
                    start = self._event_factory(enable_timing=True, external=True)
                    end = self._event_factory(enable_timing=True, external=True)
                    start.record()
                    last = case.replay(*inputs)
                    end.record()
                    pairs.append((start, end))
            return graph, pairs, last

        warm_graph, warm_events, warm_output = capture(False)
        cold_graph, cold_events, cold_output = capture(True)
        self._synchronize()
        warm_samples, cold_samples = [], []
        # Balance order within this small screen; no full-model benchmark here.
        for repeat in range(config.repeats):
            order = ((warm_graph, warm_events, warm_samples),
                     (cold_graph, cold_events, cold_samples))
            if repeat & 1:
                order = order[::-1]
            for graph, events, samples in order:
                graph.replay()
                self._synchronize()
                samples.append(median(float(a.elapsed_time(b)) for a, b in events))
        exact = self._bit_exact(case.expected, warm_output) and self._bit_exact(case.expected, cold_output)
        # Callbacks must also retain shape/stride-specialized dispatch. Exactness
        # qualifies the outputs; it does not prove identical machine-code choices.
        return dict(exact=exact, error=None if exact else "captured_replay_output_mismatch",
                    warm_ms=median(warm_samples) if exact else None,
                    cold_ms=median(cold_samples) if exact else None,
                    warm_samples_ms=warm_samples if exact else [],
                    cold_samples_ms=cold_samples if exact else [],
                    calls=config.calls, repeats=config.repeats, warmup=config.warmup,
                    flush_bytes=config.flush_mb * _MIB, private_input_bytes=private_bytes,
                    output_layout=self._layouts(warm_output))

    def profile_idle(self, config: ProbeConfig | None = None):
        """Profile qualified local leaves after generation is idle on both ranks.

        A write/read cache-pressure kernel precedes every cold call, outside its
        event pair. This is controlled cache pressure, not a measurement of DRAM
        bandwidth or an exact emulation of whole-graph reuse. Warm/cold samples
        are reported separately and compared only after exact output validation.
        """
        if self._capturing or self._profile_running:
            raise RuntimeError("isolation requires an idle probe")
        if self.device.type != "cuda":
            raise RuntimeError("isolation timing requires CUDA")
        config = config or self.config
        if not isinstance(config, ProbeConfig):
            raise TypeError("profile config must be ProbeConfig")
        self._synchronize()
        self._profile_running = True
        flush = None
        try:
            if config.flush_mb:
                flush = self.torch.empty(config.flush_mb * _MIB, dtype=self.torch.uint8,
                                         device=self.device)
                flush.fill_(0)
            # A narrowed later run must not make old, differently configured
            # cases look like fresh measurements from the current request.
            for records in self._records.values():
                for case in records:
                    case.isolation = None
            for records in self._records.values():
                for case in records:
                    if (not case.replay_eligible or not self._selected(case.name, config) or
                            not self._replays.get((case.key, case.phase), 0)):
                        continue
                    try:
                        case.isolation = self._isolate_case(case, config, flush)
                    except Exception as exc:
                        # Do not serialize exception strings: a library can embed
                        # tensor values or source arguments in them.
                        case.isolation = dict(exact=False, error="isolation_failed",
                                              error_type=type(exc).__name__, warm_ms=None, cold_ms=None)
                    case.isolation["measured_replay_count"] = self._replays[(case.key, case.phase)]
        finally:
            self._profile_running = False
            del flush
        report = self.collect_live()
        report["status"] = "complete"
        report["profile_config"] = asdict(config)
        report["cache_pressure"] = "flush before each cold call; flush excluded from latency; no DRAM counters"
        report["coverage"]["isolated_cases"] = sum("isolation" in c for c in report["cases"])
        report["coverage"]["exact_isolated_cases"] = sum(
            c.get("isolation", {}).get("exact", False) for c in report["cases"])
        self._last_report = report
        return report
