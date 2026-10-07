"""Read the live CPU arena directory without taking the generation lock."""
import time


def snapshot(engine, *, busy=False, fault=None, model='deepseek'):
    store = getattr(engine, 'store', None)
    args = getattr(engine, 'args', None)
    if store is None or args is None:
        return None
    layers, experts = int(args.n_layers), int(args.n_routed_experts)
    # These short list copies run under Python's GIL. Do not hold the serving
    # lock, read CUDA masks, or call config()/prune_miss_report() from this route.
    # The separate copies can straddle a transfer: this is a live observer,
    # not an atomic inference checkpoint or independent peer integrity audit.
    residents = list(store.lru.items())
    transients = list(store.transient_map.items())
    activity = getattr(store, 'activity', None)
    activity = activity.snapshot() if activity is not None else dict(sequence=0, active=[], recent=[])
    states = [[0]*experts for _ in range(layers)]
    slots = [[-1]*experts for _ in range(layers)]
    count = int(store.arena.slots)
    sectors = [None]*count

    def put(key, slot, state):
        L, e = key
        if 0 <= L < layers and 0 <= e < experts and 0 <= slot < count:
            old = sectors[slot]
            if old is not None and old[:2] != [L, e] and slots[old[0]][old[1]] == slot:
                states[old[0]][old[1]], slots[old[0]][old[1]] = 0, -1
            states[L][e], slots[L][e] = state, slot
            sectors[slot] = [L, e, state]

    for key, slot in transients:
        put(key, slot, 2)
    for key, slot in residents:
        put(key, slot, 1)
    for load in activity['active']:
        put((load['layer'], load['expert']), load['slot'], 3)
    flat = [s for row in states for s in row]
    last = getattr(engine, 'last_stats', {})
    miss = last.get('prune_miss_request') or {}
    ep = getattr(engine, 'ep', None)
    focus = getattr(engine, 'user_prompt', None)
    return dict(version=1, sampled_at=time.time(), model=model, busy=bool(busy), fault=bool(fault),
                rank=int(getattr(ep, 'rank', 0)), world_size=int(getattr(ep, 'world', 1)),
                tensor_parallel=bool(getattr(ep, 'tensor_parallel', False)),
                layers=layers, experts_per_layer=experts, total_experts=layers*experts,
                states=[''.join(map(str, row)) for row in states], slots=slots, sectors=sectors,
                state_names=['not_loaded', 'resident', 'transient', 'loading'],
                counts=dict(not_loaded=flat.count(0), resident=flat.count(1),
                            transient=flat.count(2), loading=flat.count(3)),
                resident_per_layer=[row.count(1) for row in states], arena_slots=count,
                null_slot=store.null_slot, transient_capacity=store.transient_slots,
                expert_shard_bytes=int(getattr(engine, 'expert_bytes', 0)),
                generation=int(getattr(engine, 'expert_generation', 0)),
                dynamic=bool(getattr(engine, 'dynamic_experts', False)),
                prompt_streaming=bool(getattr(engine, 'user_prompt_stream', False)),
                stream_load_cap=int(getattr(focus, 'max_loads', 0)),
                stream_loads_used=int(getattr(focus, 'loads_used', 0)),
                resident_loads_used=int(getattr(focus, 'resident_loads', 0)),
                last_request_miss_rate=miss.get('miss_rate'),
                last_request_cached_tokens=last.get('prefix_cached_tokens'),
                **activity)
