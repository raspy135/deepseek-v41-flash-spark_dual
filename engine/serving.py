"""Two-lane cooperative execution over one shared expert arena.

Only the scheduler thread may call this module. Rank zero sends every action to
the worker before executing it locally, including cancellation. No request thread
may issue a collective or mutate GPU state.
"""

import copy
import threading

from engine.decode_events import VerifyStep


class ScheduledPeer:
    """Request termination is an explicit scheduler action on BOTH ranks."""

    def __init__(self, peer):
        self.peer = peer

    def __getattr__(self, name):
        return getattr(self.peer, name)

    def control(self, keep_going):
        return keep_going

    def release_peer(self):
        pass


def make_engine_lane(engine):
    import torch
    from engine.batch2 import clone_lane
    from engine.v41_engine import prefetch_rows
    from engine.model import PRUNE_UNIT_REQUEST

    lane = copy.copy(engine)
    lane.lock = threading.Lock()
    lane.fast = clone_lane(lane, engine.fast, copy_cache=False)
    lane.model, lane.caches = lane.fast.m, lane.fast.c
    lane.tables = {key: copy.copy(table) for key, table in engine.tables.items()}
    for table in lane.tables.values():
        table.stats = dict(table.stats)
    lane.model.engram_rows = lambda layer, hashes: lane.tables[layer].rows(hashes)
    lane.model.engram_prefetch = lambda hashes: prefetch_rows(
        lane.tables, lane.eg_pool, hashes, lane.args.engram_layer_ids)
    # One demand database, but private request/miss counters and captured addresses.
    if engine.model._want_counts is not None:
        lane.model._want_counts = engine.model._want_counts
        lane.model._want_mass = engine.model._want_mass
        if not PRUNE_UNIT_REQUEST:
            lane.model._rec_counts = lane.model._want_counts
    lane._prefix_cache, lane._prefix_snapshots = None, {}
    lane._token_types = lane._images = None
    lane._requests_since_plan = 0
    lane._blk = torch.empty_like(engine._blk)
    lane._vout = torch.empty_like(engine._vout)
    lane._vhost = torch.empty_like(engine._vhost, pin_memory=True)
    lane.last_stats = {}
    return lane


class DecodeRuntime:
    """Deterministic action executor shared by the HTTP scheduler and worker."""

    def __init__(self, engine):
        if (not engine.ep.active or engine.ep.world != 2
                or engine.fast is None or engine.fast.lut is None or not engine.spec
                or not getattr(engine.ep, 'tensor_parallel', False)):
            raise ValueError('concurrency=2 requires TP, speculative decode and a resident expert LUT')
        if engine.replica_slots:
            raise ValueError('concurrency=2 does not support prefill replicas')
        from engine.batch2 import Batch2FastDecoder
        engine._init_prefix_disk()
        second = make_engine_lane(engine)
        self.engines = (engine, second)
        self.peer = engine.ep
        for lane in self.engines:
            lane.ep = ScheduledPeer(self.peer)
            lane._cooperative_decode = True
        self.batch = Batch2FastDecoder(engine.fast, second.fast)
        self.generators = {}
        self.events = {}
        self.answers = {}
        self.rng = {}
        self.store_stats = [dict(engine.store.stats), dict(engine.store.stats)]
        self.prefix_stats = [{}, {}]
        self.stats = {}
        self.batch_steps = self.single_steps = 0

    def _activate(self, index):
        lane = self.engines[index]
        primary = self.engines[0]
        lane.expert_generation = primary.expert_generation
        lane._prefix_route = primary._prefix_route
        lane._requests_since_plan = primary._requests_since_plan
        lane.store.stats = self.store_stats[index]
        # One disk index/writer/budget per rank, not one per request. Saves stage
        # their data before yielding and lookups join the preceding disk write.
        if lane.prefix_disk is not None:
            lane.prefix_disk.engine = lane
            lane.prefix_disk.stats = self.prefix_stats[index]
        return lane

    def _advance(self, index, closing=False):
        import torch
        lane = self._activate(index)
        if index in self.rng and self.rng[index] is not None:
            cpu, gpu = self.rng[index]
            torch.set_rng_state(cpu)
            torch.cuda.set_rng_state(gpu, lane.device)
        try:
            gen = self.generators[index]
            if closing:
                gen.close()
                event = None
            else:
                event = gen.send(self.answers.pop(index, None))
        except StopIteration:
            event = None
        finally:
            if lane.prefix_disk is not None:
                self.prefix_stats[index] = lane.prefix_disk.stats
            if index in self.rng:
                self.rng[index] = (torch.get_rng_state(), torch.cuda.get_rng_state(lane.device))
            primary = self.engines[0]
            primary._requests_since_plan = lane._requests_since_plan
            # Inline adaptation happens while every other lane is suspended.
            generation = max(e.expert_generation for e in self.engines)
            for e in self.engines:
                e.expert_generation = generation
                if lane.prefix_disk is not None:
                    e._prefix_route = lane._prefix_route
        self.events[index] = event
        if event is None:
            self.generators.pop(index)
            self.rng.pop(index, None)
            self.answers.pop(index, None)
            self.stats[index] = dict(lane.stats())
            self.stats[index]['scheduler'] = {
                'max_concurrency': 2, 'batch_steps_total': self.batch_steps,
                'single_steps_total': self.single_steps,
            }
        return event

    def execute(self, action):
        """All actions, including close, must be issued identically on every rank."""
        op = action['op']
        if op == 'start':
            index = action['lane']
            if index in self.generators:
                raise RuntimeError('scheduler reused an occupied lane')
            lane = self._activate(index)
            vl = action.get('vl')
            lane.set_vl_inputs(vl['token_types'].to(lane.device), vl['images']) if vl else lane.set_vl_inputs(None, None)
            if action['kwargs'].get('temperature', 1) > 0:
                self.rng[index] = None
            self.generators[index] = lane.generate(action['prompt_ids'], **action['kwargs'])
            return self._advance(index)
        if op in ('advance', 'close'):
            return self._advance(action['lane'], closing=op == 'close')
        if op == 'verify':
            indices = action['lanes']
            if not indices or any(not isinstance(self.events[i], VerifyStep) for i in indices):
                raise RuntimeError('verify issued outside a verify boundary')
            if indices == [0, 1]:
                a, b = (self.events[i] for i in indices)
                results = self.batch.step((a.block, b.block), (a.position, b.position), (a.rows, b.rows))
                self.batch_steps += 1
                for i in indices:
                    lane = self.engines[i]
                    self.store_stats[i]['hits'] += lane.args.n_layers * lane.fast.route_idx.numel()
            elif len(indices) == 1:
                lane = self._activate(indices[0])
                step = self.events[indices[0]]
                results = [lane.fast.step(step.block, step.position, step.rows)]
                self.single_steps += 1
            else:
                raise ValueError('invalid verification lane order')
            self.answers.update(zip(indices, results))
            return None
        raise ValueError(f'unknown scheduler action: {op}')
