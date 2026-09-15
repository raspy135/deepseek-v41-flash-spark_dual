"""HTTP-to-GPU queue. All GPU work and peer messages have one owner thread."""

import queue
import threading

from engine.decode_events import VerifyStep, event_signature


class RequestStream:
    def __init__(self, payload):
        self.payload = payload
        self.output = queue.Queue()
        self.cancelled = threading.Event()
        self.finished = threading.Event()
        self.stats = {}
        self.exhausted = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.exhausted:
            raise StopIteration
        value = self.output.get()
        if isinstance(value, BaseException):
            self.exhausted = True
            raise value
        if value is None:
            self.exhausted = True
            raise StopIteration
        return value

    def finish(self, stats=None, error=None):
        self.stats = stats or {}
        self.output.put(error if error is not None else None)
        self.finished.set()

    def close(self):
        self.cancelled.set()
        # Do not return a request's stats or reuse its lane until BOTH ranks have
        # closed its generator. No GPU operations occur on this HTTP thread.
        self.finished.wait()


class Scheduler:
    def __init__(self, state, runtime):
        self.state, self.runtime = state, runtime
        self.pending = queue.Queue(maxsize=32)
        self.failure = None
        self.stopping = threading.Event()
        self.submission_lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, name='decode-scheduler', daemon=True)
        self.thread.start()

    def submit(self, payload):
        with self.submission_lock:
            if self.failure or self.stopping.is_set():
                raise RuntimeError(self.failure or 'server is shutting down')
            job = RequestStream(payload)
            self.pending.put_nowait(job)
        return job

    def stop(self):
        self.stopping.set()

    def _dispatch(self, action):
        self.state.ep.broadcast_request({'cmd': 'schedule', 'action': action})
        event = self.runtime.execute(action)
        # Catch completion/step-count divergence before the next action can put
        # one rank into a model collective and the other into the command queue.
        tag = event_signature(event)
        tags = self.state.ep.gather_objects(tag)
        if any(t != tag for t in tags):
            raise RuntimeError(f'concurrent decode ranks diverged: {tags}')
        return event

    def _run(self):
        active = {}
        try:
            while True:
                try:
                    first = self.pending.get(timeout=0.1)
                except queue.Empty:
                    if self.stopping.is_set():
                        return
                    continue
                with self.state.lock:
                    waiting = [first]
                    while waiting or active:
                        if self.stopping.is_set():
                            for job in waiting + list(active.values()):
                                job.cancelled.set()
                        while len(waiting) + len(active) < 2:
                            try:
                                waiting.append(self.pending.get_nowait())
                            except queue.Empty:
                                break
                        for job in waiting:
                            if job.cancelled.is_set() or self.stopping.is_set():
                                job.finish()
                                continue
                            lane = next(i for i in range(2) if i not in active)
                            active[lane] = job
                            event = self._dispatch({'op': 'start', 'lane': lane, **job.payload})
                            if event is not None and not isinstance(event, VerifyStep):
                                job.output.put(event)
                        waiting = []
                        for lane, job in list(active.items()):
                            event = self.runtime.events[lane]
                            if job.cancelled.is_set() and lane in self.runtime.generators:
                                event = self._dispatch({'op': 'close', 'lane': lane})
                            elif event is not None and not isinstance(event, VerifyStep):
                                event = self._dispatch({'op': 'advance', 'lane': lane})
                                if event is not None and not isinstance(event, VerifyStep):
                                    job.output.put(event)
                            if event is None:
                                job.finish(self.runtime.stats.get(lane))
                                del active[lane]
                                # Both ranks are now waiting for an action. All
                                # other requests are frozen, so adaptation is safe.
                                self.state.maintain_experts()
                        ready = sorted(i for i in active if isinstance(self.runtime.events[i], VerifyStep))
                        if ready:
                            self._dispatch({'op': 'verify', 'lanes': ready})
                            for lane in ready:
                                event = self._dispatch({'op': 'advance', 'lane': lane})
                                if event is not None and not isinstance(event, VerifyStep):
                                    active[lane].output.put(event)
        except BaseException as exc:
            with self.submission_lock:
                self.failure = f'{type(exc).__name__}: {exc}'
            self.state.ep_fault = self.failure
            for job in active.values():
                job.finish(error=RuntimeError(self.failure))
            for job in waiting if 'waiting' in locals() else []:
                if not job.finished.is_set():
                    job.finish(error=RuntimeError(self.failure))
            while True:
                try:
                    self.pending.get_nowait().finish(error=RuntimeError(self.failure))
                except queue.Empty:
                    break
