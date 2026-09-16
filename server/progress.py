"""CPU-only request progress; never inspects tensors or synchronizes the GPU."""

import time
import uuid


class DecodeProgress:
    def __init__(self, logger, prompt_tokens, max_tokens, thinking, think_end_id,
                 *, interval=10.0, clock=time.perf_counter):
        self.log, self.clock, self.interval = logger, clock, interval
        self.request_id = uuid.uuid4().hex[:8]
        self.started = clock()
        self.first = None
        self.last_time = None
        self.total = self.last_total = self.reasoning = 0
        self.thinking, self.think_end_id = thinking, think_end_id
        self.max_tokens = max_tokens
        logger.info('generation start: request=%s prompt=%d max_tokens=%d thinking=%s',
                    self.request_id, prompt_tokens, max_tokens, thinking)

    def update(self, burst):
        if not burst:
            return
        now = self.clock()
        self.total += len(burst)
        if self.thinking:
            for token in burst:
                if token == self.think_end_id:
                    self.thinking = False
                    break
                self.reasoning += 1
        if self.first is None:
            self.first = self.last_time = now
            self.last_total = self.total
            self.log.info('decode started: request=%s first_output=%.2fs generated=%d phase=%s',
                          self.request_id, now - self.started, self.total,
                          'thinking' if self.thinking else 'answer')
            return
        elapsed = now - self.last_time
        if elapsed < self.interval:
            return
        self.log.info('decode progress: request=%s phase=%s generated=%d/%d reasoning=%d '
                      'elapsed=%.1fs decode=%.1fs recent=%.2f tok/s',
                      self.request_id, 'thinking' if self.thinking else 'answer',
                      self.total, self.max_tokens, self.reasoning, now - self.started,
                      now - self.first, (self.total - self.last_total) / elapsed)
        self.last_time, self.last_total = now, self.total
