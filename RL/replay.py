#!/usr/bin/env python
"""
Prioritized Experience Replay - CORRECTED VERSION

Fixes vs the previous implementation:
  - Data and priorities now share ONE ring index (self.pos). The old version
    stored data in a deque(maxlen) but priorities in a fixed ring array; once
    the buffer filled, deque.append() shifted every element's positional index
    while prios[pos] overwrote cyclically, so priorities/IS-weights became
    permanently attached to the WRONG transitions.
  - Standard PER sampling with replacement (replace=True).
  - max priority is computed only over written slots.
"""

import numpy as np


class PrioritizedReplay:
    def __init__(self, capacity: int = 100_000, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.data = [None] * capacity
        self.prios = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def __len__(self):
        return self.size

    def add(self, *transition):
        # New transition gets the current max priority so it is sampled at least once.
        max_prio = self.prios[:self.size].max() if self.size > 0 else 1.0
        self.data[self.pos] = transition
        self.prios[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float = 0.4):
        priorities = self.prios[:self.size]
        probs = priorities ** self.alpha
        probs = probs / probs.sum()

        idx = np.random.choice(self.size, batch_size, p=probs, replace=True)
        samples = [self.data[i] for i in idx]

        # importance-sampling weights
        weights = (self.size * probs[idx]) ** (-beta)
        weights /= weights.max()
        return samples, idx, weights.astype(np.float32)

    def update_priorities(self, indices, td_errors):
        td_errors = np.asarray(td_errors).flatten()
        self.prios[indices] = np.abs(td_errors) + 1e-6
