#!/usr/bin/env python
"""
Prioritized Experience Replay - simple NumPy / Python implementation.
"""

import numpy as np
from collections import deque

class PrioritizedReplay:
    def __init__(self, capacity: int = 100_000, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = deque(maxlen=capacity)
        self.prios = np.zeros(capacity, dtype=np.float32)
        self.pos = 0

    def __len__(self):
        return len(self.buffer)

    def add(self, *transition):
        # The new transition gets the *max* priority so it will be sampled at least once.
        max_prio = self.prios.max() if self.buffer else 1.0
        self.buffer.append(transition)
        self.prios[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4):
        if len(self.buffer) == self.capacity:
            priorities = self.prios
        else:
            priorities = self.prios[:len(self.buffer)]

        probs = priorities ** self.alpha
        probs /= probs.sum()
        idx = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)

        samples = [self.buffer[i] for i in idx]
        # importance-sampling weights
        N = len(self.buffer)
        weights = (N * probs[idx]) ** (-beta)
        weights /= weights.max()
        return samples, idx, weights.astype(np.float32)

    def update_priorities(self, indices, td_errors):
        # add a tiny epsilon to keep priorities > 0
        self.prios[indices] = np.abs(td_errors) + 1e-6