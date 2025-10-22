#!/usr/bin/env python
"""
Dueling DQN - lightweight MPS-compatible PyTorch model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingDQN(nn.Module):
    def __init__(self, state_dim: int, action_dim: int = 5, hidden: int = 128):
        super().__init__()
        # shared encoder
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)

        # value stream
        self.val_fc = nn.Linear(hidden, hidden)
        self.val = nn.Linear(hidden, 1)

        # advantage stream
        self.adv_fc = nn.Linear(hidden, hidden)
        self.adv = nn.Linear(hidden, action_dim)

        # initialise
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        v = F.relu(self.val_fc(x))
        v = self.val(v)                              # (B,1)

        a = F.relu(self.adv_fc(x))
        a = self.adv(a)                              # (B,action_dim)
        a = a - a.mean(dim=1, keepdim=True)          # mean-centering for identifiability
        q = v + a
        return q