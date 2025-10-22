import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
from replay import PrioritizedReplay

# ---- Dueling-DQN Agent -------------------------------------------------------
class DuelingDQNAgent:
    """
    - Holds a `DuelingDQN` policy net + a target net.
    - Implements ε-greedy action selection (the `epsilon` attribute lives here).
    - Provides `act`, `decay_epsilon`, `replay`, `save`/`load`.
    - FIXED: Proper MPS autocast with dtype specification
    """
    def __init__(self,
                 state_dim: int,
                 action_dim: int = 5,
                 hidden: int = 128,
                 lr: float = 5e-4,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_min: float = 0.05,
                 epsilon_decay: float = 0.995,
                 per_capacity: int = 200_000,
                 per_alpha: float = 0.6):
        # ---- networks
        from model import DuelingDQN
        self.policy = DuelingDQN(state_dim, action_dim, hidden).to(self._device())
        self.target = DuelingDQN(state_dim, action_dim, hidden).to(self._device())
        self.target.load_state_dict(self.policy.state_dict())

        # ---- optimiser
        self.optimizer = torch.optim.AdamW(self.policy.parameters(),
                                           lr=lr,
                                           weight_decay=1e-5)

        # ---- RL hyper-params
        self.gamma           = gamma
        self.epsilon         = float(epsilon_start)
        self.epsilon_min     = float(epsilon_min)
        self.epsilon_decay   = float(epsilon_decay)

        # ---- PER
        self.memory = PrioritizedReplay(capacity=per_capacity, alpha=per_alpha)
        self.beta            = 0.4
        self.beta_increment  = 1e-4

        # ---- misc
        self.update_target_every = 20

    # -----------------------------------------------------------------------
    def _device(self):
        """Pick the right device (MPS on Apple-Silicon, fall back to CPU)"""
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # -----------------------------------------------------------------------
    def act(self, state: torch.Tensor, valid_mask: np.ndarray) -> int:
        """ε-greedy that respects the mask."""
        legal = np.where(valid_mask)[0]
        if len(legal) == 0:
            return 0

        # explore
        if np.random.rand() < self.epsilon:
            return int(np.random.choice(legal))

        # exploit - greedy Q-value (mask = -inf)
        q = self.policy(state).detach().cpu().numpy()[0]
        q[valid_mask == 0] = -np.inf
        return int(np.argmax(q))

    # -----------------------------------------------------------------------
    def decay_epsilon(self):
        """Call once per episode (after warm-up)."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = float(self.epsilon)

    # -----------------------------------------------------------------------
    def remember(self, s, a, r, s_next, d):
        self.memory.add(s, a, r, s_next, d)

    # -----------------------------------------------------------------------
    def replay(self, batch_size: int):
        """One gradient step (uses PER + IS-weights). FIXED: Proper MPS autocast."""
        if len(self.memory) < batch_size:
            return

        batch, idx, is_weights = self.memory.sample(batch_size, beta=self.beta)
        
        # ---- unpack to tensors
        device = self._device()
        states = torch.tensor(np.vstack([b[0] for b in batch]),
                             dtype=torch.float32, device=device)
        actions = torch.tensor([b[1] for b in batch],
                              dtype=torch.long, device=device).unsqueeze(1)
        rewards = torch.tensor([b[2] for b in batch],
                              dtype=torch.float32, device=device).unsqueeze(1)
        next_states = torch.tensor(np.vstack([b[3] for b in batch]),
                                   dtype=torch.float32, device=device)
        dones = torch.tensor([b[4] for b in batch],
                            dtype=torch.float32, device=device).unsqueeze(1)

        # ---- double-DQN target
        with torch.no_grad():
            next_q_vals = self.policy(next_states)
            next_actions = next_q_vals.max(1)[1].unsqueeze(1)
            target_q_vals = self.target(next_states)
            next_q_target = target_q_vals.gather(1, next_actions)

            td_target = rewards + self.gamma * next_q_target * (1.0 - dones)

        # ---- current Q
        q_pred = self.policy(states).gather(1, actions)

        # ---- loss (Huber) + IS-weights
        loss = F.smooth_l1_loss(q_pred, td_target, reduction="none")
        loss = (loss * torch.tensor(is_weights, device=device).unsqueeze(1)).mean()

        # ---- back-prop with PROPER autocast for MPS
        self.optimizer.zero_grad()
        
        # FIXED: Use torch.autocast with device_type and dtype for MPS
        # MPS supports float16 and bfloat16 (bfloat16 requires macOS 14+)
        if device.type == 'mps':
            # Use autocast with explicit dtype for MPS
            with torch.autocast(device_type='mps', dtype=torch.float16):
                loss.backward()
        elif device.type == 'cuda':
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                loss.backward()
        else:
            # CPU - no autocast needed
            loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        # ---- update PER priorities
        td_err = (q_pred - td_target).abs().cpu().detach().numpy()
        self.memory.update_priorities(idx, td_err)

        # ---- anneal β
        self.beta = min(1.0, self.beta + self.beta_increment)

    # -----------------------------------------------------------------------
    def update_target(self):
        """Hard copy of policy → target."""
        self.target.load_state_dict(self.policy.state_dict())

    # -----------------------------------------------------------------------
    def save(self, fpath: str):
        torch.save(self.policy.state_dict(), fpath)

    def load(self, fpath: str):
        self.policy.load_state_dict(torch.load(fpath, map_location=self._device()))
        self.target.load_state_dict(self.policy.state_dict())