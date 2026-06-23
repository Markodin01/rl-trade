#!/usr/bin/env python
"""
Tabular Q-learning learnability test for the oracle signal.

Decouples "is the signal learnable?" from "is my DQN working?". State is just
(jp_signed_100h bin, position) -> a tiny table that converges in seconds.
Reward is RAW P&L (the true objective). First confirm a hand-coded binned policy
is profitable (state-sufficiency check), then see if tabular Q-learning discovers it.

Run:  python tabular_oracle.py
"""
import numpy as np
from env import CryptoTradingEnvLongShort

EDGES = np.array([-0.66, -0.33, -0.10, 0.10, 0.33, 0.66])  # -> 7 signal bins
NB = len(EDGES) + 1
POS = {0: 0, 1: 1, -1: 2}
NS, NA, INIT = NB * 3, 5, 10000.0
ANAMES = ['HOLD', 'OPN_L', 'CLS_L', 'OPN_S', 'CLS_S']


def main():
    np.random.seed(0)
    norm = np.load("data/train/norm_train_1h.npy")
    raw = np.load("data/train/raw_train_1h.npy")
    oracle = np.load("data/train/oracle_pf_train_1h.npy")
    # Fixed sizing + no cooldown: isolate the SIGNAL-learnability question.
    env = CryptoTradingEnvLongShort(norm, raw, oracle_pf=oracle, episode_len=200,
                                    reward_mode="tail", fee_pct=0.001, cooldown=0,
                                    conviction_sizing=False)

    def sbin(env): return int(np.digitize(float(env.oracle_pf[env.t, 1]), EDGES))
    def state_id(env): return sbin(env) * 3 + POS[env.position]
    def legal(env): return np.where(env._valid_mask())[0]

    def run(policy_fn, n=500):
        rets, trades = [], []
        for _ in range(n):
            env.reset(); done = False
            while not done:
                a = policy_fn(env)
                if env._valid_mask()[a] == 0: a = 0
                _, _, done, info = env.step(a)
            rets.append(info['portfolio_return']); trades.append(info['total_trades'])
        return np.array(rets), np.mean(trades)

    # (A) hand-coded binned policy: state-sufficiency check
    def binned_policy(env):
        b, p = sbin(env), env.position
        if p == 0:  return 1 if b >= 4 else (3 if b <= 2 else 0)
        if p == 1:  return 2 if b <= 3 else 0
        return 4 if b >= 3 else 0
    r, tr = run(binned_policy)
    print(f"(A) HAND-CODED BINNED policy : mean {r.mean()*100:+.2f}%  win {(r>0).mean()*100:.0f}%  trades/ep {tr:.1f}")

    # (B) tabular Q-learning on RAW P&L reward
    Q = np.zeros((NS, NA)); GAMMA, LR, EPISODES = 0.997, 0.1, 8000
    for epi in range(EPISODES):
        eps = max(0.05, 1.0 - epi / (0.7 * EPISODES))
        env.reset(); done = False; s = state_id(env); port = INIT
        while not done:
            lg = legal(env)
            if np.random.rand() < eps:
                a = int(np.random.choice(lg))
            else:
                qa = Q[s].copy(); m = np.ones(NA, bool); m[lg] = False; qa[m] = -1e18; a = int(np.argmax(qa))
            _, _, done, info = env.step(a)
            new_port = info['portfolio_value']
            r_pnl = (new_port - port) / INIT * 100.0
            port = new_port
            s2 = state_id(env); lg2 = legal(env)
            nxt = Q[s2, lg2].max() if (len(lg2) and not done) else 0.0
            Q[s, a] += LR * (r_pnl + GAMMA * nxt - Q[s, a]); s = s2

    def greedy(env):
        s = state_id(env); lg = legal(env); qa = Q[s].copy(); m = np.ones(NA, bool); m[lg] = False; qa[m] = -1e18
        return int(np.argmax(qa))
    r, tr = run(greedy)
    print(f"(B) TABULAR Q (raw P&L)      : mean {r.mean()*100:+.2f}%  win {(r>0).mean()*100:.0f}%  trades/ep {tr:.1f}  median {np.median(r)*100:+.2f}%")

    binlbl = ['<-.66', '-.66/-.33', '-.33/-.1', '-.1/+.1', '+.1/+.33', '+.33/+.66', '>+.66']
    print("\nLEARNED POLICY (rows=signal bin, cols=position):")
    print(f"{'bin':>10} | {'FLAT':>6} {'LONG':>6} {'SHORT':>6}")
    for b in range(NB):
        row = []
        for p in range(3):
            sid = b * 3 + p; la = [0, 1, 3] if p == 0 else ([0, 2] if p == 1 else [0, 4])
            qa = Q[sid].copy(); m = np.ones(NA, bool); m[la] = False; qa[m] = -1e18
            row.append(f"{ANAMES[int(np.argmax(qa))]:>6}")
        print(f"{binlbl[b]:>10} | {row[0]} {row[1]} {row[2]}")


if __name__ == "__main__":
    main()
