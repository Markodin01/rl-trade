# 🔧 Oracle Experiment — Root Cause & Fixes

Your instinct was right: the oracle experiment failed because of **bugs in the
setup and data, not the strategy**. With clean data and a *directional* oracle,
the strategy is hugely profitable with perfect information, and the RL agent
learns to exploit it.

## TL;DR

| | Old (broken) | New (fixed) |
|---|---|---|
| Data | 1-minute candles mislabeled `_1h` | true hourly bars (66,517) |
| Episode length | 200 *minutes* (3.3h) | 200 *hours* (8.3 days) |
| fwd-200h >15% up-move | 0.19% of windows | **15%** of windows |
| Oracle "jackpot" signal | `abs(ret)>15%` → 56% are crashes | **signed** (long vs short explicit) |
| Scripted oracle policy (real env) | n/a | **+6.6%/ep, 89% win, ~2 trades/ep** |
| RL agent (300 eps) | −4.2% / 0% win / 45 trades | **+3.0% / 64% win / 32 trades, rising** |

## Root causes (all verified against the data)

1. **The training data was 1-minute data mislabeled as hourly.**
   `RL/data/train/*_1h.npy` had 2,964,983 rows — 1-minute BTC (parquet timestamp
   diffs are 60s), not hourly. So `episode_len=200` "hours" was really 200
   minutes, and the oracle's 24/100/200-"hour" horizons were minutes. A >15% move
   over 200 minutes happens ~0.19% of the time → `total_jackpots = 0`. The oracle
   had nothing to predict. The preprocess scripts never resampled; they
   date-filtered minute rows and named the output `_1h`.

2. **The oracle was direction-blind.** `will_jackpot = abs(return) > 15%` fires
   identically for +20% pumps and −20% crashes. Measured: **56% of "jackpot"
   signals were DOWN moves.** An agent that trusts the oracle and goes long buys
   into crashes most of the time → losses *even with perfect info*.

3. **Fee bleed from forced overtrading.** Epsilon decayed *per episode* (×0.995),
   so after 200 episodes it was still 0.38 — 38–100% random actions the whole
   run. Random actions round-trip ~45×/episode; at 0.1%×2 fees that's a
   deterministic ~4% loss. The agent never got to *exploit* the oracle.

4. **DQN correctness bugs** (would cap learning even after 1–3):
   - Double-DQN bootstrapped from **illegal next-state actions** (they return −10).
   - Prioritized replay used a `deque` for data but a ring array for priorities —
     after the buffer filled, priorities attached to the **wrong** transitions.
   - Position sizing read `norm[t,11]` (normalized **volume**, not volatility).
   - `autocast` wrapped `loss.backward()` (a no-op).
   - 6 catastrophic bad ticks in the raw price (e.g. `588 → 1.50 → 584`).

## What changed

New scripts:
- **`RL/build_hourly_data.py`** — minute parquet → clean hourly OHLCV (bad ticks
  removed), recomputes the 12 indicators on hourly bars, RobustScaler, splits at
  2019-12-31 → `data/train/{norm,raw}_train_1h.npy` (now genuinely hourly) +
  `hourly_close_train.npy`.
- **`RL/build_oracle_hourly.py`** — generates the 8 **signed/directional** oracle
  features → `data/train/oracle_pf_train_1h.npy`.

Edits:
- `RL/env.py` — real volatility-based position sizing; reward no longer pays a
  patience bonus for holding losers; terminal force-close no longer double-counts
  returns.
- `RL/agent.py` — Double-DQN masks illegal next-state actions; removed the no-op
  autocast; `remember()` stores the next-state mask.
- `RL/replay.py` — rewritten so data and priorities share one ring index.
- `RL/train_oracle.py` — **per-step** linear epsilon anneal; threads the
  next-state mask through.
- `RL/train.py`, `RL/train_tail.py` — updated to the new `remember()` signature.

Old minute-data files backed up to `RL/data/train_minute_backup/`.

## New oracle features (all signed; lookahead is intentional — this is the upper bound)

```
[0] jp_signed_24h   = clip(fwd_return_24h  / 0.15, -1, 1)   sign = direction
[1] jp_signed_100h  = clip(fwd_return_100h / 0.15, -1, 1)   |.|>=1 => jackpot
[2] jp_signed_200h  = clip(fwd_return_200h / 0.15, -1, 1)
[3] time_to_tail    = 1 when a >5% move is imminent
[4] next_tail_dir   = sign of the next tail move {-1,0,+1}
[5] in_window       = within 24h BEFORE the next tail (entry zone)
[6] regime          = high/low realized-vol regime
[7] criticality     = SOC index (vol + forward tail density)
```

## How to re-run from scratch

```bash
cd RL
python build_hourly_data.py            # -> data/train/{norm,raw,hourly_close}_train_1h.npy
python build_oracle_hourly.py          # -> data/train/oracle_pf_train_1h.npy
python train_oracle.py --episodes 300  # MPS, ~30 min
```

(If torch fails to import with a `libtorch_cpu.dylib` error, repair it:
`pip install --force-reinstall --no-deps torch==2.9.0`.)

## RL training result (300 episodes, MPS, ~30 min)

The agent **learns to exploit the oracle** as epsilon anneals — the exact
opposite of the old run's flat 0% win rate:

| Episode | Win rate | Avg return (last 50) | Trades/ep | Epsilon |
|---------|----------|----------------------|-----------|---------|
| 50  | 16% | −4.36% | 55.9 | 0.77 |
| 100 | 20% | −3.46% | 54.1 | 0.55 |
| 150 | 46% | +1.18% | 48.8 | 0.32 |
| 200 | 48% | +1.12% | 42.1 | 0.10 |
| 250 | 56% | +2.35% | 35.5 | 0.05 |
| 300 | **64%** | **+2.98%** | 31.9 | 0.05 |

- Oracle capture rate: **100%** (every >15% episode was an oracle-predicted move).
- Crosses break-even around episode ~140 (when epsilon drops below ~0.35).
- Still trending up and still over-trading (32 vs the scripted policy's ~2/ep),
  so it has NOT converged — more episodes and/or stronger anti-churn reward
  shaping should push it toward the +6.6% / 89% upper bound.

**Verdict: the oracle experiment now succeeds.** RL *can* exploit tails with
perfect information; the old "FAILURE" was entirely the data/oracle/DQN bugs.

## Run 2 — VC / fat-tail tuning (400 episodes)

After analysis showed run 1 was *scalping* (34 trades/ep, median hold 1–2h, fees
ate half the gross, top-5 trades only 3% of profit), four changes were made to
push it toward fat-tail capture:
1. **γ 0.99 → 0.997** (value far-future tail payoffs).
2. **Reward**: per-step P&L + convex bonus on realized gains + per-open cost.
3. **Trade throttle**: per-open cost + 4-bar re-entry cooldown.
4. **Judge on mean/compounded return**, not win rate.

Result (last 50 eps): **mean +6.88% (median +6.03%), 10.5 trades/ep, oracle
capture 94%** — essentially at the hand-coded ceiling (+6.5%) and near the true
ceiling (+8%). Trade quality flipped from scalping to asymmetric swing-riding:

| | run 1 (scalper) | run 2 (new) |
|---|---|---|
| trades/ep | 34 | 10.4 |
| per-trade win rate | 53% | 69% |
| avg win / avg loss | 1.4× | **2.2×** |
| winner vs loser hold | both 1–2h | winners 12h / losers 9h |

It now lets winners run (held longer AND larger) and cuts losers.

## Tabular Q-learning sanity tool (`tabular_oracle.py`)

A 21-state Q-table (`jp_signed_100h` bin × position), pure numpy, runs in seconds.
- With **raw P&L** reward it learns a sensible profitable policy (+4.18%).
- With the **shaped "tail"** reward it learned *backwards* (−1.38%) — its coarse
  state can't navigate the convex realize-now incentive (the DQN's 78-dim state
  can). Use it as a fast "is the signal learnable / is the reward sane?" check
  before committing to a ~40-min DQN run.

## Conclusion

The oracle upper bound is **established and reachable**: with perfect directional
information, RL captures the moves at ~+6.9%/episode. With a perfect oracle the
win rate stays *high* (you only bet when you know) — the classic low-win-rate /
fat-right-tail shape only appears under uncertainty.

## Bet sizing experiments (runs 3 & 4)

Run 2 used fixed ~50% sizing (agent picks direction/timing only). Two sizing
upgrades were tested:

- **Option 1 — conviction sizing (oracle-handed):** alloc scales 10–90% with how
  strongly the oracle agrees with the side being opened (`conviction_sizing=True`,
  env default).
- **Option 2 — discrete sizing (agent-learned):** `--discrete-sizing` gives a
  7-action space (open small/big per side, 30%/85%) so the agent learns size from
  reward.

| (last 50 eps) | Run 2 fixed | Run 3 conviction | Run 4 discrete |
|---|---|---|---|
| mean | +6.88% | +8.84% | +9.92% (last-100 +11.3%) |
| median | +6.03% | +6.08% | +5.81% |
| Sharpe | 0.32 | **0.42** | 0.31 |
| best / worst ep | +33% / −14% | +59% / −24.7% | +131% / −17% |
| sizing behavior | flat ~50% | 75% strong / 38% weak | ~82% big regardless |
| dir matches oracle | 63% | 75% | 61% |

**Key result:** the agent did NOT learn conviction sizing — it learned "always bet
big" (~82% big regardless of signal strength). That's rational under a *perfect*
oracle: with near-zero uncertainty the optimal policy is be-selective-then-max-bet,
so "always big" yields the highest raw return (+11.3%). True conviction sizing
(Option 1) shows up as better *risk-adjusted* return (Sharpe 0.42 vs 0.31).
Conviction sizing is a response to UNCERTAINTY — it will become essential, and the
agent will learn it, only once a real (uncertain) PF replaces the oracle and/or the
reward is risk-adjusted (Sharpe/drawdown-penalized).

### Next steps
1. Build a *real* (no-lookahead) particle filter to approximate the 8 signed
   oracle features; measure the gap to these upper bounds. This is where true
   fat-tail / low-win-rate / conviction-sizing dynamics appear.
2. For the real-PF stage, switch to a risk-adjusted reward so the agent *learns*
   to size down on uncertain signals (Kelly-style) instead of always max-betting.
