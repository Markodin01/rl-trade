#!/usr/bin/env python
"""
Build the DIRECTIONAL oracle particle-filter features on clean HOURLY data.

Why this replaces oracle_pf.py for the oracle experiment:
  The old oracle used will_jackpot = (|forward_return| > 0.15), i.e. it fired
  identically for +20% pumps and -20% crashes. Audit measured that 56% of
  "jackpot" signals were actually DOWN moves, so an agent that trusts the
  oracle and goes long buys into crashes the majority of the time -> 0% win
  rate even WITH perfect information. The fix: make every signal SIGNED so the
  agent can unambiguously choose long vs short.

8 features (all finite, lookahead is intentional -- this is the upper bound):
  [0] jp_signed_24h   = clip(fwd_return_24h  / 0.15, -1, 1)   sign=dir, |.|>=1 => jackpot
  [1] jp_signed_100h  = clip(fwd_return_100h / 0.15, -1, 1)
  [2] jp_signed_200h  = clip(fwd_return_200h / 0.15, -1, 1)
  [3] time_to_tail    = clip(1 - hours_to_next_tail/200, 0, 1)  1=tail imminent
  [4] next_tail_dir   = sign of the next >tail_threshold move in {-1,0,+1}
  [5] in_window       = 1 if within `window` hours BEFORE the next tail (entry zone)
  [6] regime          = 1 if realized vol (20h) above median, else 0
  [7] criticality     = SOC index in [0,1] (vol + forward tail density)

Run:
  python build_oracle_hourly.py                       # uses data/train/hourly_close_train.npy
  python build_oracle_hourly.py --close data/train_hourly/hourly_close_train.npy \
                                --out  data/train_hourly/oracle_pf_train_1h.npy
"""

import argparse
import numpy as np


def forward_return(close, h):
    fwd = np.full(len(close), np.nan, dtype=np.float64)
    fwd[:-h] = (close[h:] - close[:-h]) / close[:-h]
    return fwd


def build_oracle(close, tail_threshold=0.05, jackpot_threshold=0.15,
                 window=24, verbose=True):
    close = np.asarray(close, dtype=np.float64)
    T = len(close)
    feats = np.zeros((T, 8), dtype=np.float32)

    # --- [0..2] SIGNED jackpot strength at 3 horizons -----------------------
    for j, h in enumerate((24, 100, 200)):
        fwd = forward_return(close, h)
        signed = np.clip(fwd / jackpot_threshold, -1.0, 1.0)
        feats[:, j] = np.nan_to_num(signed, nan=0.0)

    # --- tail events (single-bar moves > threshold) -------------------------
    logret = np.zeros(T)
    logret[1:] = np.diff(np.log(close))
    is_tail = np.abs(logret) > tail_threshold
    tail_idx = np.where(is_tail)[0]
    tail_sign = np.sign(logret[tail_idx]) if len(tail_idx) else np.array([])

    # --- [3] time_to_next_tail (vectorized via searchsorted) ----------------
    hours_to_next = np.full(T, 1000.0)
    if len(tail_idx):
        pos = np.arange(T)
        nxt = np.searchsorted(tail_idx, pos, side='right')  # first tail strictly after i
        has = nxt < len(tail_idx)
        hours_to_next[has] = tail_idx[nxt[has]] - pos[has]
    feats[:, 3] = np.clip(1.0 - hours_to_next / 200.0, 0, 1)

    # --- [4] next_tail_direction --------------------------------------------
    next_dir = np.zeros(T, dtype=np.float32)
    if len(tail_idx):
        next_dir[has] = tail_sign[nxt[has]]
    feats[:, 4] = next_dir

    # --- [5] in_window: within `window` hours BEFORE the next tail ----------
    feats[:, 5] = ((hours_to_next > 0) & (hours_to_next <= window)).astype(np.float32)

    # --- [6] regime: realized 20h vol above median --------------------------
    vol = np.full(T, np.nan)
    for i in range(20, T):
        vol[i] = np.std(logret[i - 20 + 1:i + 1])
    vmed = np.nanmedian(vol)
    feats[:, 6] = np.nan_to_num((vol > vmed).astype(np.float32), nan=0.0)

    # --- [7] criticality: vol (normalized) + forward tail density -----------
    vol_norm = (vol - np.nanmin(vol)) / (np.nanmax(vol) - np.nanmin(vol) + 1e-9)
    vol_norm = np.nan_to_num(vol_norm, nan=0.0)
    # forward tail density over next 48h (lookahead is fine for an oracle)
    is_tail_f = is_tail.astype(np.float64)
    csum = np.concatenate([[0.0], np.cumsum(is_tail_f)])
    fwd_density = np.zeros(T)
    for i in range(T):
        end = min(i + 48, T)
        fwd_density[i] = (csum[end] - csum[i]) / 48.0
    criticality = 0.6 * vol_norm + 0.4 * np.clip(fwd_density * 12, 0, 1)
    feats[:, 7] = criticality.astype(np.float32)

    assert np.isfinite(feats).all(), "oracle features contain non-finite values"

    if verbose:
        print(f"Oracle features: {feats.shape}")
        jp100 = np.abs(feats[:, 1]) >= 1.0
        up = (feats[:, 1] >= 1.0).sum()
        dn = (feats[:, 1] <= -1.0).sum()
        print(f"  100h jackpots: {jp100.sum():,}  (UP {up:,} / DOWN {dn:,})  "
              f"-> direction now explicit via sign")
        print(f"  in_cascade-window rate : {feats[:,5].mean()*100:.1f}%")
        print(f"  high-vol regime rate   : {feats[:,6].mean()*100:.1f}%")
        print(f"  mean criticality       : {feats[:,7].mean():.3f}")
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--close', default='data/train/hourly_close_train.npy')
    ap.add_argument('--out', default='data/train/oracle_pf_train_1h.npy')
    ap.add_argument('--tail-threshold', type=float, default=0.05)
    ap.add_argument('--jackpot-threshold', type=float, default=0.15)
    args = ap.parse_args()

    close = np.load(args.close)
    print(f"Loaded close: {close.shape}  range {close.min():.1f}..{close.max():.1f}")
    feats = build_oracle(close, args.tail_threshold, args.jackpot_threshold)
    np.save(args.out, feats)
    print(f"\n✅ saved oracle PF -> {args.out}  {feats.shape}")


if __name__ == '__main__':
    main()
