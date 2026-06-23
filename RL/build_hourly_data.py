#!/usr/bin/env python
"""
Build CLEAN HOURLY training data from the raw 1-minute parquet batches.

ROOT-CAUSE FIX: the previous pipeline saved 1-MINUTE candles into files named
`*_1h.npy` and the whole system (env episode_len, oracle horizons, momentum
lookbacks) assumed HOURLY bars. A 200-step episode was therefore 3.3 real hours,
in which a >15% "jackpot" essentially never occurs (~0.19%), so the oracle had
nothing to predict and fees from forced exploration drained ~4%/episode.

This script:
  1. Loads the 4 parquet batches (1-min OHLCV, epoch-second timestamps)
  2. De-dupes / sorts to a clean minute series
  3. Removes catastrophic bad ticks (e.g. 588 -> 1.50 -> 584 spikes)
  4. Resamples to HOURLY OHLCV
  5. Recomputes the 12 technical indicators ON HOURLY bars
  6. Assembles the 16-col feature frame (order matches env expectations)
  7. RobustScaler-normalizes, splits at --train-end, saves norm/raw .npy

Output (into --output-dir, default RL/data/train):
  norm_train_1h.npy  (T, 16)   <- still named _1h; now genuinely hourly
  raw_train_1h.npy   (T, 4)    <- close, high, low, open
  hourly_close_train.npy (T,)  <- convenience for oracle generation

Run:
  python build_hourly_data.py
"""

import argparse
import glob
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

# 16 normalized feature columns, in the exact order env._build_obs expects.
FEATURE_COLS = [
    'close', 'high', 'low', 'open',
    'EMA_5', 'EMA_25', 'EMA_50',
    'SMA_10', 'SMA_20', 'SMA_50',
    'RSI_14', 'volume',
    'MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9',
    'BBB_5_2.0',
]
RAW_COLS = ['close', 'high', 'low', 'open']


def load_minute(parquet_glob):
    files = sorted(glob.glob(parquet_glob))
    if not files:
        raise FileNotFoundError(f"No parquet files match {parquet_glob}")
    parts = []
    for f in files:
        d = pd.read_parquet(f, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        parts.append(d)
        print(f"  loaded {os.path.basename(f)}: {len(d):,} rows")
    df = pd.concat(parts, ignore_index=True)
    df['dt'] = pd.to_datetime(df['timestamp'], unit='s')  # epoch SECONDS
    df = df.sort_values('dt').drop_duplicates('dt').set_index('dt')
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    print(f"  minute series: {len(df):,} rows  {df.index.min()} -> {df.index.max()}")
    return df


def clean_bad_ticks(df, jump=0.5):
    """Null out + interpolate minute rows whose close makes an implausible jump."""
    c = df['close'].values
    logret = np.zeros(len(c))
    logret[1:] = np.diff(np.log(c))
    # A bad tick is a huge single-bar move; mark the spike bar AND the revert bar
    spike = np.abs(logret) > np.log(1 + jump)
    bad_mask = spike.copy()
    bad_mask[:-1] |= spike[1:]  # the bar before a huge revert is also suspect
    n_bad = int(bad_mask.sum())
    if n_bad:
        for col in ['open', 'high', 'low', 'close']:
            s = df[col].copy()
            s[bad_mask] = np.nan
            df[col] = s.interpolate('time').ffill().bfill()
        print(f"  cleaned {n_bad} bad-tick minute rows (>|{jump*100:.0f}%| jumps)")
    return df


def to_hourly(df):
    o = df['open'].resample('1h').first()
    h = df['high'].resample('1h').max()
    l = df['low'].resample('1h').min()
    c = df['close'].resample('1h').last()
    v = df['volume'].resample('1h').sum()
    hourly = pd.DataFrame({'open': o, 'high': h, 'low': l, 'close': c, 'volume': v}).dropna()
    # second pass: clip residual >30% hourly close spikes
    cc = hourly['close'].values
    lr = np.zeros(len(cc)); lr[1:] = np.diff(np.log(cc))
    sp = np.abs(lr) > np.log(1.30)
    if sp.any():
        s = hourly['close'].copy(); s[sp] = np.nan
        hourly['close'] = s.interpolate('time').ffill().bfill()
        print(f"  clipped {int(sp.sum())} residual >30% hourly close spikes")
    print(f"  hourly series: {len(hourly):,} bars")
    return hourly


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rma(s, n):
    # Wilder's smoothing (used by pandas_ta RSI)
    return s.ewm(alpha=1.0 / n, adjust=False).mean()


def rsi(close, n=14):
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    rs = rma(up, n) / rma(down, n).replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def add_indicators(hourly):
    c = hourly['close']
    hourly['EMA_5'] = ema(c, 5)
    hourly['EMA_25'] = ema(c, 25)
    hourly['EMA_50'] = ema(c, 50)
    hourly['SMA_10'] = c.rolling(10).mean()
    hourly['SMA_20'] = c.rolling(20).mean()
    hourly['SMA_50'] = c.rolling(50).mean()
    hourly['RSI_14'] = rsi(c, 14)
    macd = ema(c, 12) - ema(c, 26)
    macds = ema(macd, 9)
    hourly['MACD_12_26_9'] = macd
    hourly['MACDs_12_26_9'] = macds
    hourly['MACDh_12_26_9'] = macd - macds
    mid = c.rolling(5).mean()
    sd = c.rolling(5).std(ddof=0)
    upper = mid + 2.0 * sd
    lower = mid - 2.0 * sd
    hourly['BBB_5_2.0'] = ((upper - lower) / mid * 100).replace([np.inf, -np.inf], np.nan)
    return hourly


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parquet-glob', default='../data/*.parquet')
    ap.add_argument('--train-end', default='2019-12-31')
    ap.add_argument('--output-dir', default='data/train')
    args = ap.parse_args()

    print("=" * 80)
    print("BUILD CLEAN HOURLY DATA")
    print("=" * 80)
    print("1/5 loading minute parquet...")
    dfm = load_minute(args.parquet_glob)

    print("2/5 cleaning bad ticks...")
    dfm = clean_bad_ticks(dfm)

    print("3/5 resampling to hourly...")
    hourly = to_hourly(dfm)

    print("4/5 recomputing indicators on hourly bars...")
    hourly = add_indicators(hourly)
    hourly = hourly.bfill().ffill()

    print("5/5 split + normalize + save...")
    train = hourly[hourly.index <= args.train_end]
    print(f"  train hourly bars: {len(train):,}  {train.index.min()} -> {train.index.max()}")

    feat = train[FEATURE_COLS].copy()
    assert feat.shape[1] == 16, feat.shape
    scaler = RobustScaler()
    norm = scaler.fit_transform(feat).astype(np.float32)
    raw = train[RAW_COLS].values.astype(np.float32)
    close = train['close'].values.astype(np.float32)

    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, 'norm_train_1h.npy'), norm)
    np.save(os.path.join(args.output_dir, 'raw_train_1h.npy'), raw)
    np.save(os.path.join(args.output_dir, 'hourly_close_train.npy'), close)

    print(f"\n✅ saved norm {norm.shape}, raw {raw.shape} to {args.output_dir}/")
    # sanity
    r = np.diff(close) / close[:-1]
    print(f"   hourly 1-step return std={r.std():.4f}  max|r|={np.abs(r).max():.4f}")
    for H in (24, 100, 200):
        fwd = (close[H:] - close[:-H]) / close[:-H]
        print(f"   fwd-{H}h  frac ret>+15% = {np.mean(fwd > 0.15):.4f}")


if __name__ == '__main__':
    main()
