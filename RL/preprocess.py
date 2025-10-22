#!/usr/bin/env python
"""
preprocess.py
    - reads a raw CSV with OHLC columns (open, high, low, close) and an optional timestamp index.
    - computes EMA_5, Bollinger Band middle (BBM_5_2.0) and historical volatility.
    - standardises the selected columns with sklearn.StandardScaler.
    - writes two npy files: norm.npy (66-dim normalised features) and raw.npy (raw OHLC).

Run:
    $ python preprocess.py selected_features.csv
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging
import sys

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
log.addHandler(handler)

def calc_volatility(close: np.ndarray, win: int = 20) -> np.ndarray:
    """fast numpy version of your pandas log-return volatility."""
    log_ret = np.log(close[1:] / close[:-1])
    vol = np.empty_like(close)
    vol[:] = np.nan
    # rolling std with numpy (vectorised)
    for i in range(win, len(close)):
        vol[i] = np.std(log_ret[i - win:i]) * np.sqrt(252)
    return vol

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # EMA_5
    if "EMA_5" not in df.columns:
        df["EMA_5"] = df["close"].ewm(span=5, adjust=False).mean()
    # Bollinger Bands (middle)
    if "BBM_5_2.0" not in df.columns:
        rolling = df["close"].rolling(window=5)
        df["BBM_5_2.0"] = rolling.mean()
        df["BBU_5_2.0"] = df["BBM_5_2.0"] + 2 * rolling.std()
        df["BBL_5_2.0"] = df["BBM_5_2.0"] - 2 * rolling.std()
    # Volatility
    df["volatility"] = calc_volatility(df["close"].values, win=20)
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="path to the raw CSV (must contain open/high/low/close).")
    args = parser.parse_args()

    # --------------------------------------------------------
    # Load & clean
    # --------------------------------------------------------
    df = pd.read_csv(args.csv_path, index_col="timestamp", parse_dates=True)
    df = df.sort_index()
    # drop dup timestamps
    if df.index.duplicated().any():
        log.info(f"Removing {df.index.duplicated().sum()} duplicate timestamps.")
        df = df[~df.index.duplicated(keep="first")]

    # --------------------------------------------------------
    # Indicators
    # --------------------------------------------------------
    df = add_indicators(df)
    df = df.dropna()  # remove the first few rows where rolling windows are NaN
    log.info(f"After indicator calc: {len(df)} rows.")

    # --------------------------------------------------------
    # Normalisation - we only keep the columns we will actually feed the RL net.
    # --------------------------------------------------------
    feature_cols = [
        "close", "high", "low", "open", "EMA_5", "BBM_5_2.0", "volatility"
    ]  # you can extend this list later.
    scaler = StandardScaler()
    norm_arr = scaler.fit_transform(df[feature_cols].values).astype(np.float32)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    # raw OHLC (4 columns) - keep as float32
    raw_arr = df[["close", "high", "low", "open"]].values.astype(np.float32)

    np.save("norm.npy", norm_arr)
    np.save("raw.npy", raw_arr)
    log.info(f"Saved norm.npy (shape={norm_arr.shape}) and raw.npy (shape={raw_arr.shape})")

if __name__ == "__main__":
    main()