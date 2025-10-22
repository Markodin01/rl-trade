#!/usr/bin/env python
"""
preprocess_aggregated.py - IMPROVED VERSION

Handles irregular-interval data by:
1. Aggregating to consistent 1-hour OHLCV candles
2. Recalculating technical indicators on aggregated data
3. Proper handling of volume-weighted features
4. Creates clean train/test splits by time period

Usage:
    python preprocess_aggregated.py your_data.csv --timeframe 1h
    python preprocess_aggregated.py your_data.csv --timeframe 5min --train-end 2020-12-31
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging
import sys
from datetime import datetime

# Setup logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
log.addHandler(handler)

def aggregate_to_timeframe(df: pd.DataFrame, timeframe: str = '1h') -> pd.DataFrame:
    """
    Aggregate irregular data to consistent timeframe.
    
    Args:
        df: DataFrame with timestamp index and OHLCV columns
        timeframe: '1h', '5min', '15min', '1d', etc.
    
    Returns:
        Aggregated DataFrame with proper OHLCV
    """
    log.info(f"Aggregating {len(df)} records to {timeframe} candles...")
    
    # Aggregation rules for OHLCV
    agg_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }
    
    # Resample to timeframe
    df_agg = df.resample(timeframe).agg(agg_dict)
    
    # Remove candles with no data
    df_agg = df_agg.dropna(subset=['close'])
    
    log.info(f"After aggregation: {len(df_agg)} candles")
    log.info(f"Date range: {df_agg.index[0]} to {df_agg.index[-1]}")
    
    return df_agg

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators on aggregated data.
    Uses only the essential indicators for RL.
    """
    log.info("Calculating technical indicators...")
    
    # EMAs
    df['EMA_5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['EMA_25'] = df['close'].ewm(span=25, adjust=False).mean()
    
    # Bollinger Bands (using 20-period on hourly data)
    rolling = df['close'].rolling(window=20)
    df['BBM_20'] = rolling.mean()
    bb_std = rolling.std()
    df['BBU_20'] = df['BBM_20'] + 2 * bb_std
    df['BBL_20'] = df['BBM_20'] - 2 * bb_std
    df['BBB_20'] = (df['BBU_20'] - df['BBL_20']) / df['BBM_20']  # Bandwidth
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # Volatility (log returns)
    log_returns = np.log(df['close'] / df['close'].shift(1))
    df['volatility'] = log_returns.rolling(window=20).std() * np.sqrt(252)
    
    # Price momentum (different lookback periods)
    df['mom_1h'] = df['close'].pct_change(1)
    df['mom_6h'] = df['close'].pct_change(6)
    df['mom_24h'] = df['close'].pct_change(24)
    
    # Volume indicators
    df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma_20']
    
    log.info(f"Indicators calculated. NaN rows: {df.isnull().any(axis=1).sum()}")
    
    return df

def select_features(df: pd.DataFrame) -> tuple:
    """
    Select features for RL training.
    
    Returns:
        market_features: List of feature column names
        raw_ohlc: List of OHLC column names
    """
    # Market features for observation (normalized)
    market_features = [
        'close', 'high', 'low', 'open',           # OHLC (4)
        'EMA_5', 'EMA_25',                        # Trend (2)
        'BBM_20', 'BBU_20', 'BBL_20', 'BBB_20',  # Bollinger (4)
        'RSI_14',                                 # Momentum (1)
        'volatility',                             # Risk (1)
        'volume_ratio',                           # Volume (1)
        'mom_1h', 'mom_6h', 'mom_24h',           # Momentum (3)
    ]
    # Total: 16 features
    
    # Raw OHLC for price calculations (not normalized)
    raw_ohlc = ['close', 'high', 'low', 'open']
    
    return market_features, raw_ohlc

def create_train_test_split(df: pd.DataFrame, train_end: str = None, test_start: str = None):
    """
    Create time-based train/test split.
    
    Args:
        df: Full dataset
        train_end: End date for training (e.g., '2020-12-31')
        test_start: Start date for testing (e.g., '2021-01-01')
    """
    if train_end:
        train_df = df[:train_end]
        log.info(f"Train set: {len(train_df)} candles ({train_df.index[0]} to {train_df.index[-1]})")
    else:
        # Default: 80% train
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx]
        log.info(f"Train set: {len(train_df)} candles (80% of data)")
    
    if test_start:
        test_df = df[test_start:]
        log.info(f"Test set: {len(test_df)} candles ({test_df.index[0]} to {test_df.index[-1]})")
    else:
        # Default: 20% test
        split_idx = int(len(df) * 0.8)
        test_df = df.iloc[split_idx:]
        log.info(f"Test set: {len(test_df)} candles (20% of data)")
    
    return train_df, test_df

def main():
    parser = argparse.ArgumentParser(description='Preprocess crypto data for RL training')
    parser.add_argument('csv_path', help='Path to raw CSV file')
    parser.add_argument('--timeframe', default='1h', 
                       help='Aggregation timeframe: 1h, 5min, 15min, 1d (default: 1h)')
    parser.add_argument('--train-end', default='2020-12-31',
                       help='End date for training data (default: 2020-12-31)')
    parser.add_argument('--test-start', default='2021-01-01',
                       help='Start date for test data (default: 2021-01-01)')
    parser.add_argument('--output-prefix', default='',
                       help='Prefix for output files (default: timeframe)')
    
    args = parser.parse_args()
    
    log.info("="*80)
    log.info("CRYPTO DATA PREPROCESSING - IMPROVED VERSION")
    log.info("="*80)
    
    # --------------------------------------------------------
    # 1. Load raw data
    # --------------------------------------------------------
    log.info(f"\n1. Loading data from {args.csv_path}")
    df = pd.read_csv(args.csv_path, parse_dates=['timestamp'])
    df = df.set_index('timestamp')
    df = df.sort_index()
    
    # Remove duplicates
    if df.index.duplicated().any():
        log.info(f"   Removing {df.index.duplicated().sum()} duplicate timestamps")
        df = df[~df.index.duplicated(keep='first')]
    
    log.info(f"   Loaded {len(df)} records")
    log.info(f"   Date range: {df.index[0]} to {df.index[-1]}")
    log.info(f"   Duration: {(df.index[-1] - df.index[0]).days} days")
    
    # --------------------------------------------------------
    # 2. Aggregate to consistent timeframe
    # --------------------------------------------------------
    log.info(f"\n2. Aggregating to {args.timeframe} candles")
    df_agg = aggregate_to_timeframe(df, args.timeframe)
    
    # Check for gaps
    expected_freq = pd.Timedelta(args.timeframe)
    time_diffs = df_agg.index.to_series().diff()
    large_gaps = time_diffs[time_diffs > expected_freq * 2]
    if len(large_gaps) > 0:
        log.warning(f"   Found {len(large_gaps)} gaps >2x expected interval")
        log.warning(f"   Largest gap: {large_gaps.max()}")
    
    # --------------------------------------------------------
    # 3. Calculate indicators
    # --------------------------------------------------------
    log.info(f"\n3. Calculating technical indicators")
    df_agg = calculate_indicators(df_agg)
    
    # Drop NaN rows from indicator calculations
    df_agg = df_agg.dropna()
    log.info(f"   After removing NaN: {len(df_agg)} candles")
    
    # --------------------------------------------------------
    # 4. Select features
    # --------------------------------------------------------
    log.info(f"\n4. Selecting features for RL")
    market_features, raw_ohlc = select_features(df_agg)
    
    log.info(f"   Market features ({len(market_features)}): {', '.join(market_features)}")
    log.info(f"   Raw OHLC ({len(raw_ohlc)}): {', '.join(raw_ohlc)}")
    
    # Check all features exist
    missing_features = [f for f in market_features if f not in df_agg.columns]
    if missing_features:
        log.error(f"   Missing features: {missing_features}")
        sys.exit(1)
    
    # --------------------------------------------------------
    # 5. Create train/test split
    # --------------------------------------------------------
    log.info(f"\n5. Creating train/test split")
    train_df, test_df = create_train_test_split(
        df_agg, 
        train_end=args.train_end,
        test_start=args.test_start
    )
    
    # --------------------------------------------------------
    # 6. Normalize features (fit on train only!)
    # --------------------------------------------------------
    log.info(f"\n6. Normalizing features (StandardScaler)")
    
    scaler = StandardScaler()
    train_norm = scaler.fit_transform(train_df[market_features].values)
    test_norm = scaler.transform(test_df[market_features].values)
    
    log.info(f"   Train normalized shape: {train_norm.shape}")
    log.info(f"   Test normalized shape: {test_norm.shape}")
    
    # --------------------------------------------------------
    # 7. Save arrays
    # --------------------------------------------------------
    log.info(f"\n7. Saving arrays")
    
    # Output prefix
    prefix = args.output_prefix if args.output_prefix else args.timeframe
    
    # Training data
    np.save(f'norm_train_{prefix}.npy', train_norm.astype(np.float32))
    np.save(f'raw_train_{prefix}.npy', train_df[raw_ohlc].values.astype(np.float32))
    
    log.info(f"   ✅ Saved norm_train_{prefix}.npy (shape={train_norm.shape})")
    log.info(f"   ✅ Saved raw_train_{prefix}.npy (shape={train_df[raw_ohlc].values.shape})")
    
    # Test data
    np.save(f'norm_test_{prefix}.npy', test_norm.astype(np.float32))
    np.save(f'raw_test_{prefix}.npy', test_df[raw_ohlc].values.astype(np.float32))
    
    log.info(f"   ✅ Saved norm_test_{prefix}.npy (shape={test_norm.shape})")
    log.info(f"   ✅ Saved raw_test_{prefix}.npy (shape={test_df[raw_ohlc].values.shape})")
    
    # Save scaler for later use
    import joblib
    joblib.dump(scaler, f'scaler_{prefix}.pkl')
    log.info(f"   ✅ Saved scaler_{prefix}.pkl")
    
    # --------------------------------------------------------
    # 8. Summary
    # --------------------------------------------------------
    log.info(f"\n" + "="*80)
    log.info("PREPROCESSING COMPLETE")
    log.info("="*80)
    log.info(f"Timeframe: {args.timeframe}")
    log.info(f"Features: {len(market_features)}")
    log.info(f"Train candles: {len(train_df)} ({train_df.index[0]} to {train_df.index[-1]})")
    log.info(f"Test candles: {len(test_df)} ({test_df.index[0]} to {test_df.index[-1]})")
    
    # Episode implications
    log.info(f"\nRL Episode Implications ({args.timeframe} candles):")
    for steps in [500, 1000, 2000]:
        if args.timeframe == '1h':
            days = steps / 24
        elif args.timeframe == '5min':
            days = steps / 288
        elif args.timeframe == '15min':
            days = steps / 96
        elif args.timeframe == '1d':
            days = steps
        else:
            days = None
        
        if days:
            log.info(f"  {steps} steps = {days:.1f} days")
        
        episodes_possible = len(train_df) // steps
        log.info(f"  Possible {steps}-step episodes from train data: {episodes_possible}")
    
    log.info("="*80)

if __name__ == "__main__":
    main()