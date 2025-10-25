#!/usr/bin/env python
"""
Diagnostic script to verify data and environment before training.
Run this first to catch issues early.

Usage:
    python diagnostic.py
"""

import numpy as np
import os
import sys

def check_data_files():
    """Check if data files exist and are valid."""
    print("="*80)
    print("1. CHECKING DATA FILES")
    print("="*80)
    
    files_to_check = [
        ("norm_train_1h.npy", "Normalized training data"),
        ("raw_train_1h.npy", "Raw training data"),
        ("norm_test_1h.npy", "Normalized test data (optional)"),
        ("raw_test_1h.npy", "Raw test data (optional)"),
    ]
    
    found_files = {}
    for filename, description in files_to_check:
        paths = [filename, f"data/processed/train/{filename}"]
        found = False
        for path in paths:
            if os.path.exists(path):
                print(f"✅ Found {description}: {path}")
                found_files[filename] = path
                found = True
                break
        
        if not found:
            if "test" in filename:
                print(f"⚠️  Missing {description} (optional)")
            else:
                print(f"❌ Missing {description}")
                return None
    
    return found_files

def validate_data_shapes(found_files):
    """Validate data shapes and content."""
    print("\n" + "="*80)
    print("2. VALIDATING DATA SHAPES")
    print("="*80)
    
    try:
        norm_train = np.load(found_files["norm_train_1h.npy"])
        raw_train = np.load(found_files["raw_train_1h.npy"])
        
        print(f"Normalized training shape: {norm_train.shape}")
        print(f"Raw training shape: {raw_train.shape}")
        
        # Check dimensions
        if norm_train.shape[1] != 16:
            print(f"❌ ERROR: Expected 16 features, got {norm_train.shape[1]}")
            return False
        
        if raw_train.shape[1] != 4:
            print(f"❌ ERROR: Expected 4 OHLC columns, got {raw_train.shape[1]}")
            return False
        
        if norm_train.shape[0] != raw_train.shape[0]:
            print(f"❌ ERROR: Length mismatch: norm={norm_train.shape[0]}, raw={raw_train.shape[0]}")
            return False
        
        print(f"✅ Shapes are correct")
        
        # Check for NaN values
        nan_norm = np.isnan(norm_train).sum()
        nan_raw = np.isnan(raw_train).sum()
        
        if nan_norm > 0:
            print(f"⚠️  WARNING: {nan_norm} NaN values in normalized data")
        else:
            print(f"✅ No NaN values in normalized data")
        
        if nan_raw > 0:
            print(f"⚠️  WARNING: {nan_raw} NaN values in raw data")
        else:
            print(f"✅ No NaN values in raw data")
        
        # Check for inf values
        inf_norm = np.isinf(norm_train).sum()
        inf_raw = np.isinf(raw_train).sum()
        
        if inf_norm > 0:
            print(f"⚠️  WARNING: {inf_norm} Inf values in normalized data")
        
        if inf_raw > 0:
            print(f"⚠️  WARNING: {inf_raw} Inf values in raw data")
        
        # Check value ranges
        print(f"\nNormalized data ranges:")
        print(f"  Min: {norm_train.min():.2f}")
        print(f"  Max: {norm_train.max():.2f}")
        print(f"  Mean: {norm_train.mean():.2f}")
        print(f"  Std: {norm_train.std():.2f}")
        
        print(f"\nRaw OHLC ranges:")
        print(f"  Open - Min: {raw_train[:, 3].min():.2f}, Max: {raw_train[:, 3].max():.2f}")
        print(f"  High - Min: {raw_train[:, 1].min():.2f}, Max: {raw_train[:, 1].max():.2f}")
        print(f"  Low - Min: {raw_train[:, 2].min():.2f}, Max: {raw_train[:, 2].max():.2f}")
        print(f"  Close - Min: {raw_train[:, 0].min():.2f}, Max: {raw_train[:, 0].max():.2f}")
        
        # Check for negative prices (should never happen)
        if (raw_train < 0).any():
            print(f"❌ ERROR: Negative prices found in raw data!")
            return False
        
        # Check for zero prices
        if (raw_train == 0).any():
            print(f"⚠️  WARNING: Zero prices found in raw data")
        
        print(f"✅ Data validation passed")
        return True
        
    except Exception as e:
        print(f"❌ ERROR loading data: {e}")
        return False

def test_environment():
    """Test environment creation and basic functionality."""
    print("\n" + "="*80)
    print("3. TESTING ENVIRONMENT")
    print("="*80)
    
    try:
        # Import after sys.path adjustment if needed
        from env import CryptoTradingEnvLongShort
        
        print("✅ Successfully imported environment")
        
        # Load data
        try:
            norm = np.load("norm_train_1h.npy")
            raw = np.load("raw_train_1h.npy")
        except:
            norm = np.load("data/processed/train/norm_train_1h.npy")
            raw = np.load("data/processed/train/raw_train_1h.npy")
        
        # Create environment
        env = CryptoTradingEnvLongShort(
            norm, raw,
            init_balance=10_000,
            fee_pct=0.001,
            episode_len=500,
            random_start=False,  # Deterministic for testing
            lookback=10,
            drawdown_limit=0.30
        )
        
        print("✅ Environment created successfully")
        print(f"   Observation space: {env.observation_space.shape}")
        print(f"   Action space: {env.action_space.n}")
        
        # Test reset
        obs = env.reset()
        print(f"✅ Reset successful, observation shape: {obs.shape}")
        
        if obs.shape[0] != 70:
            print(f"❌ ERROR: Expected 70-dim observation, got {obs.shape[0]}")
            return False
        
        # Test a few steps
        print("\nTesting episode steps:")
        for i in range(5):
            mask = env._valid_mask()
            valid_actions = np.where(mask == 1)[0]
            action = np.random.choice(valid_actions)
            
            obs, reward, done, info = env.step(action)
            
            print(f"  Step {i+1}: action={action}, reward={reward:.2f}, "
                  f"portfolio=${info['portfolio_value']:,.2f}, done={done}")
            
            if done:
                print(f"    Episode terminated early")
                break
        
        print("✅ Environment test passed")
        return True
        
    except ImportError as e:
        print(f"❌ ERROR: Could not import environment: {e}")
        print("   Make sure env_fixed.py is in the current directory")
        return False
    except Exception as e:
        print(f"❌ ERROR during environment test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_agent():
    """Test agent creation."""
    print("\n" + "="*80)
    print("4. TESTING AGENT")
    print("="*80)
    
    try:
        import torch
        from agent import DuelingDQNAgent
        
        print("✅ Successfully imported agent")
        
        # Check device
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"   Device: {device}")
        
        # Create agent
        agent = DuelingDQNAgent(
            state_dim=70,
            action_dim=5,
            hidden=128,
            lr=5e-4,
            gamma=0.99,
            per_capacity=50_000
        )
        
        print("✅ Agent created successfully")
        print(f"   Policy network: {sum(p.numel() for p in agent.policy.parameters())} parameters")
        print(f"   Replay buffer capacity: {agent.memory.capacity}")
        
        # Test forward pass
        dummy_state = torch.randn(1, 70).to(device)
        with torch.no_grad():
            q_values = agent.policy(dummy_state)
        
        print(f"✅ Forward pass successful, Q-values shape: {q_values.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR during agent test: {e}")
        import traceback
        traceback.print_exc()
        return False

def estimate_training_time():
    """Estimate training time."""
    print("\n" + "="*80)
    print("5. TRAINING TIME ESTIMATE")
    print("="*80)
    
    episodes = 1500
    seconds_per_episode = 12  # Conservative estimate after fixes
    
    total_seconds = episodes * seconds_per_episode
    hours = total_seconds / 3600
    
    print(f"For {episodes} episodes:")
    print(f"   @ {seconds_per_episode}s/episode = {hours:.1f} hours")
    print(f"   Quick test (100 episodes): {100 * seconds_per_episode / 60:.1f} minutes")
    
    print(f"\n💡 Recommendation:")
    print(f"   Start with: python train_fixed.py --quick")
    print(f"   Monitor first 10 episodes closely")
    print(f"   If stable, continue with full training")

def main():
    print("\n")
    print("╔" + "═"*78 + "╗")
    print("║" + " "*20 + "RL CRYPTO AGENT DIAGNOSTIC TOOL" + " "*27 + "║")
    print("╚" + "═"*78 + "╝")
    print()
    
    all_passed = True
    
    # 1. Check data files
    found_files = check_data_files()
    if found_files is None:
        print("\n❌ CRITICAL: Required data files not found!")
        print("   Run preprocess.py first to generate training data")
        sys.exit(1)
    
    # 2. Validate data
    if not validate_data_shapes(found_files):
        all_passed = False
        print("\n⚠️  Data validation failed - check your preprocessing")
    
    # 3. Test environment
    if not test_environment():
        all_passed = False
        print("\n⚠️  Environment test failed")
    
    # 4. Test agent
    if not test_agent():
        all_passed = False
        print("\n⚠️  Agent test failed")
    
    # 5. Estimate time
    estimate_training_time()
    
    # Final summary
    print("\n" + "="*80)
    print("DIAGNOSTIC SUMMARY")
    print("="*80)
    
    if all_passed:
        print("✅ All checks passed! Ready to train.")
        print("\nNext steps:")
        print("  1. python train_fixed.py --quick")
        print("  2. Monitor training_logs/run_XXX/main.log")
        print("  3. Check wins/losses folders for episode logs")
        print("  4. If stable after 100 episodes, run full training")
    else:
        print("⚠️  Some checks failed. Review the errors above.")
        print("   Fix issues before starting training to avoid wasted time.")
    
    print("="*80)

if __name__ == "__main__":
    main()