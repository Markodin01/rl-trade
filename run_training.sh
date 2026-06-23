#!/bin/bash
# Oracle RL training pipeline (hourly data + directional oracle PF).
# Run from the rl-trade root directory.

set -e

echo "=========================================="
echo "ORACLE RL TRAINING PIPELINE"
echo "=========================================="
echo ""

if [ ! -f "RL/train_oracle.py" ]; then
    echo "❌ ERROR: Must run from rl-trade root directory"
    exit 1
fi

cd RL

# Step 1: Build clean HOURLY data + directional oracle (if missing)
if [ ! -f "data/train/norm_train_1h.npy" ] || [ ! -f "data/train/oracle_pf_train_1h.npy" ]; then
    echo "📊 Step 1: Building hourly data + oracle PF..."
    python build_hourly_data.py        # minute parquet -> clean hourly norm/raw
    python build_oracle_hourly.py      # 8 signed directional oracle features
    echo "✅ Data ready"
    echo ""
else
    echo "✅ Data already built (data/train/*.npy present)"
    echo ""
fi

# Step 2: Train. Add --discrete-sizing to let the agent choose bet size.
EPISODES="${1:-400}"
echo "🚀 Step 2: Training oracle agent for ${EPISODES} episodes (MPS, ~6s/ep)..."
echo "   reward=tail, gamma=0.997, conviction sizing on by default"
echo ""
python train_oracle.py --episodes "${EPISODES}"

echo ""
echo "=========================================="
echo "TRAINING COMPLETE  —  see RL/training_logs/"
echo "  python analyze_episodes.py training_logs/run_LATEST --top 10"
echo "=========================================="
