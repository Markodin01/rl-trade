> **⚠️ Historical plan.** See [ORACLE_FIXES.md](ORACLE_FIXES.md) for the current,
> working pipeline and results. Some scripts named below (`oracle_pf.py`,
> `preprocess_with_oracle.py`) were replaced by `RL/build_hourly_data.py` and
> `RL/build_oracle_hourly.py`.

# 🔮 Oracle Particle Filter Training

## The Correct Architecture

```
┌──────────────────────────────────────────────────────────┐
│                   ORACLE EXPERIMENT                       │
│            (Upper Bound / Proof of Concept)              │
└──────────────────────────────────────────────────────────┘

Step 1: Generate Oracle PF (Perfect State Estimation)
        ↓
     [Future Data]  ← Uses lookahead to create perfect features
        ↓
   Oracle PF Features (8-dim):
   - will_jackpot_24h       (Binary: Jackpot in next 24h?)
   - will_jackpot_100h      (Binary: Jackpot in next 100h?)
   - will_jackpot_200h      (Binary: Jackpot in next 200h?)
   - time_to_next_tail      (Continuous: Hours until next tail)
   - in_cascade             (Binary: Currently in volatility cascade?)
   - optimal_entry_score    (0-1: Quality of current entry)
   - regime                 (Binary: High/low volatility)
   - criticality            (0-1: SOC criticality index)
        ↓
Step 2: Train RL Agent with Oracle PF
        ↓
   Observation Space (78-dim):
   - 70 base features (market + position + history)
   - 8 oracle PF features ⭐
        ↓
   RL Agent learns to:
   - Enter when oracle_entry_score > 0.7
   - Hold through cascades (in_cascade = 1)
   - Close before tail events end
        ↓
Step 3: Measure Upper Bound Performance
        ↓
   If SUCCESS (>3% avg return):
      ✅ Strategy is viable
      ✅ Next: Build real PF to approximate oracle

   If FAILURE (<0% avg return):
      ❌ Strategy fundamentally flawed
      ❌ Need different approach
```

---

## 📊 Oracle PF Features Explained

### Generated from Your Data:

| Feature | Type | Description | How It Helps RL |
|---------|------|-------------|-----------------|
| `will_jackpot_24h` | Binary | 1 if next 24h has >15% return | Short-term signal |
| `will_jackpot_100h` | Binary | 1 if next 100h has >15% return | Medium-term signal |
| `will_jackpot_200h` | Binary | 1 if next 200h has >15% return | Long-term signal |
| `time_to_next_tail` | Float [0,1] | Normalized hours to next >5% move | Timing signal |
| `in_cascade` | Binary | 1 if within 24h of tail event | Your 75.4% clustering |
| `optimal_entry_score` | Float [0,1] | Quality of current entry point | Direct guidance |
| `regime` | Binary | High/low volatility regime | Regime detection |
| `criticality` | Float [0,1] | SOC criticality index | Phase transition proximity |

### Your Data Shows:
- **Jackpot rate (200h):** 0.17% of hours (6,178 jackpots in 2.9M hours)
- **Good entry points:** 0.07% of hours (entry_score > 0.7)
- **Oracle capture rate:** 54.5% of jackpots at good entries

---

## 🚀 How to Run

### 1. Quick Test (30 minutes)
```bash
cd RL
python train_oracle.py --episodes 100 --quick
```

**Expected results:**
- Episodes 0-50: Random exploration (~0% returns)
- Episodes 50-100: Should see learning (small positive returns)

**Success criteria:**
- Avg return (last 20): >0%
- Some trades happening (not coasting)

### 2. Full Training (overnight, 6-8 hours)
```bash
cd RL
python train_oracle.py --episodes 1500
```

**Expected timeline:**
- **Eps 0-300:** Exploration (epsilon 1.0 → 0.3)
- **Eps 300-800:** Learning (should see avg return 2-5%)
- **Eps 800+:** Exploitation (should capture jackpots!)

**Success criteria:**
- Avg return (last 100): **>3%**
- Jackpot capture rate: **>50%** of oracle predictions
- Win rate: **>52%**
- Oracle capture rate: **>40%**

### 3. Analyze Results
```bash
cd RL

# Find latest run
ls -lt training_logs/ | head -5

# Analyze top episodes
python analyze_episodes.py training_logs/run_LATEST --top 10 --plot

# Check specific episode behavior
python analyze_episodes.py training_logs/run_LATEST --episode 500
```

---

## 📈 Interpreting Results

### ✅ SUCCESS Scenario
```
Avg return (last 100): +5.2%
Jackpot rate: 18.3% of episodes
Oracle capture rate: 65.2%
Win rate: 56.8%
```

**Interpretation:**
- RL can exploit tails with perfect information
- Strategy is viable
- Next step: Build real PF (no lookahead)
- Gap from real PF to oracle = improvement opportunity

### ⚠️ MARGINAL Scenario
```
Avg return (last 100): +1.5%
Jackpot rate: 8.2% of episodes
Oracle capture rate: 30.1%
Win rate: 51.2%
```

**Interpretation:**
- RL shows some learning but not strong
- May need more training (try 2500 episodes)
- May need reward tuning
- Strategy might work with better hyperparameters

### ❌ FAILURE Scenario
```
Avg return (last 100): -2.1%
Jackpot rate: 2.3% of episodes
Oracle capture rate: 10.5%
Win rate: 45.1%
```

**Interpretation:**
- RL cannot exploit even with perfect information
- Strategy fundamentally flawed OR
- Reward structure not aligned with goal OR
- Need completely different approach

---

## 🎯 What Oracle Training Proves

### If Successful:
1. **RL can learn tail-seeking behavior** (with good features)
2. **The strategy is sound** (theoretical foundation validated)
3. **Real PF just needs to approximate oracle** (engineering problem)
4. **Performance ceiling is known** (oracle = upper bound)

### Next Steps After Success:
1. **Analyze what oracle features agent uses most**
   - Which of the 8 features drive decisions?
   - Can we simplify to fewer features?

2. **Build real PF (no lookahead)**
   - Use only past data
   - Approximate oracle predictions
   - Measure gap from oracle

3. **Compare baseline vs oracle vs real PF**
   ```
   No PF:     X%  (pure market features)
   Oracle PF: Y%  (perfect information)
   Real PF:   Z%  (our implementation)

   Goal: Z > X (real PF beats baseline)
   Gap:  Y - Z (room for PF improvement)
   ```

---

## 📊 Oracle PF Stats (From Your Data)

```
Total candles:          2,964,983 (2012-2019)
Jackpots (>15%):
  24h:                  671     (0.02%)
  100h:                 3,205   (0.11%)
  200h:                 6,178   (0.21%)

Tail events (>5%):      516
Avg time between tails: 5,961 hours (248 days)

Good entry points:      2,531   (0.07% of hours)
Excellent entries:      360     (0.01% of hours)

Jackpot capture rate at good entries: 54.5%
```

**Key Insight:** Only 0.07% of hours are "good entries" but they capture 54.5% of jackpots!

---

## 🔬 What to Monitor During Training

### Every 50 Episodes:
```
Episode 50/1500
  Avg return (last 50):    +0.2%     ← Should increase over time
  Win rate (last 50):      52.1%     ← Should stay >50%
  Jackpot rate (last 50):  4.0%      ← Should increase to 15-20%
  Oracle capture rate:     15.3%     ← KEY METRIC! Should increase
  Avg trades per ep:       5.2       ← Should stay 3-8 (not coasting)
  Epsilon:                 0.523     ← Decreases over time
```

### Key Metrics:
1. **Oracle Capture Rate** - % of jackpots that oracle predicted
   - Early (eps 0-300): Random (~20%)
   - Mid (eps 300-800): Learning (~40%)
   - Late (eps 800+): Should be >50%

2. **Jackpot Rate** - % of episodes with >15% return
   - Target: 15-20% (your data shows 22.9% at 200h)
   - If <5%: Agent not learning
   - If >20%: Agent learned well!

3. **Avg Return (last 100)** - Rolling average
   - Target: >3%
   - If negative after 500 eps: Problem
   - If >5%: Excellent!

---

## 🎓 Files Created

```
RL/
├── oracle_pf.py                  ⭐ Generates oracle PF features
├── preprocess_with_oracle.py     ⭐ Splits oracle PF with data
├── train_oracle.py               ⭐ Training script
├── env.py                        (Updated: +oracle_pf parameter)
│
├── data/train/
│   ├── norm_train_1h.npy         (Market features)
│   ├── raw_train_1h.npy          (OHLC)
│   └── oracle_pf_train_1h.npy    ⭐ Oracle PF features
│
└── oracle_pf_train.npy           (Full oracle PF, before split)
```

---

## 🚀 Ready to Run!

**The correct workflow:**

```bash
# Already done:
# ✅ Generated oracle PF features
# ✅ Preprocessed and split data
# ✅ Integrated with environment

# Now run:
cd RL

# Quick test (30 min)
python train_oracle.py --episodes 100 --quick

# OR full training (overnight)
python train_oracle.py --episodes 1500
```

**This will prove whether RL can exploit tails with perfect information.**

If successful → Build real PF (tomorrow)

If not → Debug reward/architecture before investing in real PF

---

Good luck! 🎯🔮
