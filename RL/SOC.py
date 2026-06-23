#!/usr/bin/env python
"""
analyze_tail_events.py - Statistical analysis of tail events in crypto data

Measures:
1. Frequency of tail events (by magnitude)
2. Power law distribution fitting
3. Temporal clustering
4. Return during tail vs normal periods
5. Predictability of tails

Usage:
    python analyze_tail_events.py norm_train_1h.npy raw_train_1h.npy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
import argparse
from datetime import datetime, timedelta
import json

class TailEventAnalyzer:
    def __init__(self, norm_data: np.ndarray, raw_data: np.ndarray, 
                 start_date: str = "2012-01-01"):
        """
        Args:
            norm_data: (T, 16) normalized features
            raw_data: (T, 4) OHLC prices
            start_date: Start date of data (for timestamp reconstruction)
        """
        self.norm = norm_data
        self.raw = raw_data
        self.prices = raw_data[:, 0]  # Close prices
        
        # Calculate returns (1-hour)
        self.returns = np.diff(np.log(self.prices))
        
        # Create timestamps (assuming hourly data)
        self.timestamps = pd.date_range(
            start=start_date, 
            periods=len(self.prices), 
            freq='h'
        )
        
        print(f"Loaded {len(self.prices):,} hourly candles")
        print(f"Date range: {self.timestamps[0]} to {self.timestamps[-1]}")
        print(f"Duration: {(self.timestamps[-1] - self.timestamps[0]).days} days")
    
    # ================================================================
    # 1. TAIL EVENT IDENTIFICATION
    # ================================================================
    
    def identify_tail_events(self, thresholds=[0.02, 0.03, 0.05, 0.08, 0.10]):
        """
        Identify tail events at different magnitude thresholds
        
        Returns:
            dict: {threshold: list of (timestamp, return, price_change)}
        """
        results = {}
        
        for threshold in thresholds:
            # Find events exceeding threshold (both directions)
            mask = np.abs(self.returns) > threshold
            indices = np.where(mask)[0]
            
            events = []
            for idx in indices:
                events.append({
                    'timestamp': self.timestamps[idx + 1],
                    'return': self.returns[idx],
                    'price_before': self.prices[idx],
                    'price_after': self.prices[idx + 1],
                    'magnitude': abs(self.returns[idx]),
                    'direction': 'UP' if self.returns[idx] > 0 else 'DOWN'
                })
            
            results[threshold] = events
            
            print(f"\n{'='*60}")
            print(f"Threshold: {threshold*100:.0f}% move in 1 hour")
            print(f"{'='*60}")
            print(f"Total events: {len(events)}")
            print(f"Up moves: {sum(1 for e in events if e['direction'] == 'UP')}")
            print(f"Down moves: {sum(1 for e in events if e['direction'] == 'DOWN')}")
            
            if len(events) > 0:
                avg_magnitude = np.mean([e['magnitude'] for e in events])
                max_magnitude = np.max([e['magnitude'] for e in events])
                print(f"Average magnitude: {avg_magnitude*100:.2f}%")
                print(f"Max magnitude: {max_magnitude*100:.2f}%")
                
                # Time between events
                if len(events) > 1:
                    time_diffs = np.diff([e['timestamp'] for e in events])
                    avg_hours = np.mean([td.total_seconds() / 3600 for td in time_diffs])
                    print(f"Avg time between events: {avg_hours:.0f} hours ({avg_hours/24:.1f} days)")
        
        return results
    
    # ================================================================
    # 2. POWER LAW FITTING
    # ================================================================
    
    def fit_power_law(self, min_threshold=0.01):
        """
        Fit power law to return distribution: P(|r| > x) ∝ x^(-α)
        
        Returns:
            dict: {alpha, ks_statistic, p_value, fit_quality}
        """
        # Get absolute returns above threshold
        abs_returns = np.abs(self.returns)
        tail_returns = abs_returns[abs_returns > min_threshold]
        
        if len(tail_returns) < 50:
            print(f"Warning: Only {len(tail_returns)} tail events, increase data or lower threshold")
            return None
        
        # Sort for CDF calculation
        sorted_returns = np.sort(tail_returns)
        
        # Empirical CDF (complementary: P(X > x))
        n = len(sorted_returns)
        empirical_ccdf = 1 - np.arange(1, n + 1) / n
        
        # Fit power law: log(P(X > x)) = -α * log(x) + C
        log_x = np.log(sorted_returns)
        log_ccdf = np.log(empirical_ccdf)
        
        # Linear regression in log-log space
        slope, intercept = np.polyfit(log_x, log_ccdf, 1)
        alpha = -slope
        
        # Calculate R² (goodness of fit)
        predicted = slope * log_x + intercept
        ss_res = np.sum((log_ccdf - predicted) ** 2)
        ss_tot = np.sum((log_ccdf - np.mean(log_ccdf)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Kolmogorov-Smirnov test
        # Compare to theoretical power law CDF
        theoretical_ccdf = (sorted_returns / min_threshold) ** (-alpha)
        ks_stat = np.max(np.abs(empirical_ccdf - theoretical_ccdf))
        
        print(f"\n{'='*60}")
        print(f"POWER LAW ANALYSIS")
        print(f"{'='*60}")
        print(f"Tail events analyzed: {len(tail_returns):,} (>{min_threshold*100:.1f}%)")
        print(f"Power law exponent (α): {alpha:.3f}")
        print(f"R² (goodness of fit): {r_squared:.4f}")
        print(f"KS statistic: {ks_stat:.4f}")
        
        # Interpret alpha
        if alpha < 2:
            print(f"⚠️  α < 2: Infinite variance (extremely fat tails)")
        elif alpha < 3:
            print(f"⚠️  2 < α < 3: Finite variance, infinite 3rd moment")
        else:
            print(f"✅ α > 3: Finite variance and skewness")
        
        # Compare to Gaussian
        print(f"\nComparison to Gaussian:")
        gaussian_prob_5sigma = 2 * (1 - stats.norm.cdf(5))  # Two-tailed
        
        # Power law probability
        if alpha > 1:
            powerlaw_prob_5sigma = 2 * (5 ** (-alpha))
            ratio = powerlaw_prob_5sigma / gaussian_prob_5sigma
            print(f"5-sigma event probability:")
            print(f"  Gaussian: {gaussian_prob_5sigma:.2e}")
            print(f"  Power law: {powerlaw_prob_5sigma:.2e}")
            print(f"  Ratio: {ratio:.1f}x more likely under power law")
        
        return {
            'alpha': alpha,
            'r_squared': r_squared,
            'ks_statistic': ks_stat,
            'n_tail_events': len(tail_returns),
            'min_threshold': min_threshold
        }
    
    def plot_power_law(self, min_threshold=0.01):
        """Plot empirical vs theoretical power law distribution"""
        abs_returns = np.abs(self.returns)
        tail_returns = abs_returns[abs_returns > min_threshold]
        sorted_returns = np.sort(tail_returns)
        
        n = len(sorted_returns)
        empirical_ccdf = 1 - np.arange(1, n + 1) / n
        
        # Fit power law
        log_x = np.log(sorted_returns)
        log_ccdf = np.log(empirical_ccdf)
        slope, intercept = np.polyfit(log_x, log_ccdf, 1)
        alpha = -slope
        
        # Theoretical power law
        theoretical_ccdf = np.exp(slope * log_x + intercept)
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Log-log plot
        ax1.scatter(sorted_returns, empirical_ccdf, alpha=0.5, s=10, label='Empirical')
        ax1.plot(sorted_returns, theoretical_ccdf, 'r-', linewidth=2, 
                label=f'Power Law (α={alpha:.2f})')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlabel('Return Magnitude')
        ax1.set_ylabel('P(X > x)')
        ax1.set_title('Power Law Distribution (Log-Log)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Q-Q plot
        theoretical_quantiles = np.linspace(0.01, 0.99, len(sorted_returns))
        theoretical_values = min_threshold * (theoretical_quantiles ** (-1/alpha))
        
        ax2.scatter(theoretical_values, sorted_returns, alpha=0.5, s=10)
        ax2.plot([sorted_returns.min(), sorted_returns.max()],
                [sorted_returns.min(), sorted_returns.max()],
                'r--', linewidth=2, label='Perfect fit')
        ax2.set_xlabel('Theoretical Quantiles')
        ax2.set_ylabel('Empirical Quantiles')
        ax2.set_title('Q-Q Plot: Power Law Fit')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('power_law_analysis.png', dpi=150)
        print(f"\n✅ Saved power_law_analysis.png")
    
    # ================================================================
    # 3. TEMPORAL CLUSTERING ANALYSIS
    # ================================================================
    
    def analyze_temporal_clustering(self, threshold=0.03, window_hours=24):
        """
        Analyze if tail events cluster in time (volatility cascades)
        """
        # Find tail events
        mask = np.abs(self.returns) > threshold
        indices = np.where(mask)[0]
        
        if len(indices) < 2:
            print("Not enough tail events for clustering analysis")
            return None
        
        # Calculate time between events (in hours)
        time_diffs = np.diff(indices)
        
        # Test for clustering: Are events closer together than random?
        # Random (Poisson): time_diffs ~ Exponential(λ)
        # Clustered: time_diffs has excess of small values
        
        print(f"\n{'='*60}")
        print(f"TEMPORAL CLUSTERING ANALYSIS (>{threshold*100:.0f}% events)")
        print(f"{'='*60}")
        
        # Statistics
        mean_gap = np.mean(time_diffs)
        median_gap = np.median(time_diffs)
        std_gap = np.std(time_diffs)
        
        print(f"Time between tail events:")
        print(f"  Mean: {mean_gap:.1f} hours ({mean_gap/24:.1f} days)")
        print(f"  Median: {median_gap:.1f} hours ({median_gap/24:.1f} days)")
        print(f"  Std: {std_gap:.1f} hours")
        
        # Count events in clusters (within window_hours)
        clustered = time_diffs < window_hours
        cluster_rate = clustered.mean()
        print(f"\nEvents within {window_hours}h of previous: {cluster_rate*100:.1f}%")
        
        # Expected under Poisson (random)
        lambda_rate = 1 / mean_gap
        expected_clustered = 1 - np.exp(-lambda_rate * window_hours)
        print(f"Expected under random (Poisson): {expected_clustered*100:.1f}%")
        print(f"Excess clustering: {(cluster_rate - expected_clustered)*100:.1f} percentage points")
        
        if cluster_rate > expected_clustered * 1.5:
            print("✅ STRONG CLUSTERING: Tail events trigger more tail events (SOC)")
        elif cluster_rate > expected_clustered * 1.2:
            print("⚠️  MODERATE CLUSTERING: Some cascade effects")
        else:
            print("❌ NO CLUSTERING: Events are approximately random")
        
        # Plot distribution of time gaps
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        ax1.hist(time_diffs, bins=50, alpha=0.7, edgecolor='black', density=True)
        
        # Overlay exponential (Poisson/random)
        x = np.linspace(0, time_diffs.max(), 100)
        exponential = lambda_rate * np.exp(-lambda_rate * x)
        ax1.plot(x, exponential, 'r-', linewidth=2, label='Exponential (random)')
        
        ax1.set_xlabel('Hours between tail events')
        ax1.set_ylabel('Density')
        ax1.set_title(f'Distribution of Inter-Event Times (>{threshold*100:.0f}%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Cumulative
        sorted_diffs = np.sort(time_diffs)
        empirical_cdf = np.arange(1, len(sorted_diffs) + 1) / len(sorted_diffs)
        theoretical_cdf = 1 - np.exp(-lambda_rate * sorted_diffs)
        
        ax2.plot(sorted_diffs, empirical_cdf, label='Empirical', linewidth=2)
        ax2.plot(sorted_diffs, theoretical_cdf, 'r--', label='Exponential (random)', linewidth=2)
        ax2.set_xlabel('Hours between tail events')
        ax2.set_ylabel('Cumulative Probability')
        ax2.set_title('CDF: Clustering Test')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('temporal_clustering.png', dpi=150)
        print(f"\n✅ Saved temporal_clustering.png")
        
        return {
            'mean_gap_hours': mean_gap,
            'cluster_rate': cluster_rate,
            'expected_random': expected_clustered,
            'excess_clustering': cluster_rate - expected_clustered
        }
    
    # ================================================================
    # 4. ANNUAL BREAKDOWN
    # ================================================================
    
    def analyze_by_year(self, thresholds=[0.03, 0.05, 0.10]):
        """
        Break down tail events by year
        """
        # Create DataFrame for easier analysis
        df = pd.DataFrame({
            'timestamp': self.timestamps[1:],  # Align with returns
            'return': self.returns,
            'abs_return': np.abs(self.returns),
            'year': self.timestamps[1:].year
        })
        
        print(f"\n{'='*60}")
        print(f"ANNUAL TAIL EVENT BREAKDOWN")
        print(f"{'='*60}")
        
        results = {}
        
        for threshold in thresholds:
            print(f"\n--- Threshold: {threshold*100:.0f}% ---")
            
            yearly = df[df['abs_return'] > threshold].groupby('year').agg({
                'return': ['count', 'mean', 'min', 'max'],
                'abs_return': ['mean', 'max']
            })
            
            yearly.columns = ['count', 'mean_return', 'min_return', 'max_return', 
                            'mean_magnitude', 'max_magnitude']
            
            results[threshold] = yearly
            
            # Print table
            print(f"\n{yearly.to_string()}")
            
            # Calculate total hours per year
            hours_per_year = df.groupby('year').size()
            yearly['frequency_per_1000h'] = (yearly['count'] / hours_per_year * 1000).fillna(0)
            
            print(f"\nFrequency per 1000 hours:")
            print(yearly['frequency_per_1000h'].to_string())
        
        return results
    
    # ================================================================
    # 5. JACKPOT DETECTION (VC-STYLE ANALYSIS)
    # ================================================================
    
    def identify_jackpots(self, lookback_hours=100, min_return=0.10):
        """
        Identify "jackpot" opportunities: periods where cumulative return
        over lookback_hours exceeds min_return
        
        This simulates a trading strategy holding for lookback_hours
        """
        cumulative_returns = []
        jackpot_periods = []
        
        for i in range(len(self.returns) - lookback_hours):
            # Cumulative return over next lookback_hours
            period_returns = self.returns[i:i+lookback_hours]
            cum_return = np.exp(np.sum(period_returns)) - 1
            
            cumulative_returns.append(cum_return)
            
            if abs(cum_return) > min_return:
                jackpot_periods.append({
                    'start': self.timestamps[i],
                    'end': self.timestamps[i + lookback_hours],
                    'return': cum_return,
                    'direction': 'UP' if cum_return > 0 else 'DOWN'
                })
        
        print(f"\n{'='*60}")
        print(f"JACKPOT OPPORTUNITIES (>{min_return*100:.0f}% in {lookback_hours}h)")
        print(f"{'='*60}")
        print(f"Total jackpot periods: {len(jackpot_periods)}")
        print(f"Frequency: 1 per {len(cumulative_returns) / max(len(jackpot_periods), 1):.0f} hours")
        print(f"           ({len(cumulative_returns) / max(len(jackpot_periods), 1) / 24:.1f} days)")
        
        if len(jackpot_periods) > 0:
            up_jackpots = [j for j in jackpot_periods if j['direction'] == 'UP']
            down_jackpots = [j for j in jackpot_periods if j['direction'] == 'DOWN']
            
            print(f"\nUp jackpots: {len(up_jackpots)}")
            if up_jackpots:
                avg_up = np.mean([j['return'] for j in up_jackpots])
                max_up = np.max([j['return'] for j in up_jackpots])
                print(f"  Average: {avg_up*100:.1f}%")
                print(f"  Maximum: {max_up*100:.1f}%")
            
            print(f"\nDown jackpots: {len(down_jackpots)}")
            if down_jackpots:
                avg_down = np.mean([abs(j['return']) for j in down_jackpots])
                max_down = np.min([j['return'] for j in down_jackpots])  # Most negative
                print(f"  Average: {avg_down*100:.1f}%")
                print(f"  Maximum: {abs(max_down)*100:.1f}%")
            
            # Top 10 jackpots
            sorted_jackpots = sorted(jackpot_periods, 
                                   key=lambda x: abs(x['return']), 
                                   reverse=True)[:10]
            
            print(f"\nTop 10 Jackpot Periods:")
            for i, jp in enumerate(sorted_jackpots, 1):
                print(f"{i:2d}. {jp['start'].strftime('%Y-%m-%d')} → "
                      f"{jp['end'].strftime('%Y-%m-%d')}: "
                      f"{jp['return']*100:+.1f}% ({jp['direction']})")
        
        return jackpot_periods, cumulative_returns
    
    # ================================================================
    # 6. VC-STYLE RETURN SIMULATION
    # ================================================================
    
    def simulate_vc_strategy(self, 
                            entry_threshold=0.70,  # Confidence threshold
                            hold_periods=[50, 100, 200],  # Different holding periods
                            n_simulations=1000):
        """
        Simulate VC-style strategy:
        - Only enter during high-confidence periods (>70%)
        - Hold for fixed period
        - Measure distribution of returns
        """
        print(f"\n{'='*60}")
        print(f"VC-STYLE RETURN SIMULATION")
        print(f"{'='*60}")
        
        # For this, we need confidence scores
        # Simple proxy: high volatility + high volume = high confidence
        vol = pd.Series(self.returns).rolling(24).std()
        vol_zscore = (vol - vol.rolling(100).mean()) / vol.rolling(100).std()
        
        # Crude confidence: normalize vol_zscore to [0, 1]
        confidence_proxy = (vol_zscore - vol_zscore.min()) / (vol_zscore.max() - vol_zscore.min())
        confidence_proxy = confidence_proxy.fillna(0).values
        
        results = {}
        
        for hold_period in hold_periods:
            print(f"\n--- Hold period: {hold_period} hours ({hold_period/24:.1f} days) ---")
            
            # Find entry points (high confidence)
            entry_indices = np.where(confidence_proxy[:-hold_period] > entry_threshold)[0]
            
            if len(entry_indices) == 0:
                print(f"No entry points found with confidence > {entry_threshold}")
                continue
            
            # Sample random entries
            sampled_entries = np.random.choice(entry_indices, 
                                             size=min(n_simulations, len(entry_indices)),
                                             replace=False)
            
            # Calculate returns for each entry
            trade_returns = []
            for idx in sampled_entries:
                period_returns = self.returns[idx:idx+hold_period]
                cum_return = np.exp(np.sum(period_returns)) - 1
                trade_returns.append(cum_return)
            
            trade_returns = np.array(trade_returns)
            
            # Statistics
            win_rate = (trade_returns > 0).mean()
            mean_return = trade_returns.mean()
            median_return = np.median(trade_returns)
            
            # Categorize returns (VC-style)
            small_loss = trade_returns[(trade_returns < 0) & (trade_returns > -0.05)]
            small_win = trade_returns[(trade_returns > 0) & (trade_returns < 0.05)]
            medium_win = trade_returns[(trade_returns >= 0.05) & (trade_returns < 0.15)]
            jackpot = trade_returns[trade_returns >= 0.15]
            large_loss = trade_returns[trade_returns <= -0.05]
            
            print(f"\nTrades simulated: {len(trade_returns)}")
            print(f"Win rate: {win_rate*100:.1f}%")
            print(f"Mean return: {mean_return*100:.2f}%")
            print(f"Median return: {median_return*100:.2f}%")
            
            print(f"\nReturn distribution:")
            print(f"  Large losses (<-5%):     {len(large_loss):4d} ({len(large_loss)/len(trade_returns)*100:5.1f}%)")
            print(f"  Small losses (-5% to 0): {len(small_loss):4d} ({len(small_loss)/len(trade_returns)*100:5.1f}%)")
            print(f"  Small wins (0 to 5%):    {len(small_win):4d} ({len(small_win)/len(trade_returns)*100:5.1f}%)")
            print(f"  Medium wins (5-15%):     {len(medium_win):4d} ({len(medium_win)/len(trade_returns)*100:5.1f}%)")
            print(f"  Jackpots (>15%):         {len(jackpot):4d} ({len(jackpot)/len(trade_returns)*100:5.1f}%) ⭐")
            
            if len(jackpot) > 0:
                print(f"\nJackpot statistics:")
                print(f"  Average jackpot: {jackpot.mean()*100:.1f}%")
                print(f"  Max jackpot: {jackpot.max()*100:.1f}%")
                print(f"  Total from jackpots: {jackpot.sum()*100:.1f}%")
                print(f"  Total from all trades: {trade_returns.sum()*100:.1f}%")
                print(f"  Jackpot contribution: {jackpot.sum()/trade_returns.sum()*100:.1f}%")
            
            # Gain-to-pain ratio
            gains = trade_returns[trade_returns > 0].sum()
            pains = abs(trade_returns[trade_returns < 0].sum())
            gtp = gains / pains if pains > 0 else 0
            print(f"\nGain-to-pain ratio: {gtp:.2f}")
            
            results[hold_period] = {
                'win_rate': win_rate,
                'mean_return': mean_return,
                'median_return': median_return,
                'gtp_ratio': gtp,
                'jackpot_count': len(jackpot),
                'jackpot_pct': len(jackpot) / len(trade_returns)
            }
        
        return results
    
    # ================================================================
    # 7. GENERATE REPORT
    # ================================================================
    
    def generate_full_report(self, output_file='tail_event_report.json'):
        """Run all analyses and save report"""
        
        print("\n" + "="*80)
        print("COMPREHENSIVE TAIL EVENT ANALYSIS")
        print("="*80)
        
        report = {
            'metadata': {
                'total_hours': len(self.prices),
                'total_days': len(self.prices) / 24,
                'date_range': f"{self.timestamps[0]} to {self.timestamps[-1]}"
            }
        }
        
        # 1. Tail events
        report['tail_events'] = self.identify_tail_events()
        
        # 2. Power law
        report['power_law'] = self.fit_power_law(min_threshold=0.01)
        self.plot_power_law(min_threshold=0.01)
        
        # 3. Temporal clustering
        report['clustering'] = self.analyze_temporal_clustering(threshold=0.03)
        
        # 4. Annual breakdown
        report['annual'] = self.analyze_by_year()
        
        # 5. Jackpots
        jackpots, cum_returns = self.identify_jackpots(lookback_hours=100, min_return=0.10)
        report['jackpots'] = {
            'count': len(jackpots),
            'frequency_days': len(cum_returns) / max(len(jackpots), 1) / 24
        }
        
        # 6. VC simulation
        report['vc_simulation'] = self.simulate_vc_strategy()
        
        # Save report
        # Convert numpy types to native Python for JSON serialization
        def convert(o):
            if isinstance(o, np.integer):
                return int(o)
            if isinstance(o, np.floating):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, pd.Timestamp):
                return str(o)
            if isinstance(o, pd.DataFrame):
                return o.to_dict()
            return o
        
        import json
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=convert)
        
        print(f"\n✅ Full report saved to {output_file}")
        
        return report


# ================================================================
# MAIN
# ================================================================

def main():
    parser = argparse.ArgumentParser(description='Analyze tail events in crypto data')
    parser.add_argument('norm_file', help='Path to normalized features (.npy)')
    parser.add_argument('raw_file', help='Path to raw OHLC data (.npy)')
    parser.add_argument('--start-date', default='2012-01-01', 
                       help='Start date of data (YYYY-MM-DD)')
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    norm = np.load(args.norm_file)
    raw = np.load(args.raw_file)
    
    # Run analysis
    analyzer = TailEventAnalyzer(norm, raw, start_date=args.start_date)
    report = analyzer.generate_full_report()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  - tail_event_report.json")
    print("  - power_law_analysis.png")
    print("  - temporal_clustering.png")


if __name__ == "__main__":
    main()