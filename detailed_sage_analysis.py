#!/usr/bin/env python3
"""
Detailed analysis of SAGE model training with loss component breakdown
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# Activate virtual environment
venv_path = os.path.join(os.path.dirname(__file__), '.venv1')
if os.path.exists(venv_path):
    activate_script = os.path.join(venv_path, 'Scripts', 'activate_this.py')
    if os.path.exists(activate_script):
        exec(open(activate_script).read(), {'__file__': activate_script})
    else:
        # Alternative activation method
        site_packages = os.path.join(venv_path, 'Lib', 'site-packages')
        if os.path.exists(site_packages):
            sys.path.insert(0, site_packages)

import tensorflow as tf
import pandas as pd

def extract_all_metrics(log_file_path):
    """Extract all metrics from TensorBoard logs"""
    
    print(f"📊 Analyzing: {log_file_path}")
    
    data = {}
    
    for summary in tf.compat.v1.train.summary_iterator(log_file_path):
        for value in summary.summary.value:
            tag = value.tag
            step = summary.step
            
            if tag not in data:
                data[tag] = []
            
            data[tag].append((step, value.simple_value))
    
    # Convert to DataFrames
    dataframes = {}
    for tag, values in data.items():
        if len(values) > 0:
            df = pd.DataFrame(values, columns=['episode', 'value'])
            dataframes[tag] = df.sort_values('episode')
    
    return dataframes

def analyze_loss_components(data):
    """Analyze individual loss components"""
    print("\n🔍 === LOSS COMPONENT BREAKDOWN ===")
    print("=" * 60)
    
    loss_components = [
        'act_loss_mean', 'crit_loss_mean', 'entropy_loss_mean', 'loss_mean'
    ]
    
    # Find available loss components
    available_losses = []
    for component in loss_components:
        if component in data:
            available_losses.append(component)
    
    if not available_losses:
        print("❌ No detailed loss components found!")
        return
    
    print(f"Found loss components: {available_losses}")
    
    for component in available_losses:
        df = data[component]
        if len(df) > 100:
            # Split into early and late episodes
            mid_point = len(df) // 2
            early_df = df.iloc[:mid_point]
            late_df = df.iloc[mid_point:]
            
            early_mean = early_df['value'].mean()
            late_mean = late_df['value'].mean()
            overall_trend = late_mean - early_mean
            
            print(f"\n📈 {component.upper()}:")
            print(f"   Early episodes: {early_mean:.4f}")
            print(f"   Late episodes: {late_mean:.4f}")
            print(f"   Trend: {'+' if overall_trend > 0 else ''}{overall_trend:.4f}")
            
            # Analyze trend
            if 'act_loss' in component:
                if overall_trend > 0.1:
                    print("   ⚠️  POLICY LOSS INCREASING - Policy changing rapidly")
                elif overall_trend < -0.1:
                    print("   ✅ Policy loss decreasing - Policy stabilizing")
                else:
                    print("   ➡️  Policy loss stable")
                    
            elif 'crit_loss' in component:
                if overall_trend > 0.1:
                    print("   ⚠️  VALUE LOSS INCREASING - Value function struggling")
                elif overall_trend < -0.1:
                    print("   ✅ Value loss decreasing - Value function improving")
                else:
                    print("   ➡️  Value loss stable")
                    
            elif 'entropy_loss' in component:
                if overall_trend > 0.1:
                    print("   ⚠️  ENTROPY LOSS INCREASING - More exploration")
                elif overall_trend < -0.1:
                    print("   ✅ Entropy loss decreasing - Less exploration")
                else:
                    print("   ➡️  Entropy loss stable")

def analyze_entropy_breakdown(data):
    """Analyze entropy by action type"""
    print("\n🎯 === ENTROPY BREAKDOWN BY ACTION TYPE ===")
    print("=" * 60)
    
    entropy_components = [
        'placement_entropy_mean', 'edge_entropy_mean', 'army_entropy_mean'
    ]
    
    available_entropies = []
    for component in entropy_components:
        if component in data:
            available_entropies.append(component)
    
    if not available_entropies:
        print("❌ No entropy components found!")
        return
    
    print(f"Found entropy components: {available_entropies}")
    
    for component in available_entropies:
        df = data[component]
        if len(df) > 100:
            # Calculate recent trend
            recent_values = df.tail(500)['value']
            early_values = df.head(500)['value']
            
            recent_mean = recent_values.mean()
            early_mean = early_values.mean()
            trend = recent_mean - early_mean
            
            print(f"\n📊 {component.replace('_', ' ').upper()}:")
            print(f"   Early: {early_mean:.4f}")
            print(f"   Recent: {recent_mean:.4f}")
            print(f"   Change: {'+' if trend > 0 else ''}{trend:.4f}")
            
            # Analyze what this means
            if 'army' in component:
                max_entropy = np.log(4)  # 4 army options
                randomness_pct = (recent_mean / max_entropy) * 100
                print(f"   Randomness: {randomness_pct:.1f}% (100% = completely random)")
                
                if randomness_pct > 90:
                    print("   ❌ ARMY CHOICES STILL RANDOM!")
                elif randomness_pct > 70:
                    print("   ⚠️  Army choices mostly random")
                elif randomness_pct > 40:
                    print("   🔍 Army choices becoming decisive")
                else:
                    print("   ✅ Army choices are decisive")
                    
            elif 'placement' in component:
                if trend > 0.1:
                    print("   ⚠️  Placement becoming more exploratory")
                elif trend < -0.1:
                    print("   ✅ Placement becoming more decisive")
                else:
                    print("   ➡️  Placement strategy stable")
                    
            elif 'edge' in component:
                if trend > 0.1:
                    print("   ⚠️  Attack choices becoming more exploratory")
                elif trend < -0.1:
                    print("   ✅ Attack choices becoming more decisive")
                else:
                    print("   ➡️  Attack strategy stable")

def analyze_performance_vs_loss_correlation(data):
    """Analyze correlation between performance and loss metrics"""
    print("\n🔗 === PERFORMANCE VS LOSS CORRELATION ===")
    print("=" * 60)
    
    # Get performance metrics
    performance_metrics = ['win', 'reward']
    loss_metrics = ['loss_mean', 'act_loss_mean', 'crit_loss_mean']
    
    correlations = {}
    
    for perf_metric in performance_metrics:
        if perf_metric not in data:
            continue
            
        perf_df = data[perf_metric]
        
        for loss_metric in loss_metrics:
            if loss_metric not in data:
                continue
                
            loss_df = data[loss_metric]
            
            # Align episodes (take intersection)
            min_episodes = min(len(perf_df), len(loss_df))
            perf_values = perf_df.head(min_episodes)['value'].values
            loss_values = loss_df.head(min_episodes)['value'].values
            
            # Calculate correlation
            correlation = np.corrcoef(perf_values, loss_values)[0, 1]
            correlations[f"{perf_metric}_vs_{loss_metric}"] = correlation
            
            print(f"\n📈 {perf_metric.upper()} vs {loss_metric.upper()}:")
            print(f"   Correlation: {correlation:.3f}")
            
            if abs(correlation) < 0.1:
                print("   ➡️  No significant correlation")
            elif correlation > 0.3:
                print("   ⚠️  POSITIVE correlation - loss increases with performance!")
                print("       This suggests the model is exploring/learning rapidly")
            elif correlation < -0.3:
                print("   ✅ NEGATIVE correlation - loss decreases with performance")
                print("       This is the expected healthy pattern")
            else:
                print("   🔍 Weak correlation - mixed relationship")

def create_trend_plots(data):
    """Create trend plots for key metrics"""
    print("\n📊 === CREATING TREND PLOTS ===")
    print("=" * 60)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('SAGE Model Training Analysis', fontsize=16)
    
    # Plot 1: Performance metrics
    ax1 = axes[0, 0]
    if 'win' in data:
        win_df = data['win']
        # Calculate rolling win rate
        window = 100
        rolling_win_rate = win_df['value'].rolling(window=window, min_periods=1).mean()
        ax1.plot(win_df['episode'], rolling_win_rate * 100, label='Win Rate %', color='green')
    
    if 'reward' in data:
        reward_df = data['reward']
        # Use a different scale for reward
        ax1_twin = ax1.twinx()
        ax1_twin.plot(reward_df['episode'], reward_df['value'], label='Reward', color='blue', alpha=0.7)
        ax1_twin.set_ylabel('Reward', color='blue')
        ax1_twin.tick_params(axis='y', labelcolor='blue')
    
    ax1.set_title('Performance Over Time')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Win Rate %', color='green')
    ax1.tick_params(axis='y', labelcolor='green')
    ax1.legend(loc='upper left')
    
    # Plot 2: Loss components
    ax2 = axes[0, 1]
    loss_components = ['act_loss_mean', 'crit_loss_mean', 'entropy_loss_mean']
    colors = ['red', 'orange', 'purple']
    
    for i, component in enumerate(loss_components):
        if component in data:
            df = data[component]
            ax2.plot(df['episode'], df['value'], label=component.replace('_mean', ''), color=colors[i])
    
    ax2.set_title('Loss Components')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    # Plot 3: Entropy breakdown
    ax3 = axes[1, 0]
    entropy_components = ['placement_entropy_mean', 'edge_entropy_mean', 'army_entropy_mean']
    colors = ['cyan', 'magenta', 'yellow']
    
    for i, component in enumerate(entropy_components):
        if component in data:
            df = data[component]
            ax3.plot(df['episode'], df['value'], label=component.replace('_entropy_mean', ''), color=colors[i])
    
    ax3.set_title('Entropy by Action Type')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Entropy')
    ax3.legend()
    
    # Plot 4: Overall loss vs performance
    ax4 = axes[1, 1]
    if 'loss_mean' in data and 'reward' in data:
        loss_df = data['loss_mean']
        reward_df = data['reward']
        
        ax4.scatter(reward_df['value'], loss_df['value'], alpha=0.5, s=1)
        ax4.set_title('Loss vs Reward Scatter')
        ax4.set_xlabel('Reward')
        ax4.set_ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig('sage_model_analysis.png', dpi=150, bbox_inches='tight')
    print("📊 Plots saved as 'sage_model_analysis.png'")
    
    return fig

def diagnose_loss_entropy_paradox(data):
    """Specifically diagnose why loss and entropy are increasing while performance improves"""
    print("\n🩺 === DIAGNOSING THE LOSS/ENTROPY PARADOX ===")
    print("=" * 60)
    
    # Check if we have the necessary data
    required_metrics = ['win', 'reward', 'loss_mean']
    missing_metrics = [m for m in required_metrics if m not in data]
    
    if missing_metrics:
        print(f"❌ Missing required metrics: {missing_metrics}")
        return
    
    # Analyze recent trends
    win_df = data['win']
    reward_df = data['reward']
    loss_df = data['loss_mean']
    
    # Get recent data (last 1000 episodes)
    recent_episodes = 1000
    
    recent_wins = win_df.tail(recent_episodes)['value'].mean()
    recent_rewards = reward_df.tail(recent_episodes)['value'].mean()
    recent_loss = loss_df.tail(recent_episodes)['value'].mean()
    
    early_wins = win_df.head(recent_episodes)['value'].mean()
    early_rewards = reward_df.head(recent_episodes)['value'].mean()
    early_loss = loss_df.head(recent_episodes)['value'].mean()
    
    print(f"📊 TREND ANALYSIS (last {recent_episodes} vs first {recent_episodes} episodes):")
    print(f"   Win Rate: {early_wins:.3f} → {recent_wins:.3f} ({'+' if recent_wins > early_wins else ''}{recent_wins - early_wins:.3f})")
    print(f"   Reward: {early_rewards:.2f} → {recent_rewards:.2f} ({'+' if recent_rewards > early_rewards else ''}{recent_rewards - early_rewards:.2f})")
    print(f"   Loss: {early_loss:.4f} → {recent_loss:.4f} ({'+' if recent_loss > early_loss else ''}{recent_loss - early_loss:.4f})")
    
    # Diagnose the pattern
    performance_improving = (recent_wins > early_wins) and (recent_rewards > early_rewards)
    loss_increasing = recent_loss > early_loss
    
    print(f"\n🔍 PATTERN IDENTIFICATION:")
    print(f"   Performance improving: {'✅ YES' if performance_improving else '❌ NO'}")
    print(f"   Loss increasing: {'⚠️ YES' if loss_increasing else '✅ NO'}")
    
    if performance_improving and loss_increasing:
        print(f"\n💡 LIKELY EXPLANATIONS:")
        print(f"   1. 🎯 Value Function Lag: The critic (value network) is struggling")
        print(f"      to keep up with rapid policy improvements")
        print(f"   2. 🔄 Exploration Phase: The agent is discovering new strategies")
        print(f"      and needs higher entropy to explore effectively")
        print(f"   3. 📈 Non-stationary Environment: As the agent gets better,")
        print(f"      the effective environment changes (opponent adaptation)")
        print(f"   4. 🎲 Stochastic Success: Winning through high-variance strategies")
        print(f"      that are hard for the value function to predict")
        
        # Check entropy trends
        if 'army_entropy_mean' in data:
            army_entropy = data['army_entropy_mean']
            recent_army_entropy = army_entropy.tail(recent_episodes)['value'].mean()
            early_army_entropy = army_entropy.head(recent_episodes)['value'].mean()
            
            print(f"\n🎲 ENTROPY ANALYSIS:")
            print(f"   Army entropy: {early_army_entropy:.3f} → {recent_army_entropy:.3f}")
            
            if recent_army_entropy > early_army_entropy:
                print(f"   📊 Army entropy INCREASING suggests the agent is:")
                print(f"      - Discovering army amounts work better in different situations")
                print(f"      - Still in active learning phase for army allocation")
                print(f"      - May need longer training to converge on optimal army strategy")
        
        print(f"\n🔧 RECOMMENDATIONS:")
        print(f"   1. ✅ Continue training - this pattern can be normal during learning")
        print(f"   2. 📊 Monitor value loss specifically - if it's the main contributor")
        print(f"   3. 🎯 Consider slightly reducing entropy coefficients if trend continues")
        print(f"   4. 📈 Track if loss eventually plateaus as performance stabilizes")

def main():
    log_file = "analysis/logs/sage_model_stable/events.out.tfevents.1755036616.Deskie.17888.1"
    
    print("🔍 Detailed SAGE Model Analysis")
    print("=" * 70)
    print("Investigating the loss/entropy vs performance paradox...")
    
    # Load and analyze data
    data = extract_all_metrics(log_file)
    
    if not data:
        print("❌ Failed to load TensorBoard data")
        return
    
    print(f"\n📋 Available metrics: {len(data)} total")
    
    # Run all analyses
    analyze_loss_components(data)
    analyze_entropy_breakdown(data)
    analyze_performance_vs_loss_correlation(data)
    diagnose_loss_entropy_paradox(data)
    
    # Create plots
    try:
        create_trend_plots(data)
    except Exception as e:
        print(f"⚠️ Could not create plots: {e}")
    
    print("\n" + "=" * 70)
    print("🎯 === SUMMARY ===")
    print("This analysis should help explain why your SAGE model shows")
    print("improving performance alongside increasing loss and entropy.")
    print("Check the plots and recommendations above for actionable insights!")

if __name__ == "__main__":
    main()
