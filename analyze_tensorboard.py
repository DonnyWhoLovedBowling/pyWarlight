"""
TensorBoard Analysis Script for Autoregressive Transformer V1
Analyzes training metrics and provides recommendations
"""

import os
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
from pathlib import Path

def load_tensorboard_data(log_file):
    """Load all scalar data from a TensorBoard event file"""
    ea = event_accumulator.EventAccumulator(log_file)
    ea.Reload()
    
    # Get all available tags
    scalar_tags = ea.Tags()['scalars']
    
    data = {}
    for tag in scalar_tags:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        data[tag] = {'steps': steps, 'values': values}
    
    return data, scalar_tags

def compute_statistics(values):
    """Compute comprehensive statistics for a metric"""
    if not values or len(values) == 0:
        return {}
    
    values = np.array(values)
    stats = {
        'mean': np.mean(values),
        'std': np.std(values),
        'min': np.min(values),
        'max': np.max(values),
        'median': np.median(values),
        'recent_mean': np.mean(values[-100:]) if len(values) >= 100 else np.mean(values),
        'recent_std': np.std(values[-100:]) if len(values) >= 100 else np.std(values),
        'trend': 'increasing' if len(values) > 10 and np.mean(values[-10:]) > np.mean(values[:10]) else 'decreasing' if len(values) > 10 else 'stable',
        'count': len(values)
    }
    
    # Compute trend strength (linear regression slope)
    if len(values) > 1:
        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)
        stats['trend_slope'] = coeffs[0]
    else:
        stats['trend_slope'] = 0.0
    
    return stats

def analyze_training_stability(data):
    """Analyze training stability metrics"""
    print("\n" + "="*80)
    print("TRAINING STABILITY ANALYSIS")
    print("="*80)
    
    stability_metrics = []
    
    # Check for policy loss
    if 'policy_loss' in data:
        policy_stats = compute_statistics(data['policy_loss']['values'])
        print(f"\n📊 Policy Loss:")
        print(f"   Mean: {policy_stats['mean']:.4f} ± {policy_stats['std']:.4f}")
        print(f"   Recent (last 100): {policy_stats['recent_mean']:.4f} ± {policy_stats['recent_std']:.4f}")
        print(f"   Trend: {policy_stats['trend']} (slope: {policy_stats['trend_slope']:.6f})")
        
        if policy_stats['std'] > policy_stats['mean'] * 0.5:
            stability_metrics.append("⚠️ High policy loss variance detected")
    
    # Check for value loss
    if 'value_loss' in data:
        value_stats = compute_statistics(data['value_loss']['values'])
        print(f"\n📊 Value Loss:")
        print(f"   Mean: {value_stats['mean']:.4f} ± {value_stats['std']:.4f}")
        print(f"   Recent (last 100): {value_stats['recent_mean']:.4f} ± {value_stats['recent_std']:.4f}")
        print(f"   Trend: {value_stats['trend']} (slope: {value_stats['trend_slope']:.6f})")
    
    # Check for total loss
    if 'total_loss' in data:
        total_stats = compute_statistics(data['total_loss']['values'])
        print(f"\n📊 Total Loss:")
        print(f"   Mean: {total_stats['mean']:.4f} ± {total_stats['std']:.4f}")
        print(f"   Recent (last 100): {total_stats['recent_mean']:.4f} ± {total_stats['recent_std']:.4f}")
        print(f"   Trend: {total_stats['trend']} (slope: {total_stats['trend_slope']:.6f})")
    
    # Check KL divergence
    if 'kl_divergence' in data:
        kl_stats = compute_statistics(data['kl_divergence']['values'])
        print(f"\n📊 KL Divergence:")
        print(f"   Mean: {kl_stats['mean']:.6f} ± {kl_stats['std']:.6f}")
        print(f"   Recent: {kl_stats['recent_mean']:.6f}")
        print(f"   Max: {kl_stats['max']:.6f}")
        
        if kl_stats['recent_mean'] > 0.03:
            stability_metrics.append("⚠️ High KL divergence - policy changing too rapidly")
        elif kl_stats['recent_mean'] < 0.001:
            stability_metrics.append("⚠️ Very low KL divergence - policy may be stuck")
    
    # Check PPO ratio
    if 'ppo_ratio' in data:
        ratio_stats = compute_statistics(data['ppo_ratio']['values'])
        print(f"\n📊 PPO Ratio:")
        print(f"   Mean: {ratio_stats['mean']:.4f} ± {ratio_stats['std']:.4f}")
        print(f"   Range: [{ratio_stats['min']:.4f}, {ratio_stats['max']:.4f}]")
        
        if ratio_stats['mean'] < 0.95 or ratio_stats['mean'] > 1.05:
            stability_metrics.append(f"⚠️ PPO ratio off-center (mean: {ratio_stats['mean']:.4f})")
    
    return stability_metrics

def analyze_exploration(data):
    """Analyze exploration vs exploitation balance"""
    print("\n" + "="*80)
    print("EXPLORATION ANALYSIS")
    print("="*80)
    
    exploration_metrics = []
    
    # Entropy metrics
    if 'placement_entropy' in data:
        place_stats = compute_statistics(data['placement_entropy']['values'])
        print(f"\n📊 Placement Entropy:")
        print(f"   Mean: {place_stats['mean']:.4f} ± {place_stats['std']:.4f}")
        print(f"   Recent: {place_stats['recent_mean']:.4f}")
        print(f"   Trend: {place_stats['trend']} (slope: {place_stats['trend_slope']:.6f})")
        
        if place_stats['recent_mean'] < 0.5:
            exploration_metrics.append("⚠️ Low placement entropy - agent may be too deterministic")
        elif place_stats['recent_mean'] > 3.0:
            exploration_metrics.append("⚠️ High placement entropy - agent may be too random")
    
    if 'attack_entropy' in data:
        attack_stats = compute_statistics(data['attack_entropy']['values'])
        print(f"\n📊 Attack Entropy:")
        print(f"   Mean: {attack_stats['mean']:.4f} ± {attack_stats['std']:.4f}")
        print(f"   Recent: {attack_stats['recent_mean']:.4f}")
        print(f"   Trend: {attack_stats['trend']} (slope: {attack_stats['trend_slope']:.6f})")
        
        if attack_stats['recent_mean'] < 0.5:
            exploration_metrics.append("⚠️ Low attack entropy - agent may be too deterministic")
    
    if 'total_entropy_loss' in data:
        entropy_loss_stats = compute_statistics(data['total_entropy_loss']['values'])
        print(f"\n📊 Total Entropy Loss:")
        print(f"   Mean: {entropy_loss_stats['mean']:.4f}")
        print(f"   Trend: {entropy_loss_stats['trend']}")
    
    return exploration_metrics

def analyze_performance(data):
    """Analyze agent performance metrics"""
    print("\n" + "="*80)
    print("PERFORMANCE ANALYSIS")
    print("="*80)
    
    performance_metrics = []
    
    # Win rate
    if 'win' in data:
        wins = data['win']['values']
        win_stats = compute_statistics(wins)
        print(f"\n🏆 Win Rate:")
        print(f"   Overall: {win_stats['mean']*100:.2f}%")
        print(f"   Recent (last 100): {win_stats['recent_mean']*100:.2f}%")
        print(f"   Trend: {win_stats['trend']}")
        
        if win_stats['recent_mean'] < 0.3:
            performance_metrics.append("❌ Low win rate - agent struggling")
        elif win_stats['recent_mean'] > 0.7:
            performance_metrics.append("✅ High win rate - agent performing well")
        
        # Check if improving
        if len(wins) > 100:
            early_winrate = np.mean(wins[:50])
            recent_winrate = np.mean(wins[-50:])
            improvement = recent_winrate - early_winrate
            print(f"   Improvement: {improvement*100:+.2f}% (from {early_winrate*100:.2f}% to {recent_winrate*100:.2f}%)")
    
    # Combat metrics
    if 'attacks_per_turn' in data:
        attack_stats = compute_statistics(data['attacks_per_turn']['values'])
        print(f"\n⚔️ Attacks per Turn:")
        print(f"   Mean: {attack_stats['mean']:.2f} ± {attack_stats['std']:.2f}")
        print(f"   Recent: {attack_stats['recent_mean']:.2f}")
    
    if 'won_battles_per_turn' in data:
        won_stats = compute_statistics(data['won_battles_per_turn']['values'])
        print(f"\n✅ Won Battles per Turn:")
        print(f"   Mean: {won_stats['mean']:.2f} ± {won_stats['std']:.2f}")
        print(f"   Recent: {won_stats['recent_mean']:.2f}")
        
        if 'attacks_per_turn' in data:
            attack_mean = compute_statistics(data['attacks_per_turn']['values'])['recent_mean']
            if attack_mean > 0:
                win_rate = won_stats['recent_mean'] / attack_mean
                print(f"   Battle Success Rate: {win_rate*100:.2f}%")
    
    # Territory control
    if 'gained_regions' in data and 'lost_regions' in data:
        gained_stats = compute_statistics(data['gained_regions']['values'])
        lost_stats = compute_statistics(data['lost_regions']['values'])
        net_gain = gained_stats['recent_mean'] - lost_stats['recent_mean']
        
        print(f"\n🌍 Territory Control:")
        print(f"   Gained per Turn: {gained_stats['recent_mean']:.2f}")
        print(f"   Lost per Turn: {lost_stats['recent_mean']:.2f}")
        print(f"   Net: {net_gain:+.2f}")
        
        if net_gain < -0.1:
            performance_metrics.append("❌ Losing territory overall")
        elif net_gain > 0.1:
            performance_metrics.append("✅ Gaining territory overall")
    
    # Army efficiency
    if 'armies_per_attack' in data:
        army_stats = compute_statistics(data['armies_per_attack']['values'])
        print(f"\n🪖 Armies per Attack:")
        print(f"   Mean: {army_stats['mean']:.2f} ± {army_stats['std']:.2f}")
        print(f"   Recent: {army_stats['recent_mean']:.2f}")
    
    if 'army_difference' in data:
        diff_stats = compute_statistics(data['army_difference']['values'])
        print(f"\n⚖️ Army Difference:")
        print(f"   Mean: {diff_stats['mean']:.2f} ± {diff_stats['std']:.2f}")
        print(f"   Recent: {diff_stats['recent_mean']:.2f}")
    
    # Aggression metrics
    if 'turn_with_attack' in data:
        turn_attack_stats = compute_statistics(data['turn_with_attack']['values'])
        print(f"\n🎯 Turn with Attack:")
        print(f"   Mean: {turn_attack_stats['mean']*100:.2f}%")
        print(f"   Recent: {turn_attack_stats['recent_mean']*100:.2f}%")
        
        if turn_attack_stats['recent_mean'] < 0.5:
            performance_metrics.append("⚠️ Low aggression - agent playing too passively")
    
    if 'turn_with_mult_attacks' in data:
        mult_attack_stats = compute_statistics(data['turn_with_mult_attacks']['values'])
        print(f"\n🔥 Turn with Multiple Attacks:")
        print(f"   Mean: {mult_attack_stats['mean']*100:.2f}%")
        print(f"   Recent: {mult_attack_stats['recent_mean']*100:.2f}%")
    
    return performance_metrics

def analyze_gradient_health(data):
    """Analyze gradient flow and optimization health"""
    print("\n" + "="*80)
    print("GRADIENT & OPTIMIZATION HEALTH")
    print("="*80)
    
    gradient_metrics = []
    
    if 'grad_norm_raw' in data:
        grad_stats = compute_statistics(data['grad_norm_raw']['values'])
        print(f"\n📈 Gradient Norm (Raw):")
        print(f"   Mean: {grad_stats['mean']:.4f} ± {grad_stats['std']:.4f}")
        print(f"   Recent: {grad_stats['recent_mean']:.4f}")
        print(f"   Range: [{grad_stats['min']:.4f}, {grad_stats['max']:.4f}]")
        
        if grad_stats['recent_mean'] < 0.001:
            gradient_metrics.append("⚠️ Vanishing gradients detected")
        elif grad_stats['recent_mean'] > 10.0:
            gradient_metrics.append("⚠️ Exploding gradients detected")
    
    if 'grad_norm_clipped' in data:
        clipped_stats = compute_statistics(data['grad_norm_clipped']['values'])
        print(f"\n✂️ Gradient Norm (Clipped):")
        print(f"   Mean: {clipped_stats['mean']:.4f}")
        print(f"   Recent: {clipped_stats['recent_mean']:.4f}")
    
    if 'grad_scale_factor' in data:
        scale_stats = compute_statistics(data['grad_scale_factor']['values'])
        print(f"\n⚖️ Gradient Scale Factor:")
        print(f"   Mean: {scale_stats['mean']:.4f}")
        print(f"   Recent: {scale_stats['recent_mean']:.4f}")
    
    return gradient_metrics

def generate_recommendations(data, stability_issues, exploration_issues, performance_issues, gradient_issues):
    """Generate actionable recommendations based on analysis"""
    print("\n" + "="*80)
    print("🎯 RECOMMENDATIONS")
    print("="*80)
    
    recommendations = []
    
    # Check overall training progress
    if 'win' in data:
        wins = data['win']['values']
        if len(wins) > 100:
            recent_winrate = np.mean(wins[-100:])
            early_winrate = np.mean(wins[:100]) if len(wins) >= 100 else np.mean(wins[:50])
            
            if recent_winrate > 0.6:
                recommendations.append(("✅ CONTINUE TRAINING", "Agent is performing well. Continue current approach."))
            elif recent_winrate - early_winrate > 0.1:
                recommendations.append(("✅ CONTINUE TRAINING", "Agent is improving. Continue current training."))
            elif recent_winrate < 0.3 and len(wins) > 500:
                recommendations.append(("❌ MAJOR CHANGES NEEDED", "Agent not learning effectively after many episodes."))
    
    # Stability recommendations
    if stability_issues:
        for issue in stability_issues:
            if "KL divergence" in issue and "too rapidly" in issue:
                recommendations.append(("🔧 REDUCE LEARNING RATE", 
                    "High KL divergence suggests policy changing too fast. Reduce learning rate by 30-50%."))
            elif "policy may be stuck" in issue:
                recommendations.append(("🔧 INCREASE LEARNING RATE", 
                    "Very low KL divergence suggests convergence. Consider increasing learning rate or entropy bonus."))
            elif "High policy loss variance" in issue:
                recommendations.append(("🔧 INCREASE BATCH SIZE", 
                    "High loss variance suggests unstable updates. Increase batch_size or reduce learning rate."))
    
    # Exploration recommendations
    if exploration_issues:
        for issue in exploration_issues:
            if "too deterministic" in issue:
                recommendations.append(("🔧 INCREASE ENTROPY COEFFICIENT", 
                    "Agent is too deterministic. Increase entropy coefficients by 20-50%."))
            elif "too random" in issue:
                recommendations.append(("🔧 DECREASE ENTROPY COEFFICIENT", 
                    "Agent is too random. Decrease entropy coefficients by 20-30%."))
    
    # Performance recommendations
    if performance_issues:
        for issue in performance_issues:
            if "Low aggression" in issue:
                recommendations.append(("🔧 ADJUST REWARD SHAPING", 
                    "Agent playing too passively. Consider increasing rewards for attacks/territory gains."))
            elif "Losing territory" in issue:
                recommendations.append(("🔧 REVIEW STRATEGY", 
                    "Agent losing territory. May need longer training or reward adjustments."))
    
    # Gradient recommendations
    if gradient_issues:
        for issue in gradient_issues:
            if "Vanishing gradients" in issue:
                recommendations.append(("🔧 INCREASE LEARNING RATE", 
                    "Vanishing gradients detected. Increase learning rate or check network architecture."))
            elif "Exploding gradients" in issue:
                recommendations.append(("🔧 REDUCE LEARNING RATE", 
                    "Exploding gradients. Reduce learning rate and ensure gradient clipping is active."))
    
    # Check entropy trend
    if 'placement_entropy' in data and 'attack_entropy' in data:
        place_stats = compute_statistics(data['placement_entropy']['values'])
        attack_stats = compute_statistics(data['attack_entropy']['values'])
        
        if place_stats['trend'] == 'decreasing' and attack_stats['trend'] == 'decreasing':
            if place_stats['recent_mean'] > 1.0:
                recommendations.append(("✅ ENTROPY ANNEALING WORKING", 
                    "Entropy naturally decreasing as agent learns. This is expected behavior."))
    
    # Training duration recommendations
    if 'win' in data:
        num_games = len(data['win']['values'])
        if num_games < 1000:
            recommendations.append(("⏳ CONTINUE TRAINING", 
                f"Only {num_games} games played. Consider training for 2000-5000 games minimum."))
        elif num_games > 5000:
            recent_mean = compute_statistics(data['win']['values'])['recent_mean']
            if recent_mean < 0.4:
                recommendations.append(("🔍 INVESTIGATE MODEL", 
                    "Trained for many games but performance still low. May need architecture changes."))
    
    # Print recommendations
    if recommendations:
        for i, (title, desc) in enumerate(recommendations, 1):
            print(f"\n{i}. {title}")
            print(f"   {desc}")
    else:
        print("\n✅ No critical issues detected. Monitor training and continue.")
    
    return recommendations

def create_summary_report(data, tags):
    """Create a summary report"""
    print("\n" + "="*80)
    print("📝 SUMMARY")
    print("="*80)
    
    print(f"\nTotal metrics logged: {len(tags)}")
    print(f"Metrics available: {', '.join(sorted(tags))}")
    
    if 'win' in data:
        num_games = len(data['win']['values'])
        print(f"\nTotal games played: {num_games}")
        
        if num_games > 0:
            recent_games = min(100, num_games)
            recent_winrate = np.mean(data['win']['values'][-recent_games:])
            print(f"Recent win rate (last {recent_games} games): {recent_winrate*100:.2f}%")

def main():
    log_file = r"C:\Users\pcvan\Projects\pyWarlight\analysis\logs\autoregressive_transformer_v1\events.out.tfevents.1759817477.Deskie.10800.1"
    
    print("="*80)
    print("TENSORBOARD ANALYSIS: Autoregressive Transformer V1")
    print("="*80)
    print(f"\nAnalyzing: {log_file}")
    
    # Load data
    print("\n📂 Loading TensorBoard data...")
    data, tags = load_tensorboard_data(log_file)
    print(f"✅ Loaded {len(tags)} metrics")
    
    # Run analyses
    stability_issues = analyze_training_stability(data)
    exploration_issues = analyze_exploration(data)
    performance_issues = analyze_performance(data)
    gradient_issues = analyze_gradient_health(data)
    
    # Generate recommendations
    recommendations = generate_recommendations(
        data, stability_issues, exploration_issues, 
        performance_issues, gradient_issues
    )
    
    # Summary
    create_summary_report(data, tags)
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)

if __name__ == "__main__":
    main()

