#!/usr/bin/env python3
"""
TensorBoard Analysis Script for pyWarlight RL Training
Provides perceptive analysis and recommendations based on training metrics.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    import tensorflow as tf
except ImportError:
    print("❌ Missing dependencies. Install with:")
    print("   pip install tensorboard tensorflow matplotlib seaborn pandas")
    sys.exit(1)


class TensorBoardAnalyzer:
    """Comprehensive analysis of TensorBoard training data for RL agents."""
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.data = {}
        self.metrics = defaultdict(list)
        self.analysis = {}
        
    def load_data(self) -> bool:
        """Load all scalar data from TensorBoard events files."""
        print(f"🔍 Loading data from: {self.log_dir}")
        
        # Find all event files
        event_files = list(self.log_dir.glob("events.out.tfevents.*"))
        if not event_files:
            print(f"❌ No TensorBoard event files found in {self.log_dir}")
            return False
            
        print(f"📊 Found {len(event_files)} event file(s)")
        
        for event_file in event_files:
            try:
                ea = EventAccumulator(str(event_file))
                ea.Reload()
                
                # Extract all scalar metrics
                for tag in ea.Tags()['scalars']:
                    events = ea.Scalars(tag)
                    values = [event.value for event in events]
                    steps = [event.step for event in events]
                    
                    self.metrics[tag] = {
                        'values': values,
                        'steps': steps,
                        'latest': values[-1] if values else 0,
                        'mean': np.mean(values) if values else 0,
                        'std': np.std(values) if values else 0,
                        'trend': self._calculate_trend(values) if len(values) > 10 else 'insufficient_data'
                    }
                    
                print(f"✅ Loaded {len(self.metrics)} metrics from {event_file.name}")
                
            except Exception as e:
                print(f"⚠️  Error loading {event_file.name}: {e}")
                
        return len(self.metrics) > 0
    
    def _calculate_trend(self, values: List[float], window: int = 50) -> str:
        """Calculate trend in recent values."""
        if len(values) < window:
            window = len(values) // 2
            
        if window < 5:
            return 'insufficient_data'
            
        recent = values[-window:]
        older = values[-2*window:-window] if len(values) >= 2*window else values[:window]
        
        recent_mean = np.mean(recent)
        older_mean = np.mean(older)
        
        change_pct = (recent_mean - older_mean) / (abs(older_mean) + 1e-8) * 100
        
        if abs(change_pct) < 2:
            return 'stable'
        elif change_pct > 5:
            return 'improving'
        elif change_pct > 2:
            return 'slightly_improving'
        elif change_pct < -5:
            return 'declining'
        else:
            return 'slightly_declining'
    
    def analyze_training_stability(self) -> Dict:
        """Analyze training stability and convergence."""
        stability = {
            'loss_stability': 'unknown',
            'value_stability': 'unknown',
            'policy_stability': 'unknown',
            'entropy_health': 'unknown',
            'overall_assessment': 'unknown'
        }
        
        # Loss analysis
        if 'loss_mean' in self.metrics:
            loss_std = self.metrics['loss_mean']['std']
            loss_mean = self.metrics['loss_mean']['mean']
            cv = loss_std / (loss_mean + 1e-8)  # Coefficient of variation
            
            if cv < 0.2:
                stability['loss_stability'] = 'very_stable'
            elif cv < 0.5:
                stability['loss_stability'] = 'stable'
            elif cv < 1.0:
                stability['loss_stability'] = 'moderately_unstable'
            else:
                stability['loss_stability'] = 'unstable'
        
        # Value function analysis
        value_metrics = ['crit_loss_mean', 'value_mean', 'value_pred_mean']
        value_stability_score = 0
        for metric in value_metrics:
            if metric in self.metrics:
                if self.metrics[metric]['trend'] in ['stable', 'improving']:
                    value_stability_score += 1
                elif self.metrics[metric]['trend'] == 'declining':
                    value_stability_score -= 1
        
        if value_stability_score >= 2:
            stability['value_stability'] = 'good'
        elif value_stability_score >= 0:
            stability['value_stability'] = 'acceptable'
        else:
            stability['value_stability'] = 'concerning'
        
        # Policy stability
        if 'act_loss_mean' in self.metrics:
            policy_trend = self.metrics['act_loss_mean']['trend']
            stability['policy_stability'] = policy_trend
        
        # Entropy health (exploration)
        entropy_metrics = ['placement_entropy_mean', 'edge_entropy_mean', 'army_entropy_mean']
        entropy_health = []
        for metric in entropy_metrics:
            if metric in self.metrics:
                latest = self.metrics[metric]['latest']
                if latest > 0.1:  # Healthy exploration
                    entropy_health.append('good')
                elif latest > 0.05:
                    entropy_health.append('moderate')
                else:
                    entropy_health.append('low')
        
        if entropy_health:
            if entropy_health.count('good') >= len(entropy_health) // 2:
                stability['entropy_health'] = 'good'
            elif entropy_health.count('low') >= len(entropy_health) // 2:
                stability['entropy_health'] = 'low'
            else:
                stability['entropy_health'] = 'moderate'
        
        # Overall assessment
        scores = {
            'very_stable': 3, 'stable': 2, 'good': 2, 'improving': 2,
            'acceptable': 1, 'moderate': 1, 'slightly_improving': 1,
            'moderately_unstable': -1, 'concerning': -2, 'low': -1,
            'unstable': -3, 'declining': -2, 'slightly_declining': -1
        }
        
        total_score = sum(scores.get(v, 0) for v in stability.values() if isinstance(v, str))
        
        if total_score >= 4:
            stability['overall_assessment'] = 'excellent'
        elif total_score >= 2:
            stability['overall_assessment'] = 'good'
        elif total_score >= 0:
            stability['overall_assessment'] = 'acceptable'
        elif total_score >= -2:
            stability['overall_assessment'] = 'concerning'
        else:
            stability['overall_assessment'] = 'poor'
        
        return stability
    
    def analyze_performance_trends(self) -> Dict:
        """Analyze agent performance and learning progress."""
        performance = {
            'win_rate': 'unknown',
            'win_rate_trend': 'unknown',
            'strategic_development': 'unknown',
            'learning_efficiency': 'unknown'
        }
        
        # Win rate analysis
        if 'win' in self.metrics:
            win_rate = self.metrics['win']['mean']
            win_trend = self.metrics['win']['trend']
            
            performance['win_rate'] = f"{win_rate:.1%}"
            performance['win_rate_trend'] = win_trend
        
        # Strategic development (combat effectiveness)
        strategic_indicators = [
            'won_battles_per_turn', 'armies_per_attack', 'attacks_per_turn',
            'turn_with_attack', 'turn_with_mult_attacks'
        ]
        
        strategic_scores = []
        for indicator in strategic_indicators:
            if indicator in self.metrics:
                trend = self.metrics[indicator]['trend']
                if trend in ['improving', 'stable']:
                    strategic_scores.append(1)
                elif trend in ['slightly_improving']:
                    strategic_scores.append(0.5)
                else:
                    strategic_scores.append(-1)
        
        if strategic_scores:
            avg_strategic = np.mean(strategic_scores)
            if avg_strategic > 0.5:
                performance['strategic_development'] = 'excellent'
            elif avg_strategic > 0:
                performance['strategic_development'] = 'good'
            elif avg_strategic > -0.5:
                performance['strategic_development'] = 'moderate'
            else:
                performance['strategic_development'] = 'poor'
        
        # Learning efficiency (how quickly losses decrease)
        if 'loss_mean' in self.metrics and len(self.metrics['loss_mean']['values']) > 100:
            early_loss = np.mean(self.metrics['loss_mean']['values'][:50])
            recent_loss = np.mean(self.metrics['loss_mean']['values'][-50:])
            improvement = (early_loss - recent_loss) / (early_loss + 1e-8)
            
            if improvement > 0.3:
                performance['learning_efficiency'] = 'excellent'
            elif improvement > 0.1:
                performance['learning_efficiency'] = 'good'
            elif improvement > 0:
                performance['learning_efficiency'] = 'moderate'
            else:
                performance['learning_efficiency'] = 'poor'
        
        return performance
    
    def generate_recommendations(self, stability: Dict, performance: Dict) -> List[str]:
        """Generate specific recommendations based on analysis."""
        recommendations = []
        
        # Stability recommendations
        if stability['loss_stability'] == 'unstable':
            recommendations.append("🔧 CRITICAL: Reduce learning rate - loss is highly unstable")
            recommendations.append("🔧 Consider gradient clipping or lower batch size")
        
        if stability['value_stability'] == 'concerning':
            recommendations.append("🎯 VALUE FUNCTION: Increase value loss coefficient")
            recommendations.append("🎯 Consider value clipping range adjustment")
        
        if stability['entropy_health'] == 'low':
            recommendations.append("🎲 EXPLORATION: Increase entropy coefficients")
            recommendations.append("🎲 Consider higher temperature or epsilon-greedy exploration")
        elif stability['entropy_health'] == 'good' and 'win_rate_trend' in performance and performance['win_rate_trend'] == 'declining':
            recommendations.append("⚖️  EXPLOITATION: Consider reducing entropy coefficients for more exploitation")
        
        # Performance recommendations
        if performance.get('win_rate_trend') == 'declining':
            recommendations.append("📉 PERFORMANCE DROP: Check for overfitting or environment changes")
            recommendations.append("📉 Consider early stopping or model regularization")
        
        if performance.get('strategic_development') == 'poor':
            recommendations.append("⚔️  STRATEGY: Review reward function for combat effectiveness")
            recommendations.append("⚔️  Consider curriculum learning or shaped rewards")
        
        if performance.get('learning_efficiency') == 'poor':
            recommendations.append("🚀 LEARNING: Increase learning rate or adjust architecture")
            recommendations.append("🚀 Check data pipeline and batch composition")
        
        # Model-specific recommendations (residual_percentage_4options)
        if '4options' in str(self.log_dir):
            recommendations.append("📊 ARMY SELECTION: Monitor army choice distribution for bias")
            recommendations.append("📊 Verify percentage-based army selection is working correctly")
        
        # PPO-specific recommendations
        if 'ratio_mean' in self.metrics:
            ratio_mean = self.metrics['ratio_mean']['latest']
            if ratio_mean > 1.5:
                recommendations.append("📏 PPO RATIO: Decrease clip epsilon - ratio too high")
            elif ratio_mean < 0.8:
                recommendations.append("📏 PPO RATIO: Increase clip epsilon - ratio too conservative")
        
        # General recommendations based on overall assessment
        if stability['overall_assessment'] == 'poor':
            recommendations.append("🚨 URGENT: Consider architecture changes or hyperparameter reset")
        elif stability['overall_assessment'] == 'excellent':
            recommendations.append("✨ EXCELLENT: Consider increasing model complexity or harder opponents")
        
        if not recommendations:
            recommendations.append("✅ Training appears stable - continue current configuration")
        
        return recommendations
    
    def create_visualizations(self) -> None:
        """Create comprehensive visualization plots."""
        if not self.metrics:
            print("❌ No data to visualize")
            return
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        
        # Key metrics for visualization
        key_metrics = {
            'Training Progress': ['loss_mean', 'act_loss_mean', 'crit_loss_mean'],
            'Performance': ['win', 'attacks_per_turn', 'won_battles_per_turn'],
            'Exploration': ['placement_entropy_mean', 'edge_entropy_mean', 'army_entropy_mean'],
            'PPO Health': ['ratio_mean', 'advantage_mean', 'value_pred_mean']
        }
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'RL Training Analysis: {self.log_dir.name}', fontsize=16, fontweight='bold')
        
        for idx, (category, metrics) in enumerate(key_metrics.items()):
            row, col = idx // 2, idx % 2
            ax = axes[row, col]
            
            for metric in metrics:
                if metric in self.metrics:
                    steps = self.metrics[metric]['steps']
                    values = self.metrics[metric]['values']
                    
                    # Smooth the line with rolling average
                    if len(values) > 10:
                        window = min(len(values) // 10, 50)
                        smoothed = pd.Series(values).rolling(window=window, center=True).mean()
                        ax.plot(steps, smoothed, label=metric.replace('_', ' ').title(), linewidth=2)
                    else:
                        ax.plot(steps, values, label=metric.replace('_', ' ').title(), linewidth=2)
            
            ax.set_title(category, fontweight='bold')
            ax.set_xlabel('Training Steps')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        output_path = self.log_dir / 'training_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 Visualization saved to: {output_path}")
        
        # Show plot if running interactively
        try:
            plt.show()
        except:
            pass  # In case display is not available
    
    def generate_report(self) -> str:
        """Generate comprehensive analysis report."""
        if not self.metrics:
            return "❌ No data available for analysis"
        
        stability = self.analyze_training_stability()
        performance = self.analyze_performance_trends()
        recommendations = self.generate_recommendations(stability, performance)
        
        # Calculate training duration and episodes
        max_steps = max((max(m['steps']) for m in self.metrics.values() if m['steps']), default=0)
        total_episodes = max_steps
        
        report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         🎮 PYWARLIGHT RL TRAINING ANALYSIS                   ║
║                              {self.log_dir.name:<46} ║
╚══════════════════════════════════════════════════════════════════════════════╝

📊 TRAINING OVERVIEW
├─ Total Episodes: {total_episodes:,}
├─ Metrics Tracked: {len(self.metrics)}
├─ Model Type: {"Residual + Percentage Army Selection" if "residual_percentage" in str(self.log_dir) else "Unknown"}
└─ Army Options: {"4 percentage options (25%, 50%, 75%, 100%)" if "4options" in str(self.log_dir) else "Unknown"}

🏥 TRAINING STABILITY ANALYSIS
├─ Loss Stability: {stability['loss_stability'].replace('_', ' ').title()} 
├─ Value Function: {stability['value_stability'].replace('_', ' ').title()}
├─ Policy Stability: {stability['policy_stability'].replace('_', ' ').title()}
├─ Exploration Health: {stability['entropy_health'].replace('_', ' ').title()}
└─ Overall Assessment: {stability['overall_assessment'].replace('_', ' ').title()} 

🎯 PERFORMANCE ANALYSIS
├─ Win Rate: {performance.get('win_rate', 'Unknown')}
├─ Win Rate Trend: {performance.get('win_rate_trend', 'Unknown').replace('_', ' ').title()}
├─ Strategic Development: {performance.get('strategic_development', 'Unknown').replace('_', ' ').title()}
└─ Learning Efficiency: {performance.get('learning_efficiency', 'Unknown').replace('_', ' ').title()}

📈 KEY METRICS SUMMARY
"""
        
        # Add key metrics
        important_metrics = [
            'win', 'loss_mean', 'attacks_per_turn', 'placement_entropy_mean',
            'ratio_mean', 'advantage_mean', 'won_battles_per_turn'
        ]
        
        for metric in important_metrics:
            if metric in self.metrics:
                latest = self.metrics[metric]['latest']
                trend = self.metrics[metric]['trend']
                trend_symbol = {
                    'improving': '📈', 'slightly_improving': '📈',
                    'stable': '➡️', 'declining': '📉', 'slightly_declining': '📉',
                    'insufficient_data': '❓'
                }.get(trend, '❓')
                
                report += f"├─ {metric.replace('_', ' ').title()}: {latest:.4f} {trend_symbol}\n"
        
        report += f"""
🎯 RECOMMENDATIONS ({len(recommendations)} items)
"""
        for i, rec in enumerate(recommendations, 1):
            report += f"├─ {i}. {rec}\n"
        
        report += f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ 🎓 INTERPRETATION GUIDE                                                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ 📈 Improving trends indicate successful learning                             ║
║ ➡️  Stable metrics suggest convergence (good for loss, check for performance) ║
║ 📉 Declining trends may indicate overfitting or poor hyperparameters        ║
║ 🎲 High entropy = exploration, Low entropy = exploitation                   ║
║ ⚔️  Combat metrics show strategic learning effectiveness                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

💡 NEXT STEPS:
1. Review recommendations above and implement high-priority changes
2. Monitor training for 100-500 more episodes after changes
3. Consider A/B testing different configurations
4. Save model checkpoints before major hyperparameter changes

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Analysis based on {len(self.metrics)} metrics from TensorBoard logs.
"""
        
        return report


def main():
    """Main analysis function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze TensorBoard logs for pyWarlight RL training')
    parser.add_argument('log_dir', nargs='?', 
                       default='analysis/logs/residual_model_ultra_decisive',
                       help='Path to TensorBoard log directory')
    parser.add_argument('--no-plot', action='store_true', 
                       help='Skip generating plots')
    parser.add_argument('--output', '-o', 
                       help='Save report to file (default: print to console)')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = TensorBoardAnalyzer(args.log_dir)
    
    # Load data
    if not analyzer.load_data():
        print("❌ Failed to load TensorBoard data")
        return 1
    
    # Generate visualizations
    if not args.no_plot:
        print("📊 Creating visualizations...")
        analyzer.create_visualizations()
    
    # Generate report
    print("📝 Generating analysis report...")
    report = analyzer.generate_report()
    
    # Output report
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"📄 Report saved to: {args.output}")
    else:
        print(report)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
