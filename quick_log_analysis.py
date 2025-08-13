#!/usr/bin/env python3
"""
Quick analysis of the 72-episode run
"""

import os
import sys

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
import numpy as np

def extract_key_metrics(log_file_path):
    """Extract key metrics from TensorBoard logs"""
    
    print(f"📊 Analyzing: {log_file_path}")
    
    data = {}
    
    for summary in tf.compat.v1.train.summary_iterator(log_file_path):
        for value in summary.summary.value:
            tag = value.tag
            step = summary.step
            
            if tag not in data:
                data[tag] = []
            
            data[tag].append((step, value.simple_value))
    
    # Show key metrics
    key_metrics = ['win', 'reward', 'gradient_norm', 'zero_gradient_params', 'loss_mean']
    
    print("\n🔍 KEY METRICS SUMMARY:")
    print("=" * 50)
    
    for metric in key_metrics:
        if metric in data:
            values = [v[1] for v in data[metric]]
            steps = [v[0] for v in data[metric]]
            
            print(f"\n📈 {metric.upper()}:")
            print(f"   Episodes: {len(values)}")
            print(f"   First 10 values: {values[:10]}")
            print(f"   Last 10 values: {values[-10:]}")
            print(f"   Mean: {np.mean(values):.4f}")
            print(f"   Min: {np.min(values):.4f}")
            print(f"   Max: {np.max(values):.4f}")
        else:
            print(f"\n❌ {metric} not found")
    
    # Check if training stopped early
    max_step = max([max([s[0] for s in data[tag]]) for tag in data.keys()])
    print(f"\n📊 Training ran for {max_step} episodes")
    
    # Win rate analysis
    if 'win' in data:
        wins = [v[1] for v in data['win']]
        total_wins = sum(wins)
        win_rate = total_wins / len(wins) if len(wins) > 0 else 0
        print(f"🏆 Win rate: {win_rate:.2%} ({total_wins}/{len(wins)})")
        
        # Check for improvement trend
        if len(wins) >= 20:
            early_wins = sum(wins[:len(wins)//2])
            late_wins = sum(wins[len(wins)//2:])
            early_rate = early_wins / (len(wins)//2)
            late_rate = late_wins / (len(wins) - len(wins)//2)
            print(f"   Early episodes: {early_rate:.2%}")
            print(f"   Late episodes: {late_rate:.2%}")
            
            if late_rate > early_rate:
                print("   ✅ Learning trend: IMPROVING")
            elif late_rate < early_rate:
                print("   ⚠️  Learning trend: DECLINING")
            else:
                print("   ➡️  Learning trend: STABLE")
    
    # Gradient health check
    if 'gradient_norm' in data:
        grad_norms = [v[1] for v in data['gradient_norm']]
        avg_grad = np.mean(grad_norms)
        print(f"\n⚡ Gradient Health:")
        print(f"   Average gradient norm: {avg_grad:.4f}")
        if avg_grad < 1.0:
            print("   ❌ TOO LOW - gradient clipping too aggressive")
        elif avg_grad < 5.0:
            print("   ⚠️  MODERATE - may need higher gradient clip")
        elif avg_grad > 20.0:
            print("   ⚠️  HIGH - may need lower gradient clip")
        else:
            print("   ✅ HEALTHY - gradients in good range")
    
    if 'zero_gradient_params' in data:
        zero_grads = [v[1] for v in data['zero_gradient_params']]
        avg_zeros = np.mean(zero_grads)
        print(f"   Zero gradient params: {avg_zeros:.1f}")
        if avg_zeros > 10:
            print("   ❌ TOO MANY zero gradients")
        else:
            print("   ✅ Zero gradients under control")

def main():
    log_file = "analysis/logs/sage_model_stable/events.out.tfevents.1755036616.Deskie.17888.1"
    
    print("🚀 Quick Analysis of SAGE Model Run")
    print("=" * 70)
    
    extract_key_metrics(log_file)
    
    print("\n" + "=" * 70)
    print("🎯 INTERPRETATION:")
    print("If gradients are healthy and win rate is improving,")
    print("the run may have stopped due to external factors.")
    print("If gradients are still low, need higher gradient_clip_norm.")

if __name__ == "__main__":
    main()
