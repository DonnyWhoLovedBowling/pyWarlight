#!/usr/bin/env python3
"""
Analyze the failed residual_percentage_fixed_gradients run to understand why it didn't work
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
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

def analyze_tensorboard_logs(log_file_path):
    """Extract and analyze data from TensorBoard logs"""
    
    print(f"📊 Analyzing TensorBoard logs: {log_file_path}")
    
    if not os.path.exists(log_file_path):
        print(f"❌ Log file not found: {log_file_path}")
        return None
    
    # Read the TensorBoard file
    data = {}
    
    try:
        for summary in tf.compat.v1.train.summary_iterator(log_file_path):
            for value in summary.summary.value:
                tag = value.tag
                step = summary.step
                
                if tag not in data:
                    data[tag] = {'steps': [], 'values': []}
                
                data[tag]['steps'].append(step)
                data[tag]['values'].append(value.simple_value)
    except Exception as e:
        print(f"❌ Error reading TensorBoard file: {e}")
        return None
    
    # Convert to DataFrame for easier analysis
    dataframes = {}
    for tag, values in data.items():
        if len(values['steps']) > 0:
            dataframes[tag] = pd.DataFrame({
                'step': values['steps'],
                'value': values['values']
            }).sort_values('step')
    
    return dataframes

def analyze_gradient_recovery(data):
    """Analyze if gradients recovered as expected"""
    print("\n🔍 === GRADIENT ANALYSIS ===")
    
    gradient_metrics = [tag for tag in data.keys() if 'gradient' in tag.lower() or 'grad' in tag.lower()]
    
    if not gradient_metrics:
        print("❌ No gradient metrics found!")
        return False
    
    print(f"Found gradient metrics: {gradient_metrics}")
    
    for metric in gradient_metrics:
        df = data[metric]
        if len(df) > 100:  # Need decent amount of data
            recent_mean = df.tail(10)['value'].mean()
            early_mean = df.head(10)['value'].mean()
            overall_mean = df['value'].mean()
            max_val = df['value'].max()
            min_val = df['value'].min()
            
            print(f"\n📈 {metric}:")
            print(f"   Early episodes mean: {early_mean:.4f}")
            print(f"   Recent episodes mean: {recent_mean:.4f}")
            print(f"   Overall mean: {overall_mean:.4f}")
            print(f"   Range: {min_val:.4f} - {max_val:.4f}")
            
            # Check if gradients recovered
            if 'grad_norm' in metric.lower():
                if overall_mean < 5.0:
                    print(f"   ⚠️  PROBLEM: Gradient norms still too low (expected ~15)")
                    return False
                elif overall_mean > 10.0:
                    print(f"   ✅ Gradient norms recovered well")
                    return True
                else:
                    print(f"   🔍 Gradient norms moderate - may need investigation")
    
    return None

def analyze_gradient_cv(data):
    """Analyze gradient coefficient of variation (cv) metrics"""
    print("\n📊 === GRADIENT CV ANALYSIS ===")
    cv_metrics = [tag for tag in data.keys() if 'grad_cv' in tag.lower() or 'gradient_cv' in tag.lower()]
    if not cv_metrics:
        print("❌ No gradient CV metrics found!")
        return
    print(f"Found gradient CV metrics: {cv_metrics}")
    for metric in cv_metrics:
        df = data[metric]
        if len(df) > 100:
            recent_mean = df.tail(10)['value'].mean()
            early_mean = df.head(10)['value'].mean()
            overall_mean = df['value'].mean()
            print(f"\n📈 {metric}:")
            print(f"   Early mean: {early_mean:.4f}")
            print(f"   Recent mean: {recent_mean:.4f}")
            print(f"   Overall mean: {overall_mean:.4f}")
            if overall_mean > 2.0:
                print(f"   ⚠️  HIGH CV: Possible instability or poor gradient flow")
            else:
                print(f"   ✅ CV trend looks reasonable")

def analyze_weight_change(data):
    """Analyze weight change metrics"""
    print("\n🔄 === WEIGHT CHANGE ANALYSIS ===")
    weight_metrics = [tag for tag in data.keys() if 'weight_change' in tag.lower() or 'weights_delta' in tag.lower()]
    if not weight_metrics:
        print("❌ No weight change metrics found!")
        return
    print(f"Found weight change metrics: {weight_metrics}")
    for metric in weight_metrics:
        df = data[metric]
        if len(df) > 100:
            recent_mean = df.tail(10)['value'].mean()
            early_mean = df.head(10)['value'].mean()
            overall_mean = df['value'].mean()
            print(f"\n📈 {metric}:")
            print(f"   Early mean: {early_mean:.6f}")
            print(f"   Recent mean: {recent_mean:.6f}")
            print(f"   Overall mean: {F:.6f}")
            if overall_mean < 1e-6:
                print(f"   ⚠️  Very low weight change: Possible stagnation")
            else:
                print(f"   ✅ Weight change trend looks reasonable")

def analyze_win_rate_pattern(data):
    """Analyze win rate and performance patterns"""
    print("\n🎮 === PERFORMANCE ANALYSIS ===")
    
    performance_metrics = [
        'win_rate', 'episode_reward', 'total_reward', 
        'game_result', 'average_reward'
    ]
    
    found_metrics = []
    for metric in performance_metrics:
        matching = [tag for tag in data.keys() if metric in tag.lower()]
        found_metrics.extend(matching)
    
    if not found_metrics:
        print("❌ No performance metrics found!")
        return
    
    print(f"Found performance metrics: {found_metrics}")
    
    for metric in found_metrics:
        df = data[metric]
        if len(df) > 100:
            recent_mean = df.tail(100)['value'].mean()
            early_mean = df.head(100)['value'].mean()
            overall_mean = df['value'].mean()
            
            print(f"\n📊 {metric}:")
            print(f"   Early episodes: {early_mean:.4f}")
            print(f"   Recent episodes: {recent_mean:.4f}")
            print(f"   Overall: {overall_mean:.4f}")
            
            # Check for concerning patterns
            if 'win' in metric.lower() and overall_mean < 0.01:
                print(f"   ❌ CRITICAL: Near-zero win rate!")
            elif 'reward' in metric.lower() and recent_mean < early_mean - 10:
                print(f"   ⚠️  Performance declining over time")

def analyze_training_stability(data):
    """Analyze training stability and loss patterns"""
    print("\n🏋️ === TRAINING STABILITY ANALYSIS ===")
    
    training_metrics = [
        'actor_loss', 'critic_loss', 'value_loss', 
        'policy_loss', 'entropy', 'value_function'
    ]
    
    found_metrics = []
    for metric in training_metrics:
        matching = [tag for tag in data.keys() if metric in tag.lower()]
        found_metrics.extend(matching)
    
    if not found_metrics:
        print("❌ No training metrics found!")
        return
    
    print(f"Found training metrics: {found_metrics}")
    
    for metric in found_metrics:
        df = data[metric]
        if len(df) > 100:
            recent_values = df.tail(100)['value']
            early_values = df.head(100)['value']
            
            recent_mean = recent_values.mean()
            recent_std = recent_values.std()
            early_mean = early_values.mean()
            
            print(f"\n📈 {metric}:")
            print(f"   Early mean: {early_mean:.4f}")
            print(f"   Recent mean: {recent_mean:.4f}")
            print(f"   Recent std: {recent_std:.4f}")
            
            # Check for problematic patterns
            if recent_std > abs(recent_mean) * 2:
                print(f"   ⚠️  HIGH VARIANCE: Training unstable")
            
            if 'loss' in metric.lower():
                if recent_mean > early_mean * 2:
                    print(f"   ❌ PROBLEM: Loss increasing over time")
                elif recent_mean < early_mean * 0.1:
                    print(f"   ⚠️  Loss very low - possible underfitting")
                else:
                    print(f"   ✅ Loss trend looks reasonable")

def analyze_zero_gradient_params(data):
    """Check zero gradient parameters specifically"""
    print("\n🔍 === ZERO GRADIENT PARAMETERS ANALYSIS ===")
    
    zero_grad_metrics = [tag for tag in data.keys() if 'zero' in tag.lower() and 'grad' in tag.lower()]
    
    if not zero_grad_metrics:
        print("❌ No zero gradient metrics found!")
        return
    
    for metric in zero_grad_metrics:
        df = data[metric]
        if len(df) > 100:
            recent_mean = df.tail(100)['value'].mean()
            early_mean = df.head(100)['value'].mean()
            max_val = df['value'].max()
            
            print(f"\n🎯 {metric}:")
            print(f"   Early episodes: {early_mean:.1f}")
            print(f"   Recent episodes: {recent_mean:.1f}")
            print(f"   Maximum: {max_val:.1f}")
            
            if recent_mean > 10:
                print(f"   ❌ CRITICAL: Too many zero gradient params (>{recent_mean:.1f})")
                print(f"   This indicates gradient flow is still broken!")
            elif recent_mean > 5:
                print(f"   ⚠️  Moderate zero gradient params - some concern")
            else:
                print(f"   ✅ Zero gradient params look good")

def diagnose_failure_mode(data):
    """Try to diagnose the specific failure mode"""
    print("\n🩺 === FAILURE MODE DIAGNOSIS ===")
    
    # Check if this is a gradient clipping issue
    grad_norms = [tag for tag in data.keys() if 'grad_norm' in tag.lower()]
    if grad_norms:
        df = data[grad_norms[0]]
        avg_grad_norm = df['value'].mean()
        
        if avg_grad_norm < 2.0:
            print("🔍 HYPOTHESIS 1: Gradient clipping still too aggressive")
            print(f"   Average gradient norm: {avg_grad_norm:.3f} (expected ~15)")
            print("   💡 SOLUTION: Try gradient_clip_norm = 25.0 or higher")
            return "gradient_clipping"
    
    # Check if this is a learning rate issue
    performance_metrics = [tag for tag in data.keys() if 'reward' in tag.lower() or 'win' in tag.lower()]
    if performance_metrics:
        for metric in performance_metrics:
            df = data[metric]
            if len(df) > 100:
                early = df.head(10)['value'].mean()
                late = df.tail(10)['value'].mean()
                
                if abs(late - early) < 0.01:  # No improvement
                    print("🔍 HYPOTHESIS 2: Learning rate too low or model capacity issue")
                    print(f"   No improvement over training ({early:.3f} → {late:.3f})")
                    print("   💡 SOLUTION: Try higher learning rate or check model architecture")
                    return "learning_rate"
    
    # Check for entropy collapse
    entropy_metrics = [tag for tag in data.keys() if 'entropy' in tag.lower()]
    if entropy_metrics:
        for metric in entropy_metrics:
            df = data[metric]
            recent_entropy = df.tail(10)['value'].mean()
            
            if recent_entropy < 0.1:
                print("🔍 HYPOTHESIS 3: Entropy collapse - policy too deterministic")
                print(f"   {metric}: {recent_entropy:.4f}")
                print("   💡 SOLUTION: Increase entropy coefficients or reduce entropy decay")
                return "entropy_collapse"
    
    print("🤔 Multiple factors may be involved - check all metrics above")
    return "unknown"

def main():
    log_file_path = r"C:\Users\pcvan\Projects\pyWarlight\analysis\logs\transformer_decisive_experiment\events.out.tfevents.1755094237.Deskie.12024.1"
    data = analyze_tensorboard_logs(log_file_path)
    if data is None:
        print("❌ Could not analyze log file.")
        return
    analyze_gradient_recovery(data)
    analyze_gradient_cv(data)
    analyze_weight_change(data)
    analyze_win_rate_pattern(data)
    analyze_training_stability(data)
    analyze_zero_gradient_params(data)

if __name__ == "__main__":
    main()
