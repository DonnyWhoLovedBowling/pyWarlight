#!/usr/bin/env python3
"""
Value Computation Monitor for Warlight Training

This script helps monitor value computation health during training.
Run this alongside your training to check for value-related issues.
"""

import torch
import numpy as np
from pathlib import Path

def analyze_value_logs(log_file_path):
    """Analyze training logs for value computation patterns."""
    
    if not Path(log_file_path).exists():
        print(f"❌ Log file not found: {log_file_path}")
        return
    
    print(f"📊 Analyzing value patterns in: {log_file_path}")
    print("=" * 60)
    
    values = []
    entropies = []
    losses = []
    
    try:
        with open(log_file_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            # Extract value-related metrics
            if "Value loss:" in line:
                try:
                    val = float(line.split("Value loss:")[1].split()[0])
                    losses.append(val)
                except:
                    pass
                    
            if "entropy:" in line:
                try:
                    val = float(line.split("entropy:")[1].split()[0])
                    entropies.append(val)
                except:
                    pass
                    
            if "Values computed" in line:
                try:
                    # Extract value statistics if logged
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if "range:" in part and i+1 < len(parts):
                            val = float(parts[i+1].strip('[](),'))
                            values.append(val)
                except:
                    pass
    
    except Exception as e:
        print(f"❌ Error reading log file: {e}")
        return
    
    # Analyze patterns
    print(f"📈 ANALYSIS RESULTS:")
    print("-" * 40)
    
    if losses:
        print(f"Value Losses ({len(losses)} samples):")
        print(f"  Mean: {np.mean(losses):.6f}")
        print(f"  Std:  {np.std(losses):.6f}")
        print(f"  Range: [{np.min(losses):.6f}, {np.max(losses):.6f}]")
        
        # Check for concerning patterns
        if np.mean(losses) > 1.0:
            print("  ⚠️  High value losses - possible value computation issues")
        elif np.std(losses) > 0.5:
            print("  ⚠️  High variance in value losses - unstable training")
        else:
            print("  ✅ Value losses look healthy")
        print()
    
    if entropies:
        print(f"Entropies ({len(entropies)} samples):")
        print(f"  Mean: {np.mean(entropies):.4f}")
        print(f"  Std:  {np.std(entropies):.4f}")
        print(f"  Range: [{np.min(entropies):.4f}, {np.max(entropies):.4f}]")
        
        # Check entropy trends (simple linear regression)
        if len(entropies) > 10:
            x = np.arange(len(entropies))
            slope = np.polyfit(x, entropies, 1)[0]
            print(f"  Trend: {slope:.6f} per step")
            
            if slope < -0.001:
                print("  ✅ Entropy decreasing - learning progress")
            elif slope > 0.001:
                print("  ⚠️  Entropy increasing - possible exploration issues")
            else:
                print("  ⚠️  Entropy flat - may need adjustment")
        print()
    
    if values:
        print(f"Values ({len(values)} samples):")
        print(f"  Mean: {np.mean(values):.6f}")
        print(f"  Std:  {np.std(values):.6f}")
        print(f"  Range: [{np.min(values):.6f}, {np.max(values):.6f}]")
        
        # Check for extreme values
        if np.max(np.abs(values)) > 10:
            print("  ⚠️  Very large value magnitudes - possible scaling issues")
        elif np.std(values) < 0.001:
            print("  ⚠️  Very low value variance - possible value collapse")
        else:
            print("  ✅ Value magnitudes look reasonable")
        print()

def check_recent_logs():
    """Check the most recent log files for value computation issues."""
    
    log_dir = Path("analysis/logs")
    if not log_dir.exists():
        print("❌ Log directory not found. Run training first.")
        return
    
    # Find most recent log directories
    subdirs = [d for d in log_dir.iterdir() if d.is_dir()]
    if not subdirs:
        print("❌ No log subdirectories found.")
        return
    
    # Sort by modification time
    subdirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    print("🔍 CHECKING RECENT TRAINING LOGS")
    print("=" * 60)
    
    for i, log_subdir in enumerate(subdirs[:3]):  # Check 3 most recent
        print(f"\n{i+1}. {log_subdir.name}")
        print("-" * len(log_subdir.name))
        
        # Look for log files in this directory
        log_files = list(log_subdir.glob("*.log")) + list(log_subdir.glob("*.txt"))
        
        if log_files:
            # Use the most recent log file
            latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
            analyze_value_logs(latest_log)
        else:
            print("  No log files found in this directory")

def value_computation_healthcheck():
    """Quick healthcheck for value computation setup."""
    
    print("🏥 VALUE COMPUTATION HEALTH CHECK")
    print("=" * 60)
    
    # Check if verification is enabled in config
    try:
        from src.config.training_config import ConfigFactory
        
        # Check residual model config (likely what user is using)
        config = ConfigFactory.create('residual_model')
        verification = config.verification
        
        print("Configuration Status:")
        print(f"  verify_value_computation: {verification.verify_value_computation}")
        print(f"  verify_gae_computation: {verification.verify_gae_computation}")
        print(f"  detailed_logging: {verification.detailed_logging}")
        
        if verification.verify_value_computation and verification.verify_gae_computation:
            print("  ✅ Value verification enabled")
        else:
            print("  ⚠️  Value verification disabled - consider enabling for debugging")
            
    except Exception as e:
        print(f"  ❌ Error checking config: {e}")
    
    print("\nVerification Methods Available:")
    try:
        from src.agents.RLUtils.PPOVerification import PPOVerifier
        verifier = PPOVerifier(None)  # Create without config for method check
        
        methods = [method for method in dir(verifier) if method.startswith('verify_')]
        for method in methods:
            print(f"  ✅ {method}")
            
    except Exception as e:
        print(f"  ❌ Error loading verification methods: {e}")

if __name__ == "__main__":
    print("VALUE COMPUTATION MONITOR")
    print("=" * 60)
    
    # Run health check
    value_computation_healthcheck()
    
    print("\n")
    
    # Check recent logs
    check_recent_logs()
    
    print("\n📋 RECOMMENDATIONS:")
    print("-" * 20)
    print("1. Enable value verification in your training config")
    print("2. Monitor entropy trends - should decrease over time")
    print("3. Watch for extreme value magnitudes (>10 or <0.001 variance)")
    print("4. Check value loss stability (mean <1.0, std <0.5)")
    print("5. Run this monitor periodically during long training runs")
