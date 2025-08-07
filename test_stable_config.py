#!/usr/bin/env python3
"""
Test script to verify the gradient stability configuration fixes value function issues
"""
import sys
import os
sys.path.append('.')

from src.config.training_config import ConfigFactory

def test_stable_config():
    """Test the gradient stability configuration"""
    print("🧪 Testing gradient_stability_fix configuration...")
    
    config = ConfigFactory.create('stable_learning')
    print(f"\n📋 Configuration Summary:")
    print(config.summary())
    
    print(f"\n🔧 Key Gradient Stability Settings:")
    print(f"   Learning Rate: {config.ppo.learning_rate} (much more conservative)")
    print(f"   Clip Epsilon: {config.ppo.clip_eps} (very tight clipping)")  
    print(f"   Gradient Clip: {config.ppo.gradient_clip_norm} (aggressive clipping)")
    print(f"   Value Loss Coeff: {config.ppo.value_loss_coeff} (much lower weight)")
    print(f"   PPO Epochs: {config.ppo.ppo_epochs} (single epoch)")
    print(f"   Normalize Advantages: {config.ppo.normalize_advantages}")
    
    print(f"\n🎯 Expected Improvements Based on Your Symptoms:")
    print(f"   📊 Gradient Issues (Current: 0-5000 oscillating):")
    print(f"      → Should stabilize between 0.1-1.0 with aggressive clipping")
    print(f"   � Value Function (Current: crit_loss ~1000):")
    print(f"      → Should drop to <100 with lower learning rate + value loss weight")
    print(f"   📊 Policy Ratio (Current: stuck at 0.9946):")
    print(f"      → Should show more variation with tighter clipping")
    print(f"   📊 Value Estimates (Current: -40 vs returns -25):")
    print(f"      → Gap should close as value function stabilizes")
    
    print(f"\n⚠️  What to Watch For:")
    print(f"   1. Gradient norms should stay < 1.0 (not oscillate to 5000)")
    print(f"   2. Gradient CV should drop < 1.0 (you had 1.4-2.1)")
    print(f"   3. Value predictions should get closer to returns")
    print(f"   4. Critic loss should drop significantly")
    print(f"   5. Learning should be slower but more stable")
    
    print(f"\n🚨 Red Flags - Stop Training If You See:")
    print(f"   - Gradient norms still exploding > 10")
    print(f"   - Value function still diverging")
    print(f"   - No improvement after 50+ games")

if __name__ == "__main__":
    test_stable_config()
