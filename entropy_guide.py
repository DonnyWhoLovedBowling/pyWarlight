#!/usr/bin/env python3
"""
Entropy Optimization Testing Script

This script helps you test different entropy configurations for the residual model
and provides real-time monitoring of entropy reduction.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.training_config import ConfigFactory

def show_entropy_comparison():
    """Show entropy settings for different configurations"""
    print("🎯 ENTROPY CONFIGURATION COMPARISON")
    print("=" * 60)
    
    configs = {
        'residual_model': 'Residual Model (Decisive)',
        'residual_low_entropy': 'Residual Model (Ultra Decisive)',
        'value_regularized_multi_epoch': 'Original Multi-Epoch'
    }
    
    for config_name, description in configs.items():
        try:
            config = ConfigFactory.create(config_name)
            
            print(f"\n📋 {description}")
            print(f"   Config: {config_name}")
            print(f"   🎲 Initial entropy coeff: {config.ppo.entropy_coeff_start}")
            print(f"   📉 Entropy decay: {config.ppo.entropy_coeff_decay}")
            print(f"   ⏱️  Decay episodes: {config.ppo.entropy_decay_episodes}")
            print(f"   🏠 Placement entropy: {config.ppo.placement_entropy_coeff}")
            print(f"   🔗 Edge entropy: {config.ppo.edge_entropy_coeff}")
            print(f"   ⚔️  Army entropy: {config.ppo.army_entropy_coeff}")
            
            # Calculate entropy factor at different episodes
            episodes_to_test = [0, 100, 500, 1000, 2000, 5000]
            print(f"   📊 Entropy factor over time:")
            for ep in episodes_to_test:
                factor = max(config.ppo.entropy_coeff_start - (ep / config.ppo.entropy_decay_episodes) * config.ppo.entropy_coeff_decay, 0.01)
                print(f"      Episode {ep:4d}: {factor:.3f}")
                
        except Exception as e:
            print(f"   ❌ Error loading {config_name}: {e}")

def show_recommendations():
    """Show specific recommendations for entropy tuning"""
    print("\n💡 ENTROPY TUNING RECOMMENDATIONS")
    print("=" * 60)
    
    print("""
🔍 DIAGNOSIS: High Entropy = Random Actions
   Your residual model has stable gradients but high entropy means:
   • Actions are too random/uniform
   • Not learning decisive strategies
   • Need stronger entropy reduction to encourage focused actions

🎯 IMMEDIATE SOLUTIONS:

   1️⃣ MODERATE SOLUTION (Try First):
      Change line 55 in RLGNNAgent.py:
      config = ConfigFactory.create('residual_model')  # Current
      # This uses aggressive but reasonable entropy reduction

   2️⃣ AGGRESSIVE SOLUTION (If still too random):
      config = ConfigFactory.create('residual_low_entropy')
      # This uses very aggressive entropy reduction

🔧 WHAT THESE CHANGES DO:

   RESIDUAL_MODEL (Moderate):
   • Starts with entropy factor 1.0 (vs 0.5 before)
   • Decays 90% over 5000 episodes (vs 60% over 15000)
   • Higher entropy weights for placement and edge selection
   • Should see entropy drop within 500-1000 episodes

   RESIDUAL_LOW_ENTROPY (Aggressive):
   • Starts with entropy factor 2.0 (very high initial exploration)
   • Decays 90% over just 2000 episodes (very fast)
   • Much higher entropy weights
   • Should see rapid entropy drop within 200-500 episodes

📊 MONITORING EXPECTATIONS:

   Episode 100:  Army entropy should drop from ~1970 to ~1500
   Episode 500:  Army entropy should be ~800-1200
   Episode 1000: Army entropy should be ~400-800
   Episode 2000: Army entropy should be ~200-400 (decisive actions)

⚠️  IMPORTANT NOTES:
   • These configs maintain gradient stability of residual model
   • Higher entropy coefficients = stronger pressure toward decisive actions
   • Monitor both entropy AND performance (win rate, regions captured)
   • If entropy drops too fast, actions might become deterministic too early

🎮 GAME BEHAVIOR CHANGES EXPECTED:
   • More consistent army placement strategies
   • Less random attack selection
   • More focused offensive/defensive patterns
   • Better continent control attempts
    """)

def show_quick_commands():
    """Show quick commands for testing"""
    print("\n⚡ QUICK TEST COMMANDS")
    print("=" * 60)
    
    print("""
🔧 TO APPLY MODERATE ENTROPY REDUCTION:
   Edit line 55 in src/agents/RLGNNAgent.py:
   config = ConfigFactory.create('residual_model')

🔧 TO APPLY AGGRESSIVE ENTROPY REDUCTION:
   Edit line 55 in src/agents/RLGNNAgent.py:  
   config = ConfigFactory.create('residual_low_entropy')

📊 TO MONITOR ENTROPY IN TENSORBOARD:
   1. Run training for 500+ episodes
   2. Open TensorBoard: tensorboard --logdir analysis/logs
   3. Watch these metrics:
      • army_entropy_mean (should decrease over time)
      • placement_entropy_mean (should decrease over time)
      • edge_entropy_mean (should decrease over time)

🎯 SUCCESS INDICATORS:
   ✅ Army entropy decreasing from ~1970 to <500 within 1000 episodes
   ✅ More consistent placement patterns
   ✅ Gradient norms staying stable (10-100 range)
   ✅ Win rate improving or stable while entropy decreases

❌ FAILURE INDICATORS:
   ❌ Entropy stays high (>1500) after 1000 episodes
   ❌ Gradient norms exploding (>1000)
   ❌ Performance degrading significantly
   ❌ Actions becoming deterministic too early (entropy <50)
    """)

def main():
    """Main function"""
    print("🎲 ENTROPY OPTIMIZATION GUIDE FOR RESIDUAL MODEL")
    print("=" * 70)
    
    show_entropy_comparison()
    show_recommendations() 
    show_quick_commands()
    
    print("\n✅ Ready to optimize entropy! Start with 'residual_model' config.")

if __name__ == "__main__":
    main()
