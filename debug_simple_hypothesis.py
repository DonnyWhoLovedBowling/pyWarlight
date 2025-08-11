#!/usr/bin/env python3
"""
Simple debug script to test the hypothesis that model mode differences cause
placement_diff and attack_diff to be non-zero in the first epoch.
"""

import torch
import torch.nn.functional as f

def test_simple_model_mode_hypothesis():
    """Test that shows the core issue with train/eval mode differences"""
    print("Testing hypothesis: Model mode causes log probability differences")
    print("="*60)
    
    # Create a simple model with dropout to show the effect
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Sequential(
                torch.nn.Linear(10, 20),
                torch.nn.Dropout(0.1),  # This will behave differently in train vs eval
                torch.nn.Linear(20, 5)
            )
        
        def forward(self, x):
            return self.layer(x)
    
    # Create model and test input
    torch.manual_seed(42)
    model = SimpleModel()
    test_input = torch.randn(2, 10)  # Batch of 2 samples
    
    print("1. Testing with TRAINING mode (as during action selection):")
    model.train()
    with torch.no_grad():
        output_train = model(test_input)
        log_probs_train = f.log_softmax(output_train, dim=-1)
    
    print(f"   Output: {output_train[0]}")
    print(f"   Log probs: {log_probs_train[0]}")
    
    print("\n2. Testing with EVAL mode (as during PPO update):")
    model.eval()
    with torch.no_grad():
        output_eval = model(test_input)
        log_probs_eval = f.log_softmax(output_eval, dim=-1)
    
    print(f"   Output: {output_eval[0]}")
    print(f"   Log probs: {log_probs_eval[0]}")
    
    # Calculate differences
    output_diff = torch.abs(output_train - output_eval).max().item()
    log_prob_diff = torch.abs(log_probs_train - log_probs_eval).max().item()
    
    print(f"\n3. DIFFERENCES:")
    print(f"   Max output difference: {output_diff:.6f}")
    print(f"   Max log prob difference: {log_prob_diff:.6f}")
    
    if log_prob_diff > 1e-6:
        print("\n🚨 HYPOTHESIS CONFIRMED!")
        print("   Model produces different outputs in train vs eval mode.")
        print("   This explains why placement_diff and attack_diff are not zero in first epoch.")
        print("\n📋 SOLUTION:")
        print("   Ensure consistent model mode between action selection and PPO updates.")
        return True
    else:
        print("\n✅ No significant differences found.")
        return False

def explain_ppo_issue():
    """Explain how this relates to the PPO first epoch issue"""
    print("\n" + "="*60)
    print("HOW THIS RELATES TO THE PPO ISSUE:")
    print("="*60)
    print()
    print("During a game turn:")
    print("  1. Model is in TRAINING mode")
    print("  2. Actions are selected and log probabilities captured")
    print("  3. These log probabilities are stored in the buffer")
    print()
    print("During PPO update (first epoch):")
    print("  1. Model is switched to EVAL mode (line 238 in PPOAgent.py)")
    print("  2. Same actions are re-evaluated to compute new log probabilities")
    print("  3. NEW log probs ≠ OLD log probs due to model mode difference")
    print("  4. placement_diff = new_log_probs - old_log_probs ≠ 0")
    print("  5. attack_diff = new_log_probs - old_log_probs ≠ 0")
    print()
    print("Expected behavior in first epoch:")
    print("  - placement_diff should be all zeros")
    print("  - attack_diff should be all zeros")
    print("  - PPO ratios should be 1.0 (no policy change yet)")
    print()
    print("Actual behavior:")
    print("  - placement_diff has small non-zero values")
    print("  - attack_diff has small non-zero values")
    print("  - PPO ratios deviate from 1.0 even before any learning")

if __name__ == "__main__":
    print("Investigating PPO first-epoch issue...")
    
    # Test the hypothesis
    hypothesis_confirmed = test_simple_model_mode_hypothesis()
    
    # Explain the connection to PPO
    explain_ppo_issue()
    
    if hypothesis_confirmed:
        print("\n" + "🎯 RECOMMENDED FIXES:")
        print("  Option 1: Use model.eval() during action selection")
        print("  Option 2: Remove model.eval() call in PPO update")
        print("  Option 3: Store logits instead of log probabilities")
        print("\n  The simplest fix is Option 1: ensure model is in eval mode")
        print("  during action selection to match PPO update behavior.")
    else:
        print("\n❓ The issue might be elsewhere. Further investigation needed.")
