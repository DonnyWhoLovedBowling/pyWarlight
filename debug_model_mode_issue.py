#!/usr/bin/env python3
"""
Debug script to verify that model mode (train vs evdef test_log_prob_differences():
    """Test how model mode differences affect log probabilities"""
    print("\n" + "="*50)
    print("Testing log probability differences...")
    
    torch.manual_seed(42)
    model = WarlightPolicyNet(
        node_feat_dim=10,
        embed_dim=64,
        n_army_options=10
    )probability differences.
This will help confirm the hypothesis about why placement_diff and attack_diff are not zero
in the first epoch.
"""

import torch
import torch.nn.functional as f
import numpy as np
from src.agents.RLUtils.WarlightModel import WarlightPolicyNet

def test_model_mode_consistency():
    """Test if switching between train/eval mode affects log probabilities"""
    print("Testing model mode consistency...")
    
    # Create a simple test model
    torch.manual_seed(42)  # For reproducibility
    
    # Create model with correct constructor parameters
    model = WarlightPolicyNet(
        node_feat_dim=10,
        embed_dim=64,
        n_army_options=10
    )
    
    # Create test input
    batch_size = 2
    num_nodes = 42
    num_features = 10
    
    node_features = torch.randn(batch_size, num_nodes, num_features)
    action_edges = torch.randint(0, num_nodes, (batch_size, 42, 2))
    mask = torch.ones(batch_size, num_nodes)
    
    print("=== Testing with same inputs but different model modes ===")
    
    # Test 1: Model in training mode (as during action selection)
    model.train()
    print("\\n1. Model in TRAINING mode:")
    with torch.no_grad():  # Disable gradients to match action selection
        placement_logits_train, attack_logits_train, army_logits_train = model(
            node_features, action_edges, mask
        )
    
    print(f"Placement logits sample: {placement_logits_train[0, :5]}")
    print(f"Attack logits sample: {attack_logits_train[0, :5]}")
    print(f"Army logits sample: {army_logits_train[0, 0, :5]}")
    
    # Test 2: Model in eval mode (as during PPO update)
    model.eval()
    print("\\n2. Model in EVAL mode:")
    with torch.no_grad():
        placement_logits_eval, attack_logits_eval, army_logits_eval = model(
            node_features, action_edges, mask
        )
    
    print(f"Placement logits sample: {placement_logits_eval[0, :5]}")
    print(f"Attack logits sample: {attack_logits_eval[0, :5]}")
    print(f"Army logits sample: {army_logits_eval[0, 0, :5]}")
    
    # Compare differences
    placement_diff = torch.abs(placement_logits_train - placement_logits_eval).max()
    attack_diff = torch.abs(attack_logits_train - attack_logits_eval).max()
    army_diff = torch.abs(army_logits_train - army_logits_eval).max()
    
    print(f"\\n=== DIFFERENCES ===")
    print(f"Max placement logits difference: {placement_diff:.6f}")
    print(f"Max attack logits difference: {attack_diff:.6f}")
    print(f"Max army logits difference: {army_diff:.6f}")
    
    if placement_diff > 1e-6 or attack_diff > 1e-6 or army_diff > 1e-6:
        print("\\n🚨 ISSUE CONFIRMED: Model produces different outputs in train vs eval mode!")
        print("This explains why placement_diff and attack_diff are not zero in first epoch.")
        print("\\nSOLUTION: Ensure consistent model mode between action selection and PPO updates.")
        return True
    else:
        print("\\n✅ No significant differences found. The issue might be elsewhere.")
        return False

def test_log_prob_differences():
    """Test how model mode differences affect log probabilities"""
    print("\\n" + "="*50)
    print("Testing log probability differences...")
    
    torch.manual_seed(42)
    model = WarlightPolicyNet(
        num_nodes=42,
        num_node_features=10,
        num_edge_features=2,
        hidden_dim=64,
        n_placement_options=42,
        n_attack_options=42,
        n_army_options=10
    )
    
    # Create test data
    batch_size = 2
    node_features = torch.randn(batch_size, 42, 10)
    action_edges = torch.randint(0, 42, (batch_size, 42, 2))
    mask = torch.ones(batch_size, 42)
    
    # Simulate some actions
    placements = torch.tensor([[5, 10, 15], [2, 8, 20]])  # [batch_size, 3]
    attacks = torch.tensor([
        [[5, 10, 2, 5], [15, 20, 3, 6], [-1, -1, -1, -1]],
        [[2, 8, 1, 4], [-1, -1, -1, -1], [-1, -1, -1, -1]]
    ])  # [batch_size, 3, 4]
    
    # Get logits in training mode (like action selection)
    model.train()
    with torch.no_grad():
        placement_logits_train, attack_logits_train, army_logits_train = model(
            node_features, action_edges, mask
        )
    
    # Get logits in eval mode (like PPO update)
    model.eval()
    with torch.no_grad():
        placement_logits_eval, attack_logits_eval, army_logits_eval = model(
            node_features, action_edges, mask
        )
    
    # Compute log probabilities for the same actions using both sets of logits
    from src.agents.RLUtils.RLUtils import compute_individual_log_probs
    
    # Training mode log probs (simulating stored old log probs)
    old_placement_log_probs, old_attack_log_probs = compute_individual_log_probs(
        attacks, attack_logits_train, army_logits_train, placements,
        placement_logits_train, action_edges
    )
    
    # Eval mode log probs (simulating new log probs in PPO)
    new_placement_log_probs, new_attack_log_probs = compute_individual_log_probs(
        attacks, attack_logits_eval, army_logits_eval, placements,
        placement_logits_eval, action_edges
    )
    
    # Calculate differences (this is what should be zero in first epoch but isn't)
    placement_diff = new_placement_log_probs - old_placement_log_probs
    attack_diff = new_attack_log_probs - old_attack_log_probs
    
    print(f"\\nPlacement log prob differences:")
    print(f"  Mean absolute difference: {torch.abs(placement_diff).mean():.6f}")
    print(f"  Max absolute difference: {torch.abs(placement_diff).max():.6f}")
    print(f"  Sample differences: {placement_diff[0, :3]}")
    
    print(f"\\nAttack log prob differences:")
    print(f"  Mean absolute difference: {torch.abs(attack_diff).mean():.6f}")
    print(f"  Max absolute difference: {torch.abs(attack_diff).max():.6f}")
    print(f"  Sample differences: {attack_diff[0, :2]}")
    
    max_diff = max(torch.abs(placement_diff).max(), torch.abs(attack_diff).max())
    if max_diff > 1e-6:
        print(f"\\n🚨 CONFIRMED: Log probability differences due to model mode!")
        print(f"Maximum difference: {max_diff:.6f}")
        print("This is why placement_diff and attack_diff are not zero in first epoch.")
    else:
        print(f"\\n✅ No significant log probability differences found.")
    
    return max_diff > 1e-6

if __name__ == "__main__":
    print("Debugging model mode consistency issue...")
    print("This test checks if train/eval mode switching causes the PPO first-epoch issue.")
    
    issue_found = False
    issue_found |= test_model_mode_consistency()
    issue_found |= test_log_prob_differences()
    
    if issue_found:
        print("\\n" + "="*60)
        print("🎯 ROOT CAUSE IDENTIFIED:")
        print("The model behaves differently in train vs eval mode, causing")
        print("log probability mismatches between action selection and PPO updates.")
        print("\\n📋 RECOMMENDED FIXES:")
        print("1. Use model.eval() during action selection")
        print("2. OR remove the model.eval() call in PPO update")
        print("3. OR disable dropout/batch norm that cause train/eval differences")
    else:
        print("\\n❓ Model mode doesn't seem to be the issue. Investigation needed.")
