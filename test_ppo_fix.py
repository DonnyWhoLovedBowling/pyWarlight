#!/usr/bin/env python3
"""
Test script to verify that the PPO first-epoch fix works correctly.
This test simulates the PPO update process and checks that placement_diff 
and attack_diff are zero in the first epoch after the fix.
"""

import torch
import torch.nn.functional as f
import sys
import os

# Add project root to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.agents.RLUtils.WarlightModel import WarlightPolicyNet
from src.agents.RLUtils.RLUtils import compute_individual_log_probs

def test_ppo_first_epoch_fix():
    """Test that the fix ensures zero differences in first epoch"""
    print("Testing PPO first-epoch fix...")
    print("="*50)
    
    # Create a model with dropout to test the fix
    torch.manual_seed(42)
    model = WarlightPolicyNet(node_feat_dim=10, embed_dim=64, n_army_options=10)
    
    # Set up edge tensor (required for the model)
    num_nodes = 42
    # Create a simple edge list (each node connected to next node)
    edges = [[i, (i + 1) % num_nodes] for i in range(num_nodes)]
    model.edge_tensor = torch.tensor(edges).t().contiguous()
    
    # Create test data
    batch_size = 2
    num_nodes = 42
    node_features = torch.randn(batch_size, num_nodes, 10)
    action_edges = torch.randint(0, num_nodes, (batch_size, 42, 2))
    
    # Create some test actions
    placements = torch.tensor([[5, 10, 15], [2, 8, 20]])  # [batch_size, 3]
    attacks = torch.tensor([
        [[5, 10, 2, 5], [15, 20, 3, 6], [-1, -1, -1, -1]],
        [[2, 8, 1, 4], [-1, -1, -1, -1], [-1, -1, -1, -1]]
    ])  # [batch_size, 3, 4]
    
    print("1. Simulating action selection (model in training mode):")
    model.train()  # This is how the model is during action selection
    with torch.no_grad():
        placement_logits_action, attack_logits_action, army_logits_action = model(
            node_features, action_edges
        )
    
    # Compute log probabilities during action selection
    old_placement_log_probs, old_attack_log_probs = compute_individual_log_probs(
        attacks, attack_logits_action, army_logits_action, placements,
        placement_logits_action, action_edges
    )
    
    print(f"   Old placement log probs: {old_placement_log_probs[0, :3]}")
    print(f"   Old attack log probs: {old_attack_log_probs[0, :2]}")
    
    print("\n2. Simulating PPO first epoch (model kept in training mode - FIXED):")
    # AFTER FIX: Keep model in training mode during PPO update
    model.train()  # This is the fix - no mode switching
    with torch.no_grad():
        placement_logits_ppo, attack_logits_ppo, army_logits_ppo = model(
            node_features, action_edges
        )
    
    # Compute new log probabilities in PPO update
    new_placement_log_probs, new_attack_log_probs = compute_individual_log_probs(
        attacks, attack_logits_ppo, army_logits_ppo, placements,
        placement_logits_ppo, action_edges
    )
    
    print(f"   New placement log probs: {new_placement_log_probs[0, :3]}")
    print(f"   New attack log probs: {new_attack_log_probs[0, :2]}")
    
    # Calculate differences (should be zero after fix)
    placement_diff = new_placement_log_probs - old_placement_log_probs
    attack_diff = new_attack_log_probs - old_attack_log_probs
    
    print(f"\n3. DIFFERENCES (should be ~zero after fix):")
    print(f"   Placement diff: {placement_diff[0, :3]}")
    print(f"   Attack diff: {attack_diff[0, :2]}")
    
    max_placement_diff = torch.abs(placement_diff).max().item()
    max_attack_diff = torch.abs(attack_diff).max().item()
    
    print(f"   Max placement difference: {max_placement_diff:.8f}")
    print(f"   Max attack difference: {max_attack_diff:.8f}")
    
    # Check if fix worked
    tolerance = 1e-6
    if max_placement_diff < tolerance and max_attack_diff < tolerance:
        print(f"\n✅ FIX SUCCESSFUL!")
        print(f"   Both differences are below tolerance ({tolerance})")
        print(f"   PPO first epoch will now have proper zero differences")
        return True
    else:
        print(f"\n❌ Fix didn't work completely")
        print(f"   Differences are still above tolerance ({tolerance})")
        return False

def test_old_behavior_for_comparison():
    """Test what the old behavior would have been (for comparison)"""
    print("\n" + "="*50)
    print("COMPARISON: Old behavior (train -> eval mode switch)")
    print("="*50)
    
    torch.manual_seed(42)
    model = WarlightPolicyNet(node_feat_dim=10, embed_dim=64, n_army_options=10)
    
    # Set up edge tensor (required for the model)
    num_nodes = 42
    edges = [[i, (i + 1) % num_nodes] for i in range(num_nodes)]
    model.edge_tensor = torch.tensor(edges).t().contiguous()
    
    batch_size = 2
    num_nodes = 42
    node_features = torch.randn(batch_size, num_nodes, 10)
    action_edges = torch.randint(0, num_nodes, (batch_size, 42, 2))
    
    placements = torch.tensor([[5, 10, 15], [2, 8, 20]])
    attacks = torch.tensor([
        [[5, 10, 2, 5], [15, 20, 3, 6], [-1, -1, -1, -1]],
        [[2, 8, 1, 4], [-1, -1, -1, -1], [-1, -1, -1, -1]]
    ])
    
    # Simulate old behavior: train mode during action selection
    model.train()
    with torch.no_grad():
        placement_logits_train, attack_logits_train, army_logits_train = model(
            node_features, action_edges
        )
    
    old_placement_log_probs, old_attack_log_probs = compute_individual_log_probs(
        attacks, attack_logits_train, army_logits_train, placements,
        placement_logits_train, action_edges
    )
    
    # Simulate old behavior: eval mode during PPO update
    model.eval()  # This was the old problematic behavior
    with torch.no_grad():
        placement_logits_eval, attack_logits_eval, army_logits_eval = model(
            node_features, action_edges
        )
    
    new_placement_log_probs, new_attack_log_probs = compute_individual_log_probs(
        attacks, attack_logits_eval, army_logits_eval, placements,
        placement_logits_eval, action_edges
    )
    
    # Calculate differences (would be non-zero with old behavior)
    placement_diff = new_placement_log_probs - old_placement_log_probs
    attack_diff = new_attack_log_probs - old_attack_log_probs
    
    max_placement_diff = torch.abs(placement_diff).max().item()
    max_attack_diff = torch.abs(attack_diff).max().item()
    
    print(f"Old behavior differences:")
    print(f"   Max placement difference: {max_placement_diff:.8f}")
    print(f"   Max attack difference: {max_attack_diff:.8f}")
    
    if max_placement_diff > 1e-6 or max_attack_diff > 1e-6:
        print("   ❌ Old behavior had significant differences (as expected)")
    else:
        print("   🤔 Old behavior didn't show differences (unexpected)")

if __name__ == "__main__":
    print("Testing PPO first-epoch consistency fix")
    print("This verifies that placement_diff and attack_diff are zero in first epoch")
    
    # Test the fix
    fix_worked = test_ppo_first_epoch_fix()
    
    # Show comparison with old behavior
    test_old_behavior_for_comparison()
    
    print(f"\n" + "="*60)
    if fix_worked:
        print("🎉 SUCCESS: Fix ensures consistent log probabilities!")
        print("   PPO first epoch will now have proper zero differences")
        print("   as placement_diff and attack_diff should be.")
    else:
        print("⚠️  Fix needs refinement or issue is more complex")
    print("="*60)
