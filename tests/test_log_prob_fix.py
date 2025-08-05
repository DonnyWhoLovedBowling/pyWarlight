#!/usr/bin/env python3
"""
Test script to verify that the per-action log probability fix works correctly.
This tests that individual log probabilities are computed and stored properly.
"""

import torch
import numpy as np
from src.agents.RLUtils.RLUtils import compute_individual_log_probs, compute_log_probs, RolloutBuffer

def test_individual_log_probs():
    """Test that individual log probs are computed correctly"""
    print("Testing individual log probability computation...")
    
    # Create sample data
    batch_size = 2
    num_nodes = 4
    max_attacks = 3
    max_placements = 2
    max_army_send = 5
    
    # Sample logits
    placement_logits = torch.randn(batch_size, num_nodes)
    attack_logits = torch.randn(batch_size, 42)
    army_logits = torch.randn(batch_size, 42, max_army_send)
    
    # Sample actions
    placements = torch.tensor([[0, 1], [2, -1]], dtype=torch.long)  # Second batch has only 1 placement
    attacks = torch.tensor([[[0, 1, 2], [1, 2, 1], [-1, -1, -1]], 
                           [[2, 3, 0], [-1, -1, -1], [-1, -1, -1]]], dtype=torch.long)
    
    # Sample action edges
    action_edges = torch.zeros(batch_size, 42, 2, dtype=torch.long)
    action_edges[0, 0] = torch.tensor([0, 1])  # First edge connects nodes 0->1
    action_edges[0, 1] = torch.tensor([1, 2])  # Second edge connects nodes 1->2  
    action_edges[1, 0] = torch.tensor([2, 3])  # Different edges for second batch
    # Fill remaining with -1 (padding)
    action_edges[:, 2:] = -1
    
    # Test individual log probs
    placement_log_probs, attack_log_probs = compute_individual_log_probs(
        attacks, attack_logits, army_logits, placements, placement_logits, action_edges
    )
    
    print(f"Placement log probs shape: {placement_log_probs.shape}")
    print(f"Attack log probs shape: {attack_log_probs.shape}")
    print(f"Placement log probs:\n{placement_log_probs}")
    print(f"Attack log probs:\n{attack_log_probs}")
    
    # Test that summed individual log probs match original compute_log_probs
    total_log_probs = compute_log_probs(
        attacks, attack_logits, army_logits, placements, placement_logits, action_edges
    )
    
    # Sum individual log probs
    placement_sum = placement_log_probs.sum(dim=1)
    attack_sum = attack_log_probs.sum(dim=1)
    individual_sum = placement_sum + attack_sum
    
    print(f"\nOriginal total log probs: {total_log_probs}")
    print(f"Sum of individual log probs: {individual_sum}")
    print(f"Difference: {torch.abs(total_log_probs - individual_sum)}")
    
    # Check if they match (within numerical tolerance)
    if torch.allclose(total_log_probs, individual_sum, atol=1e-6):
        print("✓ Individual log probs sum correctly to total log probs!")
    else:
        print("✗ Mismatch between individual and total log probs!")
        return False
    
    return True

def test_rollout_buffer():
    """Test that the rollout buffer works with individual log probs"""
    print("\nTesting RolloutBuffer with individual log probabilities...")
    
    buffer = RolloutBuffer()
    
    # Sample data for buffer
    edges = torch.zeros(42, 2, dtype=torch.long)
    edges[0] = torch.tensor([0, 1])
    edges[1:] = -1
    
    attacks = [[0, 1, 2]]
    placements = [0, 1]
    
    placement_log_probs = torch.tensor([0.1, 0.2])
    attack_log_probs = torch.tensor([0.3])
    
    reward = 1.0
    value = 0.5
    done = False
    
    starting_features = torch.randn(4, 8)
    post_features = torch.randn(4, 8)
    
    # Add to buffer
    buffer.add(edges, attacks, placements, placement_log_probs, attack_log_probs, 
              reward, value, done, starting_features, post_features)
    
    # Test retrieval
    retrieved_placement = buffer.get_placement_log_probs()
    retrieved_attack = buffer.get_attack_log_probs()
    retrieved_total = buffer.get_log_probs()
    
    print(f"Retrieved placement log probs: {retrieved_placement}")
    print(f"Retrieved attack log probs: {retrieved_attack}")
    print(f"Retrieved total log probs: {retrieved_total}")
    
    # Check that total matches sum of individuals
    expected_total = placement_log_probs.sum() + attack_log_probs.sum()
    if torch.allclose(retrieved_total.cpu(), expected_total.unsqueeze(0), atol=1e-6):
        print("✓ Buffer correctly stores and retrieves individual log probs!")
    else:
        print("✗ Buffer log prob retrieval mismatch!")
        return False
    
    return True

if __name__ == "__main__":
    print("Testing PPO log probability fix...")
    
    success = True
    success &= test_individual_log_probs()
    success &= test_rollout_buffer()
    
    if success:
        print("\n✓ All tests passed! The per-action log probability fix is working correctly.")
    else:
        print("\n✗ Some tests failed. Please check the implementation.")
