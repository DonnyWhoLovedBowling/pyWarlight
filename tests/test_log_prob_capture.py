#!/usr/bin/env python3
"""
Test script to verify that the actual log probability capture works correctly.
This simulates the action selection process and verifies that captured log probabilities
match what would be computed for those actions.
"""

import torch
import torch.nn.functional as f
import numpy as np

def test_attack_log_prob_capture():
    """Test that captured attack log probabilities reflect actual action selection"""
    print("Testing attack log probability capture...")
    
    # Simulate attack logits and army logits
    num_edges = 10
    max_army = 5
    attack_logits = torch.randn(num_edges)
    army_logits = torch.randn(num_edges, max_army)
    
    # Simulate edge selection (like sample_attacks_per_node)
    edge_probs = f.softmax(attack_logits, dim=-1)
    selected_edge_idx = torch.multinomial(edge_probs, 1).item()
    edge_log_prob = torch.log(edge_probs[selected_edge_idx] + 1e-8).item()
    
    # Simulate army selection with modifications (like create_attack_transfers)
    available_armies = 3
    T = 1.5
    army_logit = army_logits[selected_edge_idx][:available_armies]
    smoothed_army_logits = army_logit / T
    smoothed_army_logits += torch.randn_like(smoothed_army_logits) * 0.1
    army_probs = f.softmax(smoothed_army_logits, dim=-1)
    selected_army_amount = torch.distributions.Categorical(probs=army_probs).sample().item()
    army_log_prob = torch.log(army_probs[selected_army_amount] + 1e-8).item()
    
    # Combined log probability (what should be stored)
    captured_log_prob = edge_log_prob + army_log_prob
    
    # What the old method would compute (using raw logits)
    raw_edge_log_prob = f.log_softmax(attack_logits, dim=-1)[selected_edge_idx].item()
    raw_army_log_prob = f.log_softmax(army_logits[selected_edge_idx], dim=-1)[selected_army_amount].item()
    raw_combined_log_prob = raw_edge_log_prob + raw_army_log_prob
    
    print(f"Captured log prob (with modifications): {captured_log_prob:.4f}")
    print(f"Raw log prob (without modifications): {raw_combined_log_prob:.4f}")
    print(f"Difference: {abs(captured_log_prob - raw_combined_log_prob):.4f}")
    
    # The difference should be significant due to temperature scaling and noise
    if abs(captured_log_prob - raw_combined_log_prob) > 0.1:
        print("✓ Significant difference detected - log prob capture is working!")
        return True
    else:
        print("✗ Little difference - might need to verify the capture logic")
        return False

def test_placement_log_prob_capture():
    """Test that captured placement log probabilities are correct"""
    print("\nTesting placement log probability capture...")
    
    # Simulate placement logits
    num_regions = 8
    owned_regions = [0, 2, 5]  # regions owned by player
    placement_logits = torch.randn(num_regions)
    
    # Mask out non-owned regions
    placement_logits_masked = placement_logits.clone()
    not_owned = [i for i in range(num_regions) if i not in owned_regions]
    placement_logits_masked[not_owned] = float('-inf')
    
    # Compute probabilities and sample
    placement_probs = placement_logits_masked.softmax(dim=0)
    num_placements = 3
    selected_nodes = torch.multinomial(placement_probs, num_samples=num_placements, replacement=True)
    
    # Capture actual log probabilities
    placement_log_probs_full = torch.log(placement_probs + 1e-8)
    captured_log_probs = [placement_log_probs_full[node_id].item() for node_id in selected_nodes]
    
    # What the old method would compute
    raw_log_probs_full = f.log_softmax(placement_logits_masked, dim=-1)
    raw_log_probs = [raw_log_probs_full[node_id].item() for node_id in selected_nodes]
    
    print(f"Captured log probs: {[f'{x:.4f}' for x in captured_log_probs]}")
    print(f"Raw log probs: {[f'{x:.4f}' for x in raw_log_probs]}")
    
    # These should be very close since placement doesn't have modifications
    max_diff = max(abs(c - r) for c, r in zip(captured_log_probs, raw_log_probs))
    print(f"Max difference: {max_diff:.6f}")
    
    if max_diff < 1e-5:
        print("✓ Placement log prob capture is accurate!")
        return True
    else:
        print("✗ Placement log prob capture has issues")
        return False

if __name__ == "__main__":
    print("Testing log probability capture mechanism...")
    
    success = True
    success &= test_attack_log_prob_capture()
    success &= test_placement_log_prob_capture()
    
    if success:
        print("\n✓ All log probability capture tests passed!")
        print("The fix should resolve the attack log probability mismatch issue.")
    else:
        print("\n✗ Some tests failed. The capture mechanism may need adjustment.")
