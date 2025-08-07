#!/usr/bin/env python3
"""
Final verification test to confirm log probability matching after removing temperature scaling and noise.
"""

import torch
import torch.nn.functional as f
from src.agents.RLUtils.RLUtils import compute_individual_log_probs
from src.agents.RLUtils.WarlightModel import WarlightPolicyNet

def test_final_verification():
    """Final test to verify log probabilities match perfectly."""
    print("=== FINAL VERIFICATION: LOG PROBABILITY MATCHING ===\n")
    
    batch_size = 2
    num_nodes = 6
    embed_dim = 64
    
    # Create test data
    node_features = torch.randn(batch_size, num_nodes, 8)
    action_edges = torch.full((batch_size, 42, 2), -1, dtype=torch.long)
    # Add valid edges only
    action_edges[0, 0] = torch.tensor([0, 1])
    action_edges[0, 1] = torch.tensor([1, 2])
    action_edges[1, 0] = torch.tensor([2, 3])
    action_edges[1, 1] = torch.tensor([3, 4])
    
    # Create model
    model = WarlightPolicyNet(8, embed_dim)
    edge_list = [[i, i + 1] for i in range(num_nodes - 1)] + [[i + 1, i] for i in range(num_nodes - 1)]
    model.edge_tensor = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    # Get model outputs
    placement_logits, attack_logits, army_logits = model(
        node_features, action_edges, node_features[:, :, -1]
    )
    
    print("SIMULATING ACTION SELECTION (NO MODIFICATIONS)")
    print("-" * 50)
    
    # Manual placement selection for verification
    placement_samples = []
    actual_placement_log_probs = []
    placement_probs = f.softmax(placement_logits, dim=-1)
    
    # Fixed placement selections for reproducibility
    placement_samples = [[0, 2], [1, 3, 4]]  # 2 placements for batch 0, 3 for batch 1
    
    for b, placements in enumerate(placement_samples):
        for node in placements:
            log_prob = torch.log(placement_probs[b, node] + 1e-8).item()
            actual_placement_log_probs.append(log_prob)
    
    # Manual attack selection - pick specific valid attacks
    attack_samples = []
    actual_attack_log_probs = []
    
    # Batch 0: Attack from edge 0 (0->1) with 2 armies
    b = 0
    edge_idx = 0
    armies = 2
    edge_probs = f.softmax(attack_logits[b], dim=-1)
    edge_log_prob = torch.log(edge_probs[edge_idx] + 1e-8).item()
    
    army_probs = f.softmax(army_logits[b, edge_idx], dim=-1)
    army_log_prob = torch.log(army_probs[armies - 1] + 1e-8).item()
    
    src, tgt = action_edges[b, edge_idx].tolist()
    attack_samples.append([src, tgt, armies])
    actual_attack_log_probs.append(edge_log_prob + army_log_prob)
    
    print(f"Batch 0: Attack {src}->{tgt} with {armies} armies")
    print(f"  Edge log prob: {edge_log_prob:.6f}")
    print(f"  Army log prob: {army_log_prob:.6f}")
    print(f"  Total actual: {edge_log_prob + army_log_prob:.6f}")
    
    # Batch 1: Attack from edge 0 (2->3) with 1 army
    b = 1
    edge_idx = 0
    armies = 1
    edge_probs = f.softmax(attack_logits[b], dim=-1)
    edge_log_prob = torch.log(edge_probs[edge_idx] + 1e-8).item()
    
    army_probs = f.softmax(army_logits[b, edge_idx], dim=-1)
    army_log_prob = torch.log(army_probs[armies - 1] + 1e-8).item()
    
    src, tgt = action_edges[b, edge_idx].tolist()
    attack_samples.append([src, tgt, armies])
    actual_attack_log_probs.append(edge_log_prob + army_log_prob)
    
    print(f"Batch 1: Attack {src}->{tgt} with {armies} armies")
    print(f"  Edge log prob: {edge_log_prob:.6f}")
    print(f"  Army log prob: {army_log_prob:.6f}")
    print(f"  Total actual: {edge_log_prob + army_log_prob:.6f}")
    
    print("\nUSING COMPUTE_INDIVIDUAL_LOG_PROBS")
    print("-" * 50)
    
    # Prepare tensors for compute_individual_log_probs
    max_placements = max(len(p) for p in placement_samples)
    placements_tensor = torch.full((batch_size, max_placements), -1, dtype=torch.long)
    for i, p in enumerate(placement_samples):
        placements_tensor[i, :len(p)] = torch.tensor(p)
    
    max_attacks = 1  # One attack per batch
    attacks_tensor = torch.full((batch_size, max_attacks, 3), -1, dtype=torch.long)
    for i, attack in enumerate(attack_samples):
        attacks_tensor[i, 0] = torch.tensor(attack)
    
    print(f"Placements tensor: {placements_tensor}")
    print(f"Attacks tensor: {attacks_tensor}")
    
    # Compute log probs
    computed_placement_log_probs, computed_attack_log_probs = compute_individual_log_probs(
        attacks_tensor, attack_logits, army_logits, placements_tensor,
        placement_logits, action_edges
    )
    
    print(f"Computed placement log probs: {computed_placement_log_probs}")
    print(f"Computed attack log probs: {computed_attack_log_probs}")
    
    print("\nFINAL COMPARISON")
    print("-" * 50)
    
    # Compare placements
    actual_placement_flat = actual_placement_log_probs
    computed_placement_flat = computed_placement_log_probs.flatten().tolist()
    
    print("PLACEMENT LOG PROBS:")
    max_placement_diff = 0
    for i, (actual, computed) in enumerate(zip(actual_placement_flat, computed_placement_flat)):
        diff = abs(actual - computed)
        max_placement_diff = max(max_placement_diff, diff)
        print(f"  Action {i}: Actual={actual:.8f}, Computed={computed:.8f}, Diff={diff:.8f}")
    
    # Compare attacks
    actual_attack_flat = actual_attack_log_probs
    computed_attack_flat = computed_attack_log_probs.flatten().tolist()
    
    print("\nATTACK LOG PROBS:")
    max_attack_diff = 0
    for i, (actual, computed) in enumerate(zip(actual_attack_flat, computed_attack_flat)):
        diff = abs(actual - computed)
        max_attack_diff = max(max_attack_diff, diff)
        print(f"  Attack {i}: Actual={actual:.8f}, Computed={computed:.8f}, Diff={diff:.8f}")
    
    print(f"\nSUMMARY")
    print("-" * 50)
    print(f"Max placement difference: {max_placement_diff:.8f}")
    print(f"Max attack difference: {max_attack_diff:.8f}")
    
    tolerance = 1e-6
    if max_placement_diff < tolerance and max_attack_diff < tolerance:
        print(f"✅ SUCCESS: All log probabilities match within tolerance ({tolerance})!")
        print("🎉 The log probability mismatch issue has been SOLVED!")
    else:
        print(f"❌ Some differences exceed tolerance ({tolerance})")

if __name__ == "__main__":
    test_final_verification()
