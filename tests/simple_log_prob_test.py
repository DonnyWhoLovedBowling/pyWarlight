#!/usr/bin/env python3
"""
Simple test to verify that log probabilities match after removing temperature scaling and noise.
"""

import torch
import torch.nn.functional as f
from src.agents.RLUtils.RLUtils import compute_individual_log_probs
from src.agents.RLUtils.WarlightModel import WarlightPolicyNet

def test_simplified_log_probs():
    """Test that log probs match when using raw logits without modifications."""
    print("=== SIMPLE LOG PROBABILITY TEST (NO TEMPERATURE/NOISE) ===\n")
    
    batch_size = 2
    num_nodes = 6
    embed_dim = 64
    
    # Create test data
    node_features = torch.randn(batch_size, num_nodes, 8)
    action_edges = torch.full((batch_size, 42, 2), -1, dtype=torch.long)
    action_edges[0, 0] = torch.tensor([0, 1])
    action_edges[0, 1] = torch.tensor([1, 2])
    action_edges[1, 0] = torch.tensor([3, 4])
    
    # Create model
    model = WarlightPolicyNet(8, embed_dim)
    edge_list = [[i, i + 1] for i in range(num_nodes - 1)] + [[i + 1, i] for i in range(num_nodes - 1)]
    model.edge_tensor = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    # Get model outputs
    placement_logits, attack_logits, army_logits = model(
        node_features, action_edges, node_features[:, :, -1]
    )
    
    print("1. SIMULATE ACTION SELECTION (NO MODIFICATIONS)")
    print("-" * 50)
    
    # Placement selection
    placement_probs = f.softmax(placement_logits, dim=-1)
    placement_samples = []
    actual_placement_log_probs = []
    
    for b in range(batch_size):
        available_armies = 3
        nodes = torch.multinomial(placement_probs[b], available_armies, replacement=True)
        placement_samples.append(nodes.tolist())
        for node in nodes:
            actual_placement_log_probs.append(torch.log(placement_probs[b, node] + 1e-8).item())
    
    # Attack selection (WITHOUT temperature scaling or noise)
    attack_samples = []
    actual_attack_log_probs = []
    
    for b in range(batch_size):
        valid_edges_mask = action_edges[b, :, 0] >= 0
        valid_edge_count = valid_edges_mask.sum().item()
        
        if valid_edge_count > 0:
            # Edge selection - use all valid edges for proper probability calculation
            edge_logits = attack_logits[b, :]  # Use full 42-edge tensor (not just valid ones)
            edge_probs = f.softmax(edge_logits, dim=-1)
            
            # Select a valid edge (make sure it's one that exists in action_edges)
            valid_indices = torch.where(valid_edges_mask)[0]
            if len(valid_indices) > 0:
                selected_valid_idx = torch.randint(0, len(valid_indices), (1,)).item()
                edge_idx = valid_indices[selected_valid_idx].item()
            else:
                continue  # Skip if no valid edges
            
            edge_log_prob = torch.log(edge_probs[edge_idx] + 1e-8).item()
            
            # Army selection (RAW LOGITS - NO MODIFICATIONS)
            available_armies = 4
            army_logits_raw = army_logits[b, edge_idx, :available_armies]
            army_probs_raw = f.softmax(army_logits_raw, dim=-1)
            army_selected = torch.multinomial(army_probs_raw, 1).item()
            army_log_prob_raw = torch.log(army_probs_raw[army_selected] + 1e-8).item()
            
            src, tgt = action_edges[b, edge_idx].tolist()
            attack_samples.append([src, tgt, army_selected + 1])  # +1 for 1-indexed armies
            actual_attack_log_probs.append(edge_log_prob + army_log_prob_raw)
            
            print(f"Batch {b}: Attack {src}->{tgt}, armies={army_selected + 1}")
            print(f"  Edge index: {edge_idx} (out of {valid_edge_count} valid)")
            print(f"  Edge log prob: {edge_log_prob:.6f}")
            print(f"  Army log prob: {army_log_prob_raw:.6f}")
            print(f"  Total: {edge_log_prob + army_log_prob_raw:.6f}")
    
    print("\n2. COMPUTE LOG PROBS USING COMPUTE_INDIVIDUAL_LOG_PROBS")
    print("-" * 50)
    
    # Prepare tensors - each batch gets its own attacks
    max_placements = max(len(p) for p in placement_samples)
    placements_tensor = torch.full((batch_size, max_placements), -1, dtype=torch.long)
    for i, p in enumerate(placement_samples):
        placements_tensor[i, :len(p)] = torch.tensor(p)
    
    # Create attacks tensor where each batch has its own attacks
    max_attacks = 1  # Assume 1 attack per batch for this test
    attacks_tensor = torch.full((batch_size, max_attacks, 3), -1, dtype=torch.long)
    for b in range(batch_size):
        if b < len(attack_samples):
            attacks_tensor[b, 0] = torch.tensor(attack_samples[b])
    
    print(f"Attacks tensor shape: {attacks_tensor.shape}")
    print(f"Attacks tensor: {attacks_tensor}")
    
    # Compute log probs
    computed_placement_log_probs, computed_attack_log_probs = compute_individual_log_probs(
        attacks_tensor, attack_logits, army_logits, placements_tensor,
        placement_logits, action_edges
    )
    
    print("\n3. COMPARE RESULTS")
    print("-" * 50)
    
    print("PLACEMENT LOG PROBS:")
    actual_flat = actual_placement_log_probs
    computed_flat = computed_placement_log_probs.flatten().tolist()
    for i, (actual, computed) in enumerate(zip(actual_flat, computed_flat)):
        diff = abs(actual - computed)
        print(f"  Action {i}: Actual={actual:.6f}, Computed={computed:.6f}, Diff={diff:.6f}")
    
    print("\nATTACK LOG PROBS:")
    actual_attack_flat = actual_attack_log_probs
    computed_attack_flat = computed_attack_log_probs.flatten().tolist()
    for i, (actual, computed) in enumerate(zip(actual_attack_flat, computed_attack_flat)):
        diff = abs(actual - computed)
        print(f"  Attack {i}: Actual={actual:.6f}, Computed={computed:.6f}, Diff={diff:.6f}")
        if diff > 0.001:
            print(f"    ⚠️  DIFFERENCE > 0.001!")
        else:
            print(f"    ✅ MATCH!")
    
    # Check for overall success
    max_placement_diff = max(abs(a - c) for a, c in zip(actual_flat, computed_flat))
    max_attack_diff = max(abs(a - c) for a, c in zip(actual_attack_flat, computed_attack_flat)) if actual_attack_flat else 0
    
    print(f"\n4. SUMMARY")
    print("-" * 50)
    print(f"Max placement difference: {max_placement_diff:.6f}")
    print(f"Max attack difference: {max_attack_diff:.6f}")
    
    if max_placement_diff < 0.001 and max_attack_diff < 0.001:
        print("✅ SUCCESS: All log probabilities match within tolerance!")
    else:
        print("❌ FAILURE: Some log probabilities still differ significantly")

if __name__ == "__main__":
    test_simplified_log_probs()
