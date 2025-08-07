#!/usr/bin/env python3
"""
Comprehensive system test to identify the source of log probability mismatch.
This test will trace through the entire pipeline to find where the discrepancy occurs.
"""

import torch
import torch.nn.functional as f
import numpy as np
from src.agents.RLUtils.RLUtils import compute_individual_log_probs, RolloutBuffer
from src.agents.RLUtils.WarlightModel import WarlightPolicyNet

def test_log_prob_consistency():
    """
    Test the entire log probability pipeline to identify where the mismatch occurs.
    """
    print("=== COMPREHENSIVE LOG PROBABILITY SYSTEM TEST ===\n")
    
    # Create a simplified test scenario
    batch_size = 2
    num_nodes = 6
    max_attacks = 3
    max_placements = 4
    max_army_send = 5
    embed_dim = 64
    
    print("1. SETUP TEST DATA")
    print("-" * 50)
    
    # Create sample node features
    node_features = torch.randn(batch_size, num_nodes, 8)
    
    # Create sample action edges (padded to 42)
    action_edges = torch.full((batch_size, 42, 2), -1, dtype=torch.long)
    # Add some valid edges
    action_edges[0, 0] = torch.tensor([0, 1])  # Node 0 -> Node 1
    action_edges[0, 1] = torch.tensor([1, 2])  # Node 1 -> Node 2
    action_edges[0, 2] = torch.tensor([2, 3])  # Node 2 -> Node 3
    action_edges[1, 0] = torch.tensor([3, 4])  # Node 3 -> Node 4
    action_edges[1, 1] = torch.tensor([4, 5])  # Node 4 -> Node 5
    
    # Create a simple model
    model = WarlightPolicyNet(8, embed_dim)
    
    # Set up the edge tensor that the model expects
    edge_list = []
    for i in range(num_nodes - 1):
        edge_list.extend([[i, i + 1], [i + 1, i]])  # Bidirectional edges
    model.edge_tensor = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    print(f"Node features shape: {node_features.shape}")
    print(f"Action edges shape: {action_edges.shape}")
    print(f"Valid edges batch 0: {action_edges[0, :3]}")
    print(f"Valid edges batch 1: {action_edges[1, :2]}")
    
    print("\n2. GENERATE MODEL OUTPUTS")
    print("-" * 50)
    
    # Get model outputs
    placement_logits, attack_logits, army_logits = model(
        node_features, action_edges, node_features[:, :, -1]
    )
    
    print(f"Placement logits shape: {placement_logits.shape}")
    print(f"Attack logits shape: {attack_logits.shape}")
    print(f"Army logits shape: {army_logits.shape}")
    print(f"Attack logits batch 0: {attack_logits[0, :5]}")
    print(f"Army logits batch 0 (first edge): {army_logits[0, 0, :5]}")
    
    print("\n3. SIMULATE ACTION SELECTION (WITH MODIFICATIONS)")
    print("-" * 50)
    
    # Simulate placement selection (simple multinomial)
    placement_probs = f.softmax(placement_logits, dim=-1)
    placement_samples = []
    placement_log_probs_actual = []
    
    for b in range(batch_size):
        available_armies = 3 + b  # Different army counts per batch
        nodes_selected = torch.multinomial(placement_probs[b], available_armies, replacement=True)
        placement_samples.append(nodes_selected.tolist())
        
        # Compute actual log probs for the selections
        for node in nodes_selected:
            placement_log_probs_actual.append(torch.log(placement_probs[b, node] + 1e-8).item())
    
    print(f"Placement selections: {placement_samples}")
    print(f"Actual placement log probs: {placement_log_probs_actual}")
    
    # Simulate attack selection (WITHOUT MODIFICATIONS)
    attack_samples = []
    attack_log_probs_actual = []
    
    # No temperature scaling or noise
    
    for b in range(batch_size):
        # Select one attack per batch
        valid_edges_mask = action_edges[b, :, 0] >= 0
        valid_edge_count = valid_edges_mask.sum().item()
        
        if valid_edge_count > 0:
            # Edge selection
            edge_logits = attack_logits[b]  # Use all 42 edges
            edge_probs = f.softmax(edge_logits, dim=-1)
            edge_idx = torch.multinomial(edge_probs, 1).item()
            edge_log_prob = torch.log(edge_probs[edge_idx] + 1e-8).item()
            
            # Army selection (NO MODIFICATIONS)
            available_armies = 4  # Assume 4 armies available
            army_logits_raw = army_logits[b, edge_idx, :available_armies]
            
            # NO temperature scaling or noise - use raw logits directly
            army_probs = f.softmax(army_logits_raw, dim=-1)
            army_selected = torch.multinomial(army_probs, 1).item()
            army_log_prob = torch.log(army_probs[army_selected] + 1e-8).item()
            
            # Store the attack
            src, tgt = action_edges[b, edge_idx].tolist()
            attack_samples.append([src, tgt, army_selected + 1])  # +1 because army count is 1-indexed
            attack_log_probs_actual.append(edge_log_prob + army_log_prob)
            
            print(f"Batch {b}:")
            print(f"  Raw army logits: {army_logits_raw}")
            print(f"  Raw army probs: {army_probs}")
            print(f"  Selected army: {army_selected + 1}, log prob: {army_log_prob:.4f}")
            print(f"  Edge log prob: {edge_log_prob:.4f}")
            print(f"  Total: {edge_log_prob + army_log_prob:.4f}")
    
    print(f"Attack selections: {attack_samples}")
    print(f"Actual attack log probs: {attack_log_probs_actual}")
    
    print("\n4. PREPARE DATA FOR COMPUTE_INDIVIDUAL_LOG_PROBS")
    print("-" * 50)
    
    # Convert to the format expected by compute_individual_log_probs
    max_placements = max(len(p) for p in placement_samples)
    max_attacks = len(attack_samples)
    
    placements_tensor = torch.full((batch_size, max_placements), -1, dtype=torch.long)
    for i, p in enumerate(placement_samples):
        placements_tensor[i, :len(p)] = torch.tensor(p)
    
    attacks_tensor = torch.full((batch_size, max_attacks, 3), -1, dtype=torch.long)
    for i, attack in enumerate(attack_samples):
        batch_idx = i  # Simplified: one attack per batch
        if batch_idx < batch_size:
            attacks_tensor[batch_idx, 0] = torch.tensor(attack)
    
    print(f"Placements tensor: {placements_tensor}")
    print(f"Attacks tensor: {attacks_tensor}")
    
    print("\n5. COMPUTE LOG PROBS USING ORIGINAL LOGITS")
    print("-" * 50)
    
    # Use the original, unmodified logits
    computed_placement_log_probs, computed_attack_log_probs = compute_individual_log_probs(
        attacks_tensor, attack_logits, army_logits, placements_tensor,
        placement_logits, action_edges
    )
    
    print(f"Computed placement log probs shape: {computed_placement_log_probs.shape}")
    print(f"Computed placement log probs: {computed_placement_log_probs}")
    print(f"Computed attack log probs shape: {computed_attack_log_probs.shape}")
    print(f"Computed attack log probs: {computed_attack_log_probs}")
    
    print("\n6. COMPARE ACTUAL vs COMPUTED LOG PROBS")
    print("-" * 50)
    
    # Flatten actual log probs for comparison
    actual_placement_flat = placement_log_probs_actual
    actual_attack_flat = attack_log_probs_actual
    
    # Flatten computed log probs
    computed_placement_flat = computed_placement_log_probs.flatten().tolist()
    computed_attack_flat = computed_attack_log_probs.flatten().tolist()
    
    print("PLACEMENT COMPARISON:")
    for i, (actual, computed) in enumerate(zip(actual_placement_flat, computed_placement_flat)):
        diff = abs(actual - computed)
        print(f"  Action {i}: Actual={actual:.4f}, Computed={computed:.4f}, Diff={diff:.4f}")
    
    print("\nATTACK COMPARISON:")
    for i, (actual, computed) in enumerate(zip(actual_attack_flat, computed_attack_flat)):
        diff = abs(actual - computed)
        print(f"  Attack {i}: Actual={actual:.4f}, Computed={computed:.4f}, Diff={diff:.4f}")
        if diff > 0.1:
            print(f"    ⚠️  LARGE DIFFERENCE DETECTED!")
    
    print("\n7. DETAILED ATTACK LOG PROB BREAKDOWN")
    print("-" * 50)
    
    # Let's manually break down the attack log prob computation to see where it diverges
    for i, attack in enumerate(attack_samples):
        if i >= batch_size:
            break
            
        src, tgt, armies = attack
        batch_idx = i
        
        print(f"\nAttack {i} (Batch {batch_idx}): {src} -> {tgt} with {armies} armies")
        
        # Find the edge index
        edge_found = False
        for edge_idx in range(action_edges.shape[1]):
            if (action_edges[batch_idx, edge_idx, 0] == src and 
                action_edges[batch_idx, edge_idx, 1] == tgt):
                edge_found = True
                break
        
        if edge_found:
            print(f"  Edge found at index: {edge_idx}")
            
            # Edge log prob (this should match)
            edge_log_probs = f.log_softmax(attack_logits[batch_idx], dim=-1)
            edge_log_prob_computed = edge_log_probs[edge_idx].item()
            print(f"  Edge log prob (computed): {edge_log_prob_computed:.4f}")
            
            # Army log prob (this is where the difference likely is)
            army_logits_for_edge = army_logits[batch_idx, edge_idx]
            army_log_probs_raw = f.log_softmax(army_logits_for_edge, dim=-1)
            army_log_prob_computed = army_log_probs_raw[armies - 1].item()  # -1 because armies is 1-indexed
            print(f"  Army log prob (computed from raw): {army_log_prob_computed:.4f}")
            
            total_computed = edge_log_prob_computed + army_log_prob_computed
            print(f"  Total computed: {total_computed:.4f}")
            print(f"  Total actual: {actual_attack_flat[i]:.4f}")
            print(f"  Difference: {abs(total_computed - actual_attack_flat[i]):.4f}")
            
            # Show the raw army logits and probabilities
            available_armies = 4
            army_logits_raw = army_logits[batch_idx, edge_idx, :available_armies]
            print(f"  Raw army logits: {army_logits_raw}")
            print(f"  Raw army probs: {f.softmax(army_logits_raw, dim=-1)}")
        else:
            print(f"  ⚠️  Edge not found!")
    
    print("\n8. CONCLUSION")
    print("=" * 50)
    print("Testing log probability consistency without temperature scaling or noise:")
    print("1. During action selection, raw logits are used directly")
    print("2. During compute_individual_log_probs, raw logits are also used")
    print("3. This should eliminate systematic bias in log probability calculations")
    print("\nIf differences still exist, they indicate other issues in the calculation.")

if __name__ == "__main__":
    test_log_prob_consistency()
