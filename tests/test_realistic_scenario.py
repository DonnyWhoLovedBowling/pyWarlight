"""
Test log probability computation with realistic game data
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
import torch.nn.functional as f
from src.agents.RLUtils.RLUtils import compute_individual_log_probs

def test_realistic_scenario():
    """Test with realistic game scenario data"""
    print("=== Testing Realistic Scenario ===")
    
    device = torch.device('cpu')
    
    # Simulate real game data
    num_nodes = 42  # World map has 42 regions
    batch_size = 1
    
    # Real placement data
    placement_logits = torch.randn(batch_size, num_nodes)
    placements = torch.tensor([[5, 10, 15, 20, 25]])  # 5 army placements
    
    # Real attack data
    attack_logits = torch.randn(batch_size, 42)  # 42 possible edges (padded)
    army_logits = torch.randn(batch_size, 42, 10)  # Up to 10 armies per attack
    
    # Simulate 2 attacks: region 5->10 with 3 armies, region 15->20 with 2 armies
    attacks = torch.tensor([[[5, 10, 3], [15, 20, 2], [-1, -1, -1]]])  # Padded
    
    # Create action edges that include our attack edges
    action_edges = torch.zeros(batch_size, 42, 2, dtype=torch.long)
    # Fill with some valid edges including our attack edges
    for i in range(42):
        src = i % num_nodes
        tgt = (i + 1) % num_nodes
        action_edges[0, i] = torch.tensor([src, tgt])
    
    # Make sure our specific attacks are in the action edges
    action_edges[0, 0] = torch.tensor([5, 10])  # First edge is 5->10
    action_edges[0, 1] = torch.tensor([15, 20])  # Second edge is 15->20
    
    print(f"Action edges (first 5): {action_edges[0, :5]}")
    print(f"Attacks: {attacks[0]}")
    
    # Test compute_individual_log_probs
    placement_log_probs, attack_log_probs = compute_individual_log_probs(
        attacks, attack_logits, army_logits, placements, placement_logits, action_edges
    )
    
    print(f"Placement log probs: {placement_log_probs[0]}")
    print(f"Attack log probs: {attack_log_probs[0]}")
    
    # Manually compute for comparison
    print("\n--- Manual Computation ---")
    
    # Placement log probs
    placement_log_probs_full = f.log_softmax(placement_logits[0], dim=0)
    manual_placement_log_probs = []
    for region in placements[0]:
        if region >= 0:
            manual_placement_log_probs.append(placement_log_probs_full[region].item())
        else:
            manual_placement_log_probs.append(0.0)
    print(f"Manual placement log probs: {manual_placement_log_probs}")
    
    # Attack log probs
    edge_log_probs = f.log_softmax(attack_logits[0], dim=0)
    manual_attack_log_probs = []
    
    for i, (src, tgt, armies) in enumerate(attacks[0]):
        if src >= 0 and tgt >= 0:
            # Find the edge index
            edge_idx = None
            for j in range(42):
                if action_edges[0, j, 0] == src and action_edges[0, j, 1] == tgt:
                    edge_idx = j
                    break
            
            if edge_idx is not None:
                edge_log_prob = edge_log_probs[edge_idx].item()
                
                # Army log prob (armies is 1-indexed, convert to 0-indexed)
                army_idx = armies - 1
                army_log_probs_for_edge = f.log_softmax(army_logits[0, edge_idx], dim=0)
                army_log_prob = army_log_probs_for_edge[army_idx].item()
                
                total_log_prob = edge_log_prob + army_log_prob
                manual_attack_log_probs.append(total_log_prob)
                
                print(f"Attack {i}: edge {edge_idx} ({src}->{tgt}), armies {armies}")
                print(f"  Edge log prob: {edge_log_prob:.6f}")
                print(f"  Army log prob: {army_log_prob:.6f}")
                print(f"  Total: {total_log_prob:.6f}")
            else:
                manual_attack_log_probs.append(0.0)
                print(f"Attack {i}: edge not found")
        else:
            manual_attack_log_probs.append(0.0)
            print(f"Attack {i}: invalid (padding)")
    
    print(f"Manual attack log probs: {manual_attack_log_probs}")
    
    # Compare
    print("\n--- Comparison ---")
    for i in range(len(manual_placement_log_probs)):
        diff = abs(placement_log_probs[0, i].item() - manual_placement_log_probs[i])
        print(f"Placement {i}: func={placement_log_probs[0, i].item():.6f}, manual={manual_placement_log_probs[i]:.6f}, diff={diff:.2e}")
    
    for i in range(len(manual_attack_log_probs)):
        diff = abs(attack_log_probs[0, i].item() - manual_attack_log_probs[i])
        print(f"Attack {i}: func={attack_log_probs[0, i].item():.6f}, manual={manual_attack_log_probs[i]:.6f}, diff={diff:.2e}")

if __name__ == "__main__":
    test_realistic_scenario()
