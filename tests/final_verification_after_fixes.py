"""
Final verification test to ensure log probability consistency after fixes
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
import torch.nn.functional as f
from src.agents.RLUtils.RLUtils import compute_individual_log_probs

def simulate_action_selection_vs_ppo_update():
    """Simulate the exact process of action selection vs PPO update"""
    print("=== Simulating Action Selection vs PPO Update ===")
    
    device = torch.device('cpu')
    
    # Create realistic game data
    num_nodes = 42
    placement_logits = torch.randn(num_nodes)
    attack_logits = torch.randn(42)
    army_logits = torch.randn(42, 10)
    action_edges = torch.tensor([[i, (i+1) % num_nodes] for i in range(42)])
    
    # Mask some placement regions (not owned by agent)
    not_owned = [0, 1, 2, 10, 11, 12, 20, 21, 22]
    placement_logits[not_owned] = float('-inf')
    
    print("=== ACTION SELECTION PHASE ===")
    
    # 1. Placement selection (NEW CORRECT METHOD)
    placement_probs = placement_logits.softmax(dim=0)
    available_armies = 5
    selected_placement_regions = torch.multinomial(placement_probs, num_samples=available_armies, replacement=True)
    
    # Compute actual log probabilities using log_softmax (FIXED)
    placement_log_probs_full = f.log_softmax(placement_logits, dim=0)
    action_selection_placement_log_probs = []
    for region in selected_placement_regions:
        action_selection_placement_log_probs.append(placement_log_probs_full[region].item())
    
    print(f"Selected placement regions: {selected_placement_regions.tolist()}")
    print(f"Action selection placement log probs: {action_selection_placement_log_probs}")
    
    # 2. Attack selection
    edge_probs = f.softmax(attack_logits, dim=0)
    num_attacks = 3
    topk_probs, selected_edge_indices = torch.topk(edge_probs, num_attacks)
    
    action_selection_attack_log_probs = []
    selected_attacks = []
    
    for edge_idx in selected_edge_indices:
        src, tgt = action_edges[edge_idx]
        available_armies_for_attack = 4  # Simulate available armies
        
        # Army selection using correct method (FIXED)
        army_logit_slice = army_logits[edge_idx][:available_armies_for_attack]
        army_probs = f.softmax(army_logit_slice, dim=0)
        selected_army_count = torch.multinomial(army_probs, 1).item()
        
        if selected_army_count > 0:  # Valid attack
            # Compute log probabilities using correct methods (FIXED)
            edge_log_prob = f.log_softmax(attack_logits, dim=0)[edge_idx].item()
            army_log_prob = f.log_softmax(army_logit_slice, dim=0)[selected_army_count].item()
            total_attack_log_prob = edge_log_prob + army_log_prob
            
            action_selection_attack_log_probs.append(total_attack_log_prob)
            selected_attacks.append((src.item(), tgt.item(), selected_army_count + 1))  # +1 for 1-indexed
    
    print(f"Selected attacks: {selected_attacks}")
    print(f"Action selection attack log probs: {action_selection_attack_log_probs}")
    
    print("\n=== PPO UPDATE PHASE ===")
    
    # Convert to format expected by compute_individual_log_probs
    # Placements: convert multinomial results to placement list
    placement_counts = torch.bincount(selected_placement_regions, minlength=num_nodes)
    placements_list = []
    for region_id, count in enumerate(placement_counts):
        placements_list.extend([region_id] * count.item())
    
    # Pad to max length
    max_placements = 10
    while len(placements_list) < max_placements:
        placements_list.append(-1)
    placements_tensor = torch.tensor([placements_list[:max_placements]])
    
    # Attacks: convert to tensor format
    max_attacks = 5
    attacks_list = selected_attacks.copy()
    while len(attacks_list) < max_attacks:
        attacks_list.append((-1, -1, -1))
    attacks_tensor = torch.tensor([attacks_list[:max_attacks]])
    
    # Add batch dimensions
    placement_logits_batch = placement_logits.unsqueeze(0)
    attack_logits_batch = attack_logits.unsqueeze(0)
    army_logits_batch = army_logits.unsqueeze(0)
    action_edges_batch = action_edges.unsqueeze(0)
    
    # Compute using PPO function
    ppo_placement_log_probs, ppo_attack_log_probs = compute_individual_log_probs(
        attacks_tensor, attack_logits_batch, army_logits_batch, 
        placements_tensor, placement_logits_batch, action_edges_batch
    )
    
    print(f"PPO placement log probs: {ppo_placement_log_probs[0].tolist()[:len(action_selection_placement_log_probs)]}")
    print(f"PPO attack log probs: {ppo_attack_log_probs[0].tolist()[:len(action_selection_attack_log_probs)]}")
    
    print("\n=== COMPARISON ===")
    
    # Compare placement log probs
    placement_differences = []
    for i, (action_lp, ppo_lp) in enumerate(zip(action_selection_placement_log_probs, ppo_placement_log_probs[0][:len(action_selection_placement_log_probs)])):
        diff = abs(action_lp - ppo_lp.item())
        placement_differences.append(diff)
        print(f"Placement {i}: action={action_lp:.6f}, ppo={ppo_lp.item():.6f}, diff={diff:.2e}")
    
    # Compare attack log probs
    attack_differences = []
    for i, (action_lp, ppo_lp) in enumerate(zip(action_selection_attack_log_probs, ppo_attack_log_probs[0][:len(action_selection_attack_log_probs)])):
        diff = abs(action_lp - ppo_lp.item())
        attack_differences.append(diff)
        print(f"Attack {i}: action={action_lp:.6f}, ppo={ppo_lp.item():.6f}, diff={diff:.2e}")
    
    print(f"\nMax placement difference: {max(placement_differences) if placement_differences else 0:.2e}")
    print(f"Max attack difference: {max(attack_differences) if attack_differences else 0:.2e}")
    
    # Success criteria
    placement_success = max(placement_differences) < 1e-5 if placement_differences else True
    attack_success = max(attack_differences) < 1e-5 if attack_differences else True
    
    print(f"\nSUCCESS: Placement consistency: {placement_success}")
    print(f"SUCCESS: Attack consistency: {attack_success}")
    print(f"OVERALL SUCCESS: {placement_success and attack_success}")

if __name__ == "__main__":
    simulate_action_selection_vs_ppo_update()
