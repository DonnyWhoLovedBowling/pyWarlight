"""
Accurate simulation of the actual RLGNNAgent flow
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
import torch.nn.functional as f
from src.agents.RLUtils.RLUtils import compute_individual_log_probs

def test_actual_flow():
    """Test the actual flow from RLGNNAgent"""
    print("=== Testing Actual RLGNNAgent Flow ===")
    
    # Simulate exact same data flow as in RLGNNAgent
    device = torch.device('cpu')
    num_nodes = 42
    
    # Create test data
    placement_logits = torch.randn(num_nodes)
    attack_logits = torch.randn(42)  
    army_logits = torch.randn(42, 10)
    action_edges = torch.tensor([[i, (i+1) % num_nodes] for i in range(42)])
    
    # Simulate not owning some regions
    not_owned = list(range(10)) + list(range(20, 30))
    placement_logits[not_owned] = float('-inf')
    
    print("=== PLACE_ARMIES PHASE ===")
    
    # Simulate place_armies method
    available_armies = 5
    placement_probs = placement_logits.softmax(dim=0)
    nodes = torch.multinomial(placement_probs, num_samples=available_armies, replacement=True)
    
    # Actual log probabilities (FIXED METHOD) - match RLGNNAgent ordering
    placement_log_probs_full = f.log_softmax(placement_logits, dim=0)
    placement_bincount = torch.bincount(nodes, minlength=num_nodes)
    actual_placement_log_probs = []
    # Store log probabilities in the same order as get_placements() will return them
    for ix, p in enumerate(placement_bincount.tolist()):
        if p > 0:
            # For each army placed in this region, add the log probability
            for _ in range(p):
                actual_placement_log_probs.append(placement_log_probs_full[ix].item())
    actual_placement_log_probs_tensor = torch.tensor(actual_placement_log_probs)
    
    print(f"Selected nodes: {nodes.tolist()}")
    print(f"Actual placement log probs: {actual_placement_log_probs}")
    
    # Convert to placements list (as done in get_placements method)
    placement_bincount = torch.bincount(nodes, minlength=num_nodes)
    placements_list = []
    for ix, p in enumerate(placement_bincount.tolist()):
        if p > 0:
            # For each army placed in this region, add the region ID
            for _ in range(p):
                placements_list.append(ix)
    
    print(f"Placements list: {placements_list}")
    
    print("\n=== ATTACK_TRANSFER PHASE ===")
    
    # Simulate attack selection (sample_n_attacks)
    probs = torch.softmax(attack_logits, dim=0)
    k = min(3, probs.size(0))
    topk_probs, selected_idxs = torch.topk(probs, k)
    
    edges = []
    for idx in selected_idxs.tolist():
        src, tgt = action_edges[idx]
        if src != tgt:
            edges.append((src.item(), tgt.item()))
    
    print(f"Selected edges: {edges}")
    
    # Simulate create_attack_transfers  
    edge_log_probs = f.log_softmax(attack_logits, dim=0)
    actual_attack_log_probs = []
    final_attacks = []
    
    for src, tgt in edges:
        mask = (action_edges[:, 0] == src) & (action_edges[:, 1] == tgt)
        indices = mask.nonzero(as_tuple=False)
        idx = indices.item()
        
        available_armies = 5  # Simulate available armies
        army_logit = army_logits[idx][:available_armies]
        army_probs = f.softmax(army_logit, dim=-1)
        k = int(torch.distributions.Categorical(probs=army_probs).sample().int())
        
        if k > 0:
            # Compute actual log probability (FIXED METHOD)
            edge_log_prob = edge_log_probs[idx].item()
            army_log_prob = f.log_softmax(army_logit, dim=-1)[k].item()
            total_attack_log_prob = edge_log_prob + army_log_prob
            actual_attack_log_probs.append(total_attack_log_prob)
            final_attacks.append((src, tgt, k + 1))  # +1 for 1-indexed armies
    
    actual_attack_log_probs_tensor = torch.tensor(actual_attack_log_probs) if actual_attack_log_probs else torch.tensor([])
    
    print(f"Final attacks: {final_attacks}")
    print(f"Actual attack log probs: {actual_attack_log_probs}")
    
    print("\n=== END_MOVE PHASE (PPO Computation) ===")
    
    # Convert to format for compute_individual_log_probs
    max_placements = 10
    max_attacks = 5
    
    # Placements tensor
    placements_padded = placements_list.copy()
    while len(placements_padded) < max_placements:
        placements_padded.append(-1)
    placements_tensor = torch.tensor([placements_padded[:max_placements]])
    
    # Attacks tensor  
    attacks_padded = final_attacks.copy()
    while len(attacks_padded) < max_attacks:
        attacks_padded.append((-1, -1, -1))
    attacks_tensor = torch.tensor([attacks_padded[:max_attacks]])
    
    # Batch dimensions
    placement_logits_batch = placement_logits.unsqueeze(0)
    attack_logits_batch = attack_logits.unsqueeze(0)
    army_logits_batch = army_logits.unsqueeze(0)
    action_edges_batch = action_edges.unsqueeze(0)
    
    # Compute using PPO function
    ppo_placement_log_probs, ppo_attack_log_probs = compute_individual_log_probs(
        attacks_tensor, attack_logits_batch, army_logits_batch,
        placements_tensor, placement_logits_batch, action_edges_batch
    )
    
    print(f"PPO placement log probs: {ppo_placement_log_probs[0].tolist()[:len(placements_list)]}")
    print(f"PPO attack log probs: {ppo_attack_log_probs[0].tolist()[:len(final_attacks)]}")
    
    print("\n=== COMPARISON ===")
    
    # Compare placement log probs
    print("Placement comparison:")
    placement_diffs = []
    for i in range(len(placements_list)):
        if i < len(actual_placement_log_probs):
            action_lp = actual_placement_log_probs[i]
            ppo_lp = ppo_placement_log_probs[0, i].item()
            diff = abs(action_lp - ppo_lp)
            placement_diffs.append(diff)
            print(f"  {i}: action={action_lp:.6f}, ppo={ppo_lp:.6f}, diff={diff:.2e}")
    
    # Compare attack log probs  
    print("Attack comparison:")
    attack_diffs = []
    for i in range(len(final_attacks)):
        if i < len(actual_attack_log_probs):
            action_lp = actual_attack_log_probs[i]
            ppo_lp = ppo_attack_log_probs[0, i].item()
            diff = abs(action_lp - ppo_lp)
            attack_diffs.append(diff)
            print(f"  {i}: action={action_lp:.6f}, ppo={ppo_lp:.6f}, diff={diff:.2e}")
    
    max_placement_diff = max(placement_diffs) if placement_diffs else 0
    max_attack_diff = max(attack_diffs) if attack_diffs else 0
    
    print(f"\nMax placement difference: {max_placement_diff:.2e}")
    print(f"Max attack difference: {max_attack_diff:.2e}")
    
    success = (max_placement_diff < 1e-5 and max_attack_diff < 1e-5)
    print(f"SUCCESS: {success}")
    
    if not success:
        print("\nDEBUG INFO:")
        print(f"Placement logits shape: {placement_logits.shape}")
        print(f"Attack logits shape: {attack_logits.shape}")
        print(f"Army logits shape: {army_logits.shape}")
        print(f"Action edges shape: {action_edges.shape}")

if __name__ == "__main__":
    test_actual_flow()
