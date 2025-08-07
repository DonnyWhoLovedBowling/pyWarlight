import torch

def test_final_fix():
    batch_size = 5
    max_attacks = 8
    
    # Simulate the exact problem: some episodes have no attacks, others have few
    old_attack_log_probs = torch.zeros(batch_size, max_attacks)
    new_attack_log_probs = torch.zeros(batch_size, max_attacks)
    
    # Episode 0: 2 real attacks
    old_attack_log_probs[0, :2] = torch.tensor([-1.2, -1.8])
    new_attack_log_probs[0, :2] = torch.tensor([-1.201, -1.799])  # Tiny difference
    
    # Episode 1: 1 real attack  
    old_attack_log_probs[1, 0] = -1.5
    new_attack_log_probs[1, 0] = -1.501  # Tiny difference
    
    # Episode 2: 3 real attacks
    old_attack_log_probs[2, :3] = torch.tensor([-1.1, -1.7, -2.0])
    new_attack_log_probs[2, :3] = torch.tensor([-1.099, -1.701, -1.999])  # Tiny differences
    
    # Episode 3: NO real attacks (all padding)
    # Episode 4: NO real attacks (all padding)
    
    print("Attack log prob structure:")
    for i in range(batch_size):
        real_count = (old_attack_log_probs[i] < -1e-6).sum()
        print(f"Episode {i}: {real_count} real attacks")
    
    # Compute differences and ratios
    attack_diff = new_attack_log_probs - old_attack_log_probs
    attack_diff = torch.clamp(attack_diff, -20, 20)
    attack_ratios = attack_diff.exp()
    
    print(f"\nAttack ratios: {attack_ratios}")
    print(f"Attack diff max: {attack_diff.abs().max()}")
    
    # Apply the new fixed approach
    valid_attack_mask = (old_attack_log_probs < -1e-6)
    episodes_with_attacks = valid_attack_mask.any(dim=1)
    
    print(f"\nEpisodes with attacks: {episodes_with_attacks}")
    
    attack_avg_ratio = torch.ones(batch_size)
    episodes_with_attacks_indices = episodes_with_attacks.nonzero(as_tuple=True)[0]
    
    if len(episodes_with_attacks_indices) > 0:
        valid_ratios = attack_ratios * valid_attack_mask.float()
        valid_count = valid_attack_mask.sum(dim=1).float()
        
        print(f"Valid count per episode: {valid_count}")
        print(f"Episodes with attacks indices: {episodes_with_attacks_indices}")
        
        attack_ratios_for_episodes = valid_ratios[episodes_with_attacks_indices].sum(dim=1) / valid_count[episodes_with_attacks_indices]
        attack_avg_ratio[episodes_with_attacks_indices] = attack_ratios_for_episodes
    
    print(f"\nFinal attack_avg_ratio: {attack_avg_ratio}")
    print(f"Max ratio: {attack_avg_ratio.max()}")
    print(f"All ratios reasonable: {(attack_avg_ratio < 10).all()}")

if __name__ == "__main__":
    test_final_fix()
