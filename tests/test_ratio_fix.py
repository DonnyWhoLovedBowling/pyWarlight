import torch

# Simulate the attack ratio explosion scenario
def test_attack_ratio_fix():
    batch_size = 5
    max_attacks = 10
    eps = 1e-8

    # Simulate old attack log probs - most are padding (0.0), few are real attacks (negative)
    old_attack_log_probs = torch.tensor([
        [-2.1, -1.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 2 real attacks
        [-1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # 1 real attack
        [-2.3, -1.9, -2.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # 3 real attacks
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],     # 0 real attacks (all padding)
        [-1.7, -2.0, -1.8, -2.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # 4 real attacks
    ])
    
    # Simulate new attack log probs - very similar but with tiny differences
    new_attack_log_probs = old_attack_log_probs + torch.randn_like(old_attack_log_probs) * 0.001
    # Keep padding as exactly 0
    new_attack_log_probs[old_attack_log_probs == 0.0] = 0.0
    
    # Compute differences and ratios
    attack_diff = new_attack_log_probs - old_attack_log_probs
    attack_diff = torch.clamp(attack_diff, -20, 20)
    attack_ratios = attack_diff.exp()
    
    print("Old approach (problematic):")
    print("attack_diff sample:", attack_diff[1])  # Episode with 1 real attack
    print("attack_ratios sample:", attack_ratios[1])
    
    # Old problematic approach
    old_mask = (old_attack_log_probs != 0.0)
    if old_mask.any():
        old_valid_ratios = attack_ratios * old_mask.float()
        old_count = old_mask.sum(dim=1).float() + eps
        old_avg_ratio = old_valid_ratios.sum(dim=1) / old_count
        print("Old count per episode:", old_count)
        print("Old avg ratios:", old_avg_ratio)
    
    print("\nNew approach (fixed):")
    # New fixed approach
    valid_attack_mask = (old_attack_log_probs < -1e-6)  # Real log probs are negative
    print("Valid attack mask sample:", valid_attack_mask[1])
    
    if valid_attack_mask.any():
        valid_ratios = attack_ratios * valid_attack_mask.float()
        valid_count = valid_attack_mask.sum(dim=1).float() + eps
        new_avg_ratio = valid_ratios.sum(dim=1) / valid_count
        print("New count per episode:", valid_count)
        print("New avg ratios:", new_avg_ratio)
    
    print("\nComparison:")
    print("Old had extreme ratios:", (old_avg_ratio > 10).any().item())
    print("New has reasonable ratios:", (new_avg_ratio < 10).all().item())

if __name__ == "__main__":
    test_attack_ratio_fix()
