import torch

# Better reproduction of the actual issue
def test_attack_ratio_explosion():
    batch_size = 24  # Your actual batch size
    max_attacks = 15  # Let's say some episodes have many attack slots
    eps = 1e-8

    # The real issue: what if attack_diff is near zero but we have masking issues?
    # Simulate a case where attack_ratios are very close to 1, but when averaged, explode
    
    # Case 1: Normal case (first few episodes)
    attack_ratios_normal = torch.ones(3, max_attacks) * 1.01  # Slight deviation from 1
    
    # Case 2: Problematic case - what if we have a single very small valid attack?
    attack_ratios_problem = torch.ones(21, max_attacks)
    
    # Simulate old attack log probs where most entries are padding zeros
    old_attack_log_probs = torch.zeros(batch_size, max_attacks)
    
    # First 3 episodes: normal with several real attacks
    old_attack_log_probs[0, :3] = torch.tensor([-1.2, -1.8, -2.1])
    old_attack_log_probs[1, :2] = torch.tensor([-1.5, -1.9])  
    old_attack_log_probs[2, :4] = torch.tensor([-1.1, -1.7, -2.0, -1.3])
    
    # Episodes 3-23: very few real attacks, mostly padding
    for i in range(3, 24):
        if i % 3 == 0:  # Every 3rd episode has 1 real attack
            old_attack_log_probs[i, 0] = -1.5
        # Others have no real attacks (all zeros = all padding)
    
    print("Old attack log probs structure:")
    for i in range(5):
        real_attacks = (old_attack_log_probs[i] < -1e-6).sum()
        print(f"Episode {i}: {real_attacks} real attacks")
    
    # Now let's see what happens with the ratio computation
    print("\nOld approach (using != 0.0):")
    old_mask = (old_attack_log_probs != 0.0)
    for i in range(5):
        count = old_mask[i].sum().float() + eps
        print(f"Episode {i}: mask count = {count.item():.8f}")
    
    print("\nNew approach (using < -1e-6):")
    new_mask = (old_attack_log_probs < -1e-6)
    for i in range(5):
        count = new_mask[i].sum().float() + eps  
        print(f"Episode {i}: mask count = {count.item():.8f}")
    
    # The key insight: if attack_diff is near zero, attack_ratios ≈ 1
    # But if we divide by a very small count (just eps), we get explosion
    print(f"\nIf we have ratio=1.001 and divide by eps={eps}:")
    print(f"Result = {1.001 / eps}")
    
    print(f"\nIf we have ratio=1.001 and divide by 1.0:")
    print(f"Result = {1.001 / 1.0}")

if __name__ == "__main__":
    test_attack_ratio_explosion()
