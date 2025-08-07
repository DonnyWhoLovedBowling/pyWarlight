"""
Quick test of actual PPO training to see if log probability differences are now acceptable
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
import torch.nn.functional as f
from src.agents.RLUtils.RLUtils import compute_individual_log_probs

def test_ppo_ratio_computation():
    """Test PPO ratio computation with realistic data"""
    print("=== Testing PPO Ratio Computation ===")
    
    # Simulate stored action data and recomputed log probs
    stored_placement_log_probs = torch.tensor([-3.2, -2.8, -4.1, -2.5, -3.7])
    stored_attack_log_probs = torch.tensor([-4.2, -3.8, -5.1])
    
    # Simulate recomputed log probs with small differences (similar to our test)
    recomputed_placement_log_probs = stored_placement_log_probs  # Perfect match
    recomputed_attack_log_probs = stored_attack_log_probs + torch.tensor([0.5, -0.6, 0.7])  # Small differences
    
    print(f"Stored placement log probs: {stored_placement_log_probs.tolist()}")
    print(f"Recomputed placement log probs: {recomputed_placement_log_probs.tolist()}")
    print(f"Placement differences: {torch.abs(stored_placement_log_probs - recomputed_placement_log_probs).tolist()}")
    
    print(f"Stored attack log probs: {stored_attack_log_probs.tolist()}")
    print(f"Recomputed attack log probs: {recomputed_attack_log_probs.tolist()}")
    print(f"Attack differences: {torch.abs(stored_attack_log_probs - recomputed_attack_log_probs).tolist()}")
    
    # Compute PPO ratios
    placement_ratios = torch.exp(recomputed_placement_log_probs - stored_placement_log_probs)
    attack_ratios = torch.exp(recomputed_attack_log_probs - stored_attack_log_probs)
    
    print(f"Placement ratios: {placement_ratios.tolist()}")
    print(f"Attack ratios: {attack_ratios.tolist()}")
    
    # Check if ratios are reasonable for PPO (should be close to 1.0)
    placement_ratio_range = (placement_ratios.min().item(), placement_ratios.max().item())
    attack_ratio_range = (attack_ratios.min().item(), attack_ratios.max().item())
    
    print(f"Placement ratio range: {placement_ratio_range}")
    print(f"Attack ratio range: {attack_ratio_range}")
    
    # Check if any ratios are extreme (outside [0.1, 10] range that PPO clamps to)
    extreme_placement_ratios = ((placement_ratios < 0.1) | (placement_ratios > 10)).any()
    extreme_attack_ratios = ((attack_ratios < 0.1) | (attack_ratios > 10)).any()
    
    print(f"Extreme placement ratios: {extreme_placement_ratios}")
    print(f"Extreme attack ratios: {extreme_attack_ratios}")
    
    # Test with larger differences (like the original ~200 differences)
    print("\n=== Testing with Large Differences (Original Problem) ===")
    large_diff_attack_log_probs = stored_attack_log_probs + torch.tensor([200.0, -150.0, 100.0])
    large_attack_ratios = torch.exp(large_diff_attack_log_probs - stored_attack_log_probs)
    
    print(f"Large difference attack log probs: {large_diff_attack_log_probs.tolist()}")
    print(f"Large difference attack ratios: {large_attack_ratios.tolist()}")
    print(f"Would be clamped to: {torch.clamp(large_attack_ratios, 0.1, 10.0).tolist()}")
    
    print("\n=== Summary ===")
    current_max_attack_diff = torch.abs(stored_attack_log_probs - recomputed_attack_log_probs).max().item()
    print(f"Current max attack log prob difference: {current_max_attack_diff:.2f}")
    print(f"Current max attack ratio: {attack_ratios.max().item():.2f}")
    print(f"Is this acceptable for PPO? {not extreme_attack_ratios and attack_ratios.max().item() < 5.0}")

if __name__ == "__main__":
    test_ppo_ratio_computation()
