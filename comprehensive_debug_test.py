"""
Comprehensive test to identify remaining log probability mismatch sources
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
import torch.nn.functional as f
from src.agents.RLUtils.RLUtils import compute_individual_log_probs

def test_edge_case_scenarios():
    """Test various edge cases that might cause log probability mismatches"""
    print("=== Testing Edge Case Scenarios ===")
    
    device = torch.device('cpu')
    
    # Test 1: Very small probabilities (numerical stability)
    print("\n1. Testing numerical stability with small probabilities:")
    logits = torch.tensor([-100.0, -50.0, -10.0, 0.0, 10.0])
    probs = f.softmax(logits, dim=0)
    log_probs1 = f.log_softmax(logits, dim=0)
    log_probs2 = torch.log(probs + 1e-8)
    print(f"Min probability: {probs.min():.2e}")
    print(f"Log softmax vs manual log: max diff = {torch.abs(log_probs1 - log_probs2).max():.2e}")
    
    # Test 2: Large logit values
    print("\n2. Testing large logit values:")
    logits = torch.tensor([100.0, 200.0, 300.0, 400.0, 500.0])
    probs = f.softmax(logits, dim=0)
    log_probs1 = f.log_softmax(logits, dim=0)
    log_probs2 = torch.log(probs + 1e-8)
    print(f"Max probability: {probs.max():.6f}")
    print(f"Log softmax vs manual log: max diff = {torch.abs(log_probs1 - log_probs2).max():.2e}")
    
    # Test 3: Near-zero gradients (potential gradient flow issues)
    print("\n3. Testing gradient flow:")
    logits = torch.tensor([1.0, 1.0001, 1.0002, 1.0003], requires_grad=True)
    probs = f.softmax(logits, dim=0)
    log_probs = f.log_softmax(logits, dim=0)
    loss = log_probs.sum()
    loss.backward()
    print(f"Logits gradients: {logits.grad}")
    print(f"Are gradients very small? {(torch.abs(logits.grad) < 1e-6).any()}")

def test_action_selection_consistency():
    """Test consistency between different action selection methods"""
    print("\n=== Testing Action Selection Consistency ===")
    
    # Create test data
    torch.manual_seed(42)  # For reproducibility
    attack_logits = torch.randn(10)
    army_logits = torch.randn(10, 5)
    
    print("\n1. Testing attack edge selection:")
    # Method 1: Direct softmax + multinomial
    probs1 = f.softmax(attack_logits, dim=0)
    sample1 = torch.multinomial(probs1, 1)
    log_prob1 = f.log_softmax(attack_logits, dim=0)[sample1]
    
    # Method 2: Using categorical distribution
    dist = torch.distributions.Categorical(logits=attack_logits)
    sample2 = dist.sample()
    log_prob2 = dist.log_prob(sample2)
    
    print(f"Method 1 sample: {sample1.item()}, log_prob: {log_prob1.item():.6f}")
    print(f"Method 2 sample: {sample2.item()}, log_prob: {log_prob2.item():.6f}")
    
    # They should give same results for same sample
    if sample1.item() == sample2.item():
        print(f"Log prob difference for same sample: {abs(log_prob1.item() - log_prob2.item()):.2e}")
    
    print("\n2. Testing army count selection:")
    # For specific edge, test army selection
    edge_idx = 3
    available_armies = 4
    army_logit_slice = army_logits[edge_idx][:available_armies]
    
    # Method 1: Direct softmax + multinomial
    army_probs1 = f.softmax(army_logit_slice, dim=0)
    army_sample1 = torch.multinomial(army_probs1, 1)
    army_log_prob1 = f.log_softmax(army_logit_slice, dim=0)[army_sample1]
    
    # Method 2: Using categorical distribution
    army_dist = torch.distributions.Categorical(probs=army_probs1)
    army_sample2 = army_dist.sample()
    army_log_prob2 = army_dist.log_prob(army_sample2)
    
    print(f"Army method 1 sample: {army_sample1.item()}, log_prob: {army_log_prob1.item():.6f}")
    print(f"Army method 2 sample: {army_sample2.item()}, log_prob: {army_log_prob2.item():.6f}")

def test_compute_individual_log_probs():
    """Test the compute_individual_log_probs function specifically"""
    print("\n=== Testing compute_individual_log_probs Function ===")
    
    device = torch.device('cpu')
    
    # Create test data matching real usage
    batch_size = 1
    num_nodes = 42
    max_attacks = 3
    max_placements = 5
    
    # Create realistic test data
    placement_logits = torch.randn(batch_size, num_nodes)
    attack_logits = torch.randn(batch_size, 42) 
    army_logits = torch.randn(batch_size, 42, 10)
    
    # Create some test actions
    placements = torch.tensor([[5, 10, 15, -1, -1]])  # Some valid, some padding
    attacks = torch.tensor([[[5, 10, 2], [15, 20, 3], [-1, -1, -1]]])  # Some valid, some padding
    action_edges = torch.tensor([[[i, (i+1) % num_nodes] for i in range(42)]]).expand(batch_size, -1, -1)
    
    # Test the function
    placement_log_probs, attack_log_probs = compute_individual_log_probs(
        attacks, attack_logits, army_logits, placements, placement_logits, action_edges
    )
    
    print(f"Placement log probs shape: {placement_log_probs.shape}")
    print(f"Attack log probs shape: {attack_log_probs.shape}")
    print(f"Placement log probs: {placement_log_probs}")
    print(f"Attack log probs: {attack_log_probs}")
    
    # Check for NaN or inf values
    if torch.isnan(placement_log_probs).any():
        print("WARNING: NaN in placement log probs!")
    if torch.isnan(attack_log_probs).any():
        print("WARNING: NaN in attack log probs!")
    if torch.isinf(placement_log_probs).any():
        print("WARNING: Inf in placement log probs!")
    if torch.isinf(attack_log_probs).any():
        print("WARNING: Inf in attack log probs!")

def test_manual_computation():
    """Manually compute log probabilities and compare with automatic computation"""
    print("\n=== Testing Manual vs Automatic Computation ===")
    
    # Simple test case
    attack_logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    army_logits = torch.tensor([[0.5, 1.0, 1.5], [1.0, 2.0, 3.0], [1.5, 2.5, 3.5], [2.0, 3.0, 4.0], [2.5, 3.5, 4.5]])
    
    # Simulate selecting edge 2 with 1 army (0-indexed)
    selected_edge = 2
    selected_army = 1
    
    # Manual computation
    edge_log_probs = f.log_softmax(attack_logits, dim=0)
    edge_log_prob = edge_log_probs[selected_edge]
    
    army_log_probs = f.log_softmax(army_logits[selected_edge], dim=0)
    army_log_prob = army_log_probs[selected_army]
    
    total_manual = edge_log_prob + army_log_prob
    
    print(f"Manual computation:")
    print(f"  Edge {selected_edge} log prob: {edge_log_prob.item():.6f}")
    print(f"  Army {selected_army} log prob: {army_log_prob.item():.6f}")
    print(f"  Total log prob: {total_manual.item():.6f}")
    
    # Now test with compute_individual_log_probs
    attacks = torch.tensor([[[0, 1, 2]]])  # src=0, tgt=1, armies=2 (1-indexed)
    attack_logits_batch = attack_logits.unsqueeze(0)
    army_logits_batch = army_logits.unsqueeze(0)
    placement_logits = torch.randn(1, 10)
    placements = torch.tensor([[-1]])  # No placements
    action_edges = torch.tensor([[[0, 1], [0, 2], [1, 2], [2, 3], [3, 4]]])  # edge 0 connects 0->1
    
    placement_log_probs, attack_log_probs = compute_individual_log_probs(
        attacks, attack_logits_batch, army_logits_batch, placements, placement_logits, action_edges
    )
    
    print(f"Function computation:")
    print(f"  Attack log prob: {attack_log_probs[0].item():.6f}")
    print(f"  Difference: {abs(total_manual.item() - attack_log_probs[0].item()):.2e}")

if __name__ == "__main__":
    test_edge_case_scenarios()
    test_action_selection_consistency()
    test_compute_individual_log_probs()
    test_manual_computation()
    print("\n=== All tests completed ===")
