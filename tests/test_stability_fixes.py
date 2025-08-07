"""
Test the numerical stability fixes for log probability calculations
"""
import torch
import torch.nn.functional as f

def test_numerical_stability_fixes():
    """Test the numerical stability fixes"""
    print("=== Testing Numerical Stability Fixes ===")
    
    # Test 1: Placement log probabilities with masked regions
    print("\n1. Testing placement log probabilities:")
    placement_logits = torch.randn(10)
    placement_logits[0:3] = float('-inf')  # Mask some regions
    
    # Old method (wrong)
    placement_probs = placement_logits.softmax(dim=0)
    old_log_probs = torch.log(placement_probs + 1e-8)
    
    # New method (correct)
    new_log_probs = f.log_softmax(placement_logits, dim=0)
    
    # Test with actual selections
    selected_regions = torch.tensor([4, 5, 6, 4])  # Some placements
    
    print(f"Placement logits: {placement_logits}")
    print(f"Selected regions: {selected_regions.tolist()}")
    
    old_selected_log_probs = [old_log_probs[r].item() for r in selected_regions]
    new_selected_log_probs = [new_log_probs[r].item() for r in selected_regions]
    
    print(f"Old method log probs: {old_selected_log_probs}")
    print(f"New method log probs: {new_selected_log_probs}")
    
    differences = [abs(o - n) for o, n in zip(old_selected_log_probs, new_selected_log_probs)]
    print(f"Differences: {differences}")
    print(f"Max difference: {max(differences):.2e}")
    
    # Test 2: Army log probabilities
    print("\n2. Testing army log probabilities:")
    army_logits = torch.randn(5)
    
    # Old method (wrong)
    army_probs = f.softmax(army_logits, dim=0)
    selected_army = 2
    old_army_log_prob = torch.log(army_probs[selected_army] + 1e-8).item()
    
    # New method (correct)
    new_army_log_prob = f.log_softmax(army_logits, dim=0)[selected_army].item()
    
    print(f"Army logits: {army_logits}")
    print(f"Selected army: {selected_army}")
    print(f"Old method log prob: {old_army_log_prob:.6f}")
    print(f"New method log prob: {new_army_log_prob:.6f}")
    print(f"Difference: {abs(old_army_log_prob - new_army_log_prob):.2e}")
    
    # Test 3: Edge selection log probabilities  
    print("\n3. Testing edge selection log probabilities:")
    candidate_logits = torch.randn(8)
    
    # Old method (wrong)
    edge_probs = f.softmax(candidate_logits, dim=0)
    selected_edge = 3
    old_edge_log_prob = torch.log(edge_probs[selected_edge] + 1e-8).item()
    
    # New method (correct)
    new_edge_log_prob = f.log_softmax(candidate_logits, dim=0)[selected_edge].item()
    
    print(f"Candidate logits: {candidate_logits}")
    print(f"Selected edge: {selected_edge}")
    print(f"Old method log prob: {old_edge_log_prob:.6f}")
    print(f"New method log prob: {new_edge_log_prob:.6f}")
    print(f"Difference: {abs(old_edge_log_prob - new_edge_log_prob):.2e}")
    
    # Test 4: Extreme case with very large/small logits
    print("\n4. Testing extreme logits:")
    extreme_logits = torch.tensor([100.0, 200.0, 300.0, -100.0, -200.0])
    
    # Old method
    extreme_probs = f.softmax(extreme_logits, dim=0)
    old_extreme_log_probs = torch.log(extreme_probs + 1e-8)
    
    # New method  
    new_extreme_log_probs = f.log_softmax(extreme_logits, dim=0)
    
    print(f"Extreme logits: {extreme_logits}")
    print(f"Old method log probs: {old_extreme_log_probs}")
    print(f"New method log probs: {new_extreme_log_probs}")
    print(f"Max difference: {torch.abs(old_extreme_log_probs - new_extreme_log_probs).max():.2e}")

if __name__ == "__main__":
    test_numerical_stability_fixes()
