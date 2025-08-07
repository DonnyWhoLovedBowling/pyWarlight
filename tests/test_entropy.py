import torch
import torch.nn.functional as f
import sys
sys.path.append('src')

# Test the entropy computation with sample placement logits
def test_entropy():
    # Create sample placement logits with masking pattern
    batch_size = 2
    num_regions = 5
    
    # Sample 1: Player owns regions 0, 2 - others are masked to -inf
    logits1 = torch.tensor([-1.5, -float('inf'), -2.0, -float('inf'), -float('inf')])
    
    # Sample 2: Player owns regions 1, 3, 4 - others are masked to -inf  
    logits2 = torch.tensor([-float('inf'), -1.0, -float('inf'), -2.5, -1.8])
    
    placement_logits = torch.stack([logits1, logits2])
    print("Original logits:")
    print(placement_logits)
    
    # Test the entropy computation
    from src.agents.RLUtils.RLUtils import compute_entropy
    
    # Dummy edge and army logits for the function
    edge_logits = torch.zeros(batch_size, 10)  
    army_logits = torch.zeros(batch_size, 5)
    
    placement_entropy, edge_entropy, army_entropy = compute_entropy(placement_logits, edge_logits, army_logits)
    
    print(f"\nPlacement entropy: {placement_entropy}")
    print(f"Edge entropy: {edge_entropy}")
    print(f"Army entropy: {army_entropy}")
    
    # Manual verification for Sample 1: owns regions 0, 2
    print("\nManual verification for Sample 1:")
    probs1 = f.softmax(logits1[logits1 != -float('inf')], dim=0)
    print(f"Probabilities over owned regions: {probs1}")
    manual_entropy1 = -(probs1 * torch.log(probs1)).sum()
    print(f"Manual entropy: {manual_entropy1}")
    
    # Manual verification for Sample 2: owns regions 1, 3, 4
    print("\nManual verification for Sample 2:")
    valid_logits2 = torch.tensor([-1.0, -2.5, -1.8])  # regions 1, 3, 4
    probs2 = f.softmax(valid_logits2, dim=0)
    print(f"Probabilities over owned regions: {probs2}")
    manual_entropy2 = -(probs2 * torch.log(probs2)).sum()
    print(f"Manual entropy: {manual_entropy2}")
    
    expected_total = manual_entropy1 + manual_entropy2
    print(f"\nExpected total entropy: {expected_total}")
    print(f"Computed total entropy: {placement_entropy}")

if __name__ == "__main__":
    test_entropy()
