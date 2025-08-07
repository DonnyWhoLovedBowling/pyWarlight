"""
Test the masking fix for placement log probabilities
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
import torch.nn.functional as f
from src.agents.RLUtils.RLUtils import apply_placement_mask

def test_masking_fix():
    """Test that masking produces consistent results"""
    print("=== Testing Masking Fix ===")
    
    # Create test data
    batch_size = 2
    num_regions = 10
    
    # Raw placement logits
    placement_logits = torch.randn(batch_size, num_regions)
    print(f"Raw logits: {placement_logits}")
    
    # Owned regions for each batch
    owned_regions_list = [
        torch.tensor([2, 5, 7, 8]),  # Batch 0 owns regions 2, 5, 7, 8
        torch.tensor([1, 3, 6, 9])   # Batch 1 owns regions 1, 3, 6, 9
    ]
    
    # Apply masking
    masked_logits = apply_placement_mask(placement_logits, owned_regions_list, num_regions)
    print(f"Masked logits: {masked_logits}")
    
    # Test that non-owned regions are indeed -inf
    for batch_idx, owned_regions in enumerate(owned_regions_list):
        all_regions = set(range(num_regions))
        not_owned = list(all_regions.difference(set(owned_regions.tolist())))
        
        print(f"Batch {batch_idx}:")
        print(f"  Owned regions: {owned_regions.tolist()}")
        print(f"  Not owned regions: {not_owned}")
        
        # Check that not owned regions are -inf
        for region in not_owned:
            is_neg_inf = torch.isinf(masked_logits[batch_idx, region]) and masked_logits[batch_idx, region] < 0
            print(f"  Region {region} is -inf: {is_neg_inf}")
        
        # Check that owned regions are unchanged
        for region in owned_regions:
            is_unchanged = torch.allclose(masked_logits[batch_idx, region], placement_logits[batch_idx, region])
            print(f"  Region {region} unchanged: {is_unchanged}")
    
    # Test log probability computation
    print("\n=== Testing Log Probability Computation ===")
    
    # Compute log probabilities from masked logits
    log_probs = f.log_softmax(masked_logits, dim=1)
    print(f"Log probabilities: {log_probs}")
    
    # Test sampling and log probability extraction
    for batch_idx in range(batch_size):
        batch_logits = masked_logits[batch_idx]
        batch_log_probs = log_probs[batch_idx]
        
        # Only owned regions should have finite log probabilities
        owned_regions = owned_regions_list[batch_idx]
        
        print(f"\nBatch {batch_idx}:")
        for region in owned_regions:
            print(f"  Region {region}: logit={batch_logits[region]:.3f}, log_prob={batch_log_probs[region]:.3f}")
        
        # Test that probabilities sum to 1 (only for owned regions)
        probs = f.softmax(batch_logits, dim=0)
        owned_probs_sum = probs[owned_regions].sum()
        print(f"  Sum of owned region probs: {owned_probs_sum:.6f}")

if __name__ == "__main__":
    test_masking_fix()
