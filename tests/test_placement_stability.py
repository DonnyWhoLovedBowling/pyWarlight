"""
Test numerical stability issue with placement log probabilities
"""
import torch
import torch.nn.functional as f

# Test the numerical stability issue
print("=== Testing Placement Log Probability Stability ===")

# Create logits with some masked regions (set to -inf)
placement_logits = torch.randn(42)
placement_logits[0:10] = float('-inf')  # Mask out regions not owned

print(f"Placement logits min/max: {placement_logits.min():.3f}/{placement_logits.max():.3f}")

# Method 1: What the code currently does (WRONG)
placement_probs = placement_logits.softmax(dim=0)
placement_log_probs_wrong = torch.log(placement_probs + 1e-8)

# Method 2: Correct way
placement_log_probs_correct = f.log_softmax(placement_logits, dim=0)

print(f"Wrong method log probs min/max: {placement_log_probs_wrong.min():.3f}/{placement_log_probs_wrong.max():.3f}")
print(f"Correct method log probs min/max: {placement_log_probs_correct.min():.3f}/{placement_log_probs_correct.max():.3f}")

# Test with extreme case
extreme_logits = torch.tensor([100.0, 200.0, 300.0, -float('inf'), -float('inf')])
extreme_probs = extreme_logits.softmax(dim=0)
extreme_log_probs_wrong = torch.log(extreme_probs + 1e-8)
extreme_log_probs_correct = f.log_softmax(extreme_logits, dim=0)

print(f"\nExtreme case:")
print(f"Logits: {extreme_logits}")
print(f"Probs: {extreme_probs}")
print(f"Wrong log probs: {extreme_log_probs_wrong}")
print(f"Correct log probs: {extreme_log_probs_correct}")
print(f"Difference: {torch.abs(extreme_log_probs_wrong - extreme_log_probs_correct)}")

# Simulate actual selection
selected_regions = torch.tensor([1, 2, 1, 2, 0])  # Some placements
wrong_log_probs = []
correct_log_probs = []

for region in selected_regions:
    wrong_log_probs.append(placement_log_probs_wrong[region].item())
    correct_log_probs.append(placement_log_probs_correct[region].item())

print(f"\nFor selected regions {selected_regions.tolist()}:")
print(f"Wrong method: {wrong_log_probs}")
print(f"Correct method: {correct_log_probs}")
print(f"Differences: {[abs(w-c) for w, c in zip(wrong_log_probs, correct_log_probs)]}")
