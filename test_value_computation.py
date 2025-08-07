#!/usr/bin/env python3
"""
Test to verify value computation correctness step by step
"""

import torch
import torch.nn.functional as f
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.agents.RLUtils.WarlightModel import WarlightPolicyNet
from src.agents.RLUtils.RLUtils import compute_gae, RolloutBuffer
from src.agents.RLUtils.PPOAgent import PPOAgent
from src.config.training_config import VerificationConfig


def test_value_computation():
    """Test value computation consistency"""
    print("=== TESTING VALUE COMPUTATION ===\n")
    
    # Create test setup
    device = torch.device('cpu')
    batch_size = 3
    num_nodes = 6
    node_feat_dim = 8
    embed_dim = 64
    
    # Create model
    model = WarlightPolicyNet(node_feat_dim, embed_dim).to(device)
    
    # Create dummy edge tensor for the model
    edge_list = [[i, i + 1] for i in range(num_nodes - 1)] + [[i + 1, i] for i in range(num_nodes - 1)]
    model.edge_tensor = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    # Create verification config with value checks enabled
    verification_config = VerificationConfig()
    verification_config.enabled = True
    verification_config.detailed_logging = True
    verification_config.verify_value_computation = True
    verification_config.verify_gae_computation = True
    
    # Create PPO agent with verification
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    ppo_agent = PPOAgent(model, optimizer, verification_config=verification_config)
    
    print("1. TESTING VALUE FUNCTION CONSISTENCY")
    print("-" * 50)
    
    # Test 1: Value function consistency across calls
    node_features = torch.randn(batch_size, num_nodes, node_feat_dim, device=device)
    
    with torch.no_grad():
        # Call value function multiple times with same input
        values1 = model.get_value(node_features)
        values2 = model.get_value(node_features)
        values3 = model.get_value(node_features)
        
        print(f"Values 1: {values1}")
        print(f"Values 2: {values2}")
        print(f"Values 3: {values3}")
        
        # Check consistency
        diff12 = (values1 - values2).abs().max()
        diff13 = (values1 - values3).abs().max()
        
        print(f"Max diff (call 1 vs 2): {diff12:.8f}")
        print(f"Max diff (call 1 vs 3): {diff13:.8f}")
        
        if diff12 < 1e-7 and diff13 < 1e-7:
            print("✅ Value function is deterministic")
        else:
            print("⚠️  WARNING: Value function is non-deterministic!")
    
    print("\n2. TESTING VALUE AGGREGATION METHODS")
    print("-" * 50)
    
    # Test 2: Compare different aggregation methods
    with torch.no_grad():
        # Process through GNN layers manually
        edge_tensor = model.edge_tensor.to(device)
        x = f.relu(model.gnn1(node_features, edge_tensor))
        node_embeddings = model.gnn2(x, edge_tensor)
        
        print(f"Node embeddings shape: {node_embeddings.shape}")
        
        # Test different aggregation methods
        mean_agg = node_embeddings.mean(dim=1)
        max_agg = node_embeddings.max(dim=1)[0] 
        sum_agg = node_embeddings.sum(dim=1)
        
        # Apply value head to each
        value_mean = model.value_head(mean_agg).squeeze(-1)
        value_max = model.value_head(max_agg).squeeze(-1)
        value_sum = model.value_head(sum_agg).squeeze(-1)
        
        print(f"Value (mean aggregation): {value_mean}")
        print(f"Value (max aggregation): {value_max}")
        print(f"Value (sum aggregation): {value_sum}")
        
        # Compare with model's get_value method
        value_model = model.get_value(node_features)
        mean_diff = (value_mean - value_model).abs().max()
        
        print(f"Difference from model get_value: {mean_diff:.8f}")
        if mean_diff < 1e-6:
            print("✅ Model uses mean aggregation correctly")
        else:
            print("⚠️  WARNING: Model aggregation differs from expected!")
    
    print("\n3. TESTING GAE COMPUTATION")
    print("-" * 50)
    
    # Test 3: GAE computation correctness
    T = 5  # sequence length
    gamma = 0.99
    lam = 0.95
    
    # Create dummy trajectory data
    rewards = torch.tensor([1.0, 0.5, -0.2, 0.8, 2.0])
    values = torch.tensor([2.1, 1.8, 1.5, 2.2, 1.9])
    last_value = torch.tensor(1.7)
    dones = torch.tensor([0, 0, 0, 0, 1])  # Last step is terminal
    
    print(f"Rewards: {rewards}")
    print(f"Values: {values}")
    print(f"Last value: {last_value}")
    print(f"Dones: {dones}")
    
    # Compute GAE
    advantages, returns = compute_gae(rewards, values, last_value, dones, gamma, lam)
    
    print(f"Computed advantages: {advantages}")
    print(f"Computed returns: {returns}")
    
    # Manual verification of first step
    V_extended = torch.cat([values, last_value.unsqueeze(0)])
    delta_0 = rewards[0] + gamma * V_extended[1] * (1 - dones[0]) - V_extended[0]
    print(f"Manual delta[0]: {delta_0:.6f}")
    print(f"Expected return[0]: {advantages[0] + values[0]:.6f}")
    print(f"Actual return[0]: {returns[0]:.6f}")
    
    print("\n4. TESTING VALUE COMPUTATION IN PPO UPDATE")
    print("-" * 50)
    
    # Test 4: Full PPO update with value verification
    buffer = RolloutBuffer()
    
    # Add dummy episodes to buffer
    for i in range(batch_size):
        # Simple dummy episode
        action_edges = torch.zeros(42, 2, dtype=torch.long)
        attacks = torch.tensor([[-1, -1, -1]])  # No attacks
        placements = torch.tensor([0, 1, 2])  # Place on first 3 regions
        placement_log_probs = torch.tensor([-1.0, -1.2, -1.1])
        attack_log_probs = torch.tensor([0.0])  # Dummy
        reward = torch.tensor(1.0 + i * 0.5)  # Varying rewards
        value = torch.tensor(2.0 + i * 0.3)   # Varying values
        done = 1 if i == batch_size - 1 else 0  # Last episode is done
        
        starting_features = torch.randn(num_nodes, node_feat_dim)
        post_features = torch.randn(num_nodes, node_feat_dim)
        
        buffer.add(
            action_edges, attacks, placements, placement_log_probs, attack_log_probs,
            reward, value, done, starting_features, post_features
        )
    
    # Create dummy agent
    class DummyAgent:
        def __init__(self, model):
            self.model = model
            self.total_rewards = {}
            self.game_number = 5
            
        def run_model(self, node_features, action_edges, action):
            if node_features.dim() == 2:
                node_features = node_features.unsqueeze(0)
            return self.model(node_features, action_edges, node_features[:, :, -1])
    
    dummy_agent = DummyAgent(model)
    
    # Run PPO update with verification
    try:
        last_value = torch.tensor(1.5)  # Terminal value
        ppo_agent.update(buffer, last_value, dummy_agent)
        print("✅ PPO update completed with value verification")
    except Exception as e:
        print(f"⚠️  PPO update failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n5. TESTING VALUE RANGES AND STABILITY")
    print("-" * 50)
    
    # Test 5: Value range and stability
    with torch.no_grad():
        # Test with different input ranges
        test_cases = [
            torch.randn(2, num_nodes, node_feat_dim) * 0.1,  # Small values
            torch.randn(2, num_nodes, node_feat_dim) * 1.0,  # Normal values  
            torch.randn(2, num_nodes, node_feat_dim) * 5.0,  # Large values
        ]
        
        for i, test_input in enumerate(test_cases):
            values = model.get_value(test_input)
            print(f"Test case {i+1}: input_std={test_input.std():.3f}, values={values}, range=[{values.min():.3f}, {values.max():.3f}]")
            
            # Check for numerical issues
            if torch.isnan(values).any():
                print(f"  ⚠️  NaN detected in values!")
            if torch.isinf(values).any():
                print(f"  ⚠️  Inf detected in values!")
    
    print("\n=== VALUE COMPUTATION TEST COMPLETE ===")


if __name__ == "__main__":
    test_value_computation()
