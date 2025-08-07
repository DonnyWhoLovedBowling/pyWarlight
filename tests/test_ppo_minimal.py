#!/usr/bin/env python3
"""
Minimal test to reproduce the log probability issue in PPO updates.
"""

import torch
import torch.nn.functional as f
from src.agents.RLUtils.RLUtils import RolloutBuffer, compute_individual_log_probs
from src.agents.RLUtils.WarlightModel import WarlightPolicyNet
from src.agents.RLUtils.PPOAgent import PPOAgent

def test_ppo_log_prob_differences():
    """Test if large log probability differences occur in PPO updates."""
    print("=== TESTING PPO LOG PROBABILITY DIFFERENCES ===\n")
    
    # Create a simple scenario
    batch_size = 2
    num_nodes = 6
    embed_dim = 64
    device = torch.device('cpu')  # Use CPU to avoid device issues
    
    # Create model and PPO agent
    model = WarlightPolicyNet(8, embed_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-6)
    ppo_agent = PPOAgent(model, optimizer, gamma=0.99, lam=0.95, clip_eps=0.2)
    
    # Create dummy data
    node_features = torch.randn(batch_size, num_nodes, 8, device=device)
    action_edges = torch.full((batch_size, 42, 2), -1, dtype=torch.long, device=device)
    action_edges[0, 0] = torch.tensor([0, 1])
    action_edges[0, 1] = torch.tensor([1, 2])
    action_edges[1, 0] = torch.tensor([2, 3])
    
    # Set up edge tensor for the model
    edge_list = [[i, i + 1] for i in range(num_nodes - 1)] + [[i + 1, i] for i in range(num_nodes - 1)]
    model.edge_tensor = torch.tensor(edge_list, dtype=torch.long, device=device).t().contiguous()
    
    # Create rollout buffer with sample data
    buffer = RolloutBuffer()
    
    # Generate some sample data to add to buffer
    for i in range(batch_size):
        # Sample placement and attack actions
        placement_actions = [0, 1, 2]  # Place armies on nodes 0, 1, 2
        attack_actions = [[0, 1, 2]]  # Attack from 0 to 1 with 2 armies
        
        # Generate model outputs for this sample
        single_node_features = node_features[i:i+1]
        single_action_edges = action_edges[i:i+1]
        
        with torch.no_grad():
            placement_logits, attack_logits, army_logits = model(
                single_node_features, single_action_edges, single_node_features[:, :, -1]
            )
        
        # Compute log probabilities for the actions taken
        placement_tensor = torch.tensor(placement_actions).unsqueeze(0)
        attack_tensor = torch.tensor(attack_actions).unsqueeze(0)
        
        placement_log_probs, attack_log_probs = compute_individual_log_probs(
            attack_tensor, attack_logits, army_logits, placement_tensor,
            placement_logits, single_action_edges
        )
        
        print(f"Sample {i}:")
        print(f"  Placement log probs: {placement_log_probs}")
        print(f"  Attack log probs: {attack_log_probs}")
        
        # Add to buffer
        buffer.add(
            single_action_edges.squeeze(0),
            attack_actions,
            placement_actions,
            placement_log_probs.squeeze(0),
            attack_log_probs.squeeze(0),
            reward=1.0,  # Dummy reward
            value=torch.tensor(0.5),  # Dummy value
            done=0,
            starting_node_features=single_node_features.squeeze(0),
            post_placement_node_features=single_node_features.squeeze(0)
        )
    
    print(f"\nBuffer contents:")
    print(f"  Placements: {buffer.get_placements()}")
    print(f"  Attacks: {buffer.get_attacks()}")
    print(f"  Old placement log probs: {buffer.get_placement_log_probs()}")
    print(f"  Old attack log probs: {buffer.get_attack_log_probs()}")
    
    # Create a dummy agent class that has the required methods
    class DummyAgent:
        def __init__(self, model, action_edges):
            self.model = model
            self.action_edges_batch = action_edges
            self.total_rewards = {}  # Required by PPO agent
            self.game_number = 2  # Set > 1 to enable advantage normalization
            
        def run_model(self, node_features, action_edges, action):
            return self.model(node_features, action_edges, node_features[:, :, -1])
    
    dummy_agent = DummyAgent(model, action_edges)
    
    print(f"\n=== RUNNING PPO UPDATE ===")
    try:
        next_value = torch.tensor(0.0)  # Dummy next value
        ppo_agent.update(buffer, next_value, dummy_agent)
        print("PPO update completed successfully")
    except Exception as e:
        print(f"Error during PPO update: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ppo_log_prob_differences()
