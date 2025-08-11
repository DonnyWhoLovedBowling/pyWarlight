#!/usr/bin/env python3

import torch
import numpy as np
from src.agents.RLGNNAgent import RLGNNAgent
from src.game.Phase import Phase

def test_batch_consistency():
    """Test if model produces identical outputs for single vs batch inference"""
    print("=== BATCH CONSISTENCY TEST ===")
    
    # Create agent
    agent = RLGNNAgent()
    
    # Create a simple edge tensor (World map has 42 territories)
    # This creates a simple ring topology for testing
    edge_list = []
    num_nodes = 42
    for i in range(num_nodes):
        edge_list.append([i, (i + 1) % num_nodes])  # Ring topology
    edge_tensor = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    agent.model.edge_tensor = edge_tensor
    
    # Set model to eval mode for deterministic behavior
    agent.model.eval()
    
    # Create dummy data
    batch_size = 2
    num_nodes = 42
    num_edges = 166
    features_dim = 7
    
    # Single batch with multiple samples
    node_features_batch = torch.randn(batch_size, num_nodes, features_dim)
    action_edges_batch = torch.randint(0, num_nodes, (batch_size, num_edges, 2))
    
    print(f"Input shapes: node_features={node_features_batch.shape}, action_edges={action_edges_batch.shape}")
    
    # Test 1: Process entire batch
    print("\n--- Test 1: Full batch processing ---")
    with torch.no_grad():
        batch_placement, batch_attack, batch_army = agent.run_model(
            node_features=node_features_batch,
            action_edges=action_edges_batch,
            action=Phase.PLACE_ARMIES
        )
    
    print(f"Batch results shapes: placement={batch_placement.shape}, attack={batch_attack.shape}, army={batch_army.shape}")
    
    # Test 2: Process samples individually
    print("\n--- Test 2: Individual sample processing ---")
    individual_placement = []
    individual_attack = []
    individual_army = []
    
    for i in range(batch_size):
        with torch.no_grad():
            single_placement, single_attack, single_army = agent.run_model(
                node_features=node_features_batch[i:i+1],  # Keep batch dimension
                action_edges=action_edges_batch[i:i+1],    # Keep batch dimension  
                action=Phase.PLACE_ARMIES
            )
            individual_placement.append(single_placement.squeeze(0))
            individual_attack.append(single_attack.squeeze(0) if single_attack.numel() > 0 else torch.tensor([]))
            individual_army.append(single_army.squeeze(0) if single_army.numel() > 0 else torch.tensor([]))
    
    # Compare results
    print("\n--- Comparison ---")
    for i in range(batch_size):
        batch_sample = batch_placement[i]
        individual_sample = individual_placement[i]
        
        diff = torch.abs(batch_sample - individual_sample).max().item()
        print(f"Sample {i} placement diff: {diff:.8f}")
        
        if diff > 1e-6:
            print(f"  ERROR: Significant difference detected!")
            print(f"  Batch sample: {batch_sample[:5].detach().cpu().numpy()}")
            print(f"  Individual sample: {individual_sample[:5].detach().cpu().numpy()}")
        else:
            print(f"  ✓ Samples match within tolerance")

    # Test 3: Test with attack phase
    print("\n--- Test 3: Attack phase ---")
    with torch.no_grad():
        batch_placement, batch_attack, batch_army = agent.run_model(
            node_features=node_features_batch,
            action_edges=action_edges_batch,
            action=Phase.ATTACK_TRANSFER
        )
    
    individual_attack = []
    individual_army = []
    
    for i in range(batch_size):
        with torch.no_grad():
            _, single_attack, single_army = agent.run_model(
                node_features=node_features_batch[i:i+1],
                action_edges=action_edges_batch[i:i+1],
                action=Phase.ATTACK_TRANSFER
            )
            individual_attack.append(single_attack.squeeze(0))
            individual_army.append(single_army.squeeze(0))
    
    for i in range(batch_size):
        attack_diff = torch.abs(batch_attack[i] - individual_attack[i]).max().item()
        army_diff = torch.abs(batch_army[i] - individual_army[i]).max().item()
        
        print(f"Sample {i} attack diff: {attack_diff:.8f}, army diff: {army_diff:.8f}")
        
        if attack_diff > 1e-6 or army_diff > 1e-6:
            print(f"  ERROR: Attack/Army differences detected!")
        else:
            print(f"  ✓ Attack/Army samples match")

if __name__ == "__main__":
    test_batch_consistency()
