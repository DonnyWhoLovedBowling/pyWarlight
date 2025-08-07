#!/usr/bin/env python3
"""
Test the actual RL agent to see if log probability differences persist in real training.
"""

import torch
import torch.nn.functional as f
from src.agents.RLGNNAgent import RLGNNAgent
from src.agents.RLUtils.RLUtils import compute_individual_log_probs
from src.game.Game import Game
from src.game.World import World

def test_actual_agent_log_probs():
    """Test log probabilities in the actual agent during a simulated turn."""
    print("=== TESTING ACTUAL AGENT LOG PROBABILITIES ===\n")
    
    # Create a simple world and config
    world = World("world")  # Use default world map
    from src.game.GameConfig import GameConfig
    config = GameConfig()
    config.num_players = 2  # Set number of players
    config.extra_armies = [0, 0]  # No extra armies for players
    
    # Create RL agent with default parameters
    agent = RLGNNAgent()
    agent.debug_log_probs = True  # Enable debug output
    
    # Create a simple game state
    game = Game(config, world)
    # Don't call initialize - it may try to set up starting regions for all players
    
    print("1. TESTING PLACEMENT ACTION")
    print("-" * 50)
    
    # Simulate placement phase
    try:
        placement_actions = agent.place_armies(game)
        print(f"Placement actions: {[str(p) for p in placement_actions]}")
        
        # Check if actual placement log probs were captured
        if hasattr(agent, 'actual_placement_log_probs'):
            print(f"Actual placement log probs: {agent.actual_placement_log_probs}")
        else:
            print("No actual placement log probs captured")
            
    except Exception as e:
        print(f"Error in placement: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n2. TESTING ATTACK ACTION")
    print("-" * 50)
    
    # Apply placements to game state
    for action in placement_actions:
        game.place_armies(action)
    
    # Simulate attack phase
    try:
        attack_actions = agent.get_attack_transfer_commands(game)
        print(f"Attack actions: {[str(a) for a in attack_actions]}")
        
        # Check if actual attack log probs were captured
        if hasattr(agent, 'actual_attack_log_probs'):
            print(f"Actual attack log probs: {agent.actual_attack_log_probs}")
        else:
            print("No actual attack log probs captured")
            
    except Exception as e:
        print(f"Error in attack: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n3. TESTING LOG PROB COMPUTATION")
    print("-" * 50)
    
    # Get the stored data for comparison
    if hasattr(agent, 'placement_logits') and hasattr(agent, 'attack_logits'):
        # Get actual actions taken
        placements = agent.get_placements()
        attacks = agent.get_attacks()
        
        print(f"Placements taken: {placements}")
        print(f"Attacks taken: {attacks}")
        
        if len(attacks) > 0 and len(placements) > 0:
            # Convert to tensors
            attacks_tensor = torch.tensor(attacks, dtype=torch.long).unsqueeze(0)  # Add batch dim
            placements_tensor = torch.tensor(placements, dtype=torch.long).unsqueeze(0)  # Add batch dim
            
            # Get model outputs (add batch dimensions)
            placement_logits = agent.placement_logits.unsqueeze(0)
            attack_logits = agent.attack_logits.unsqueeze(0)
            army_logits = agent.army_logits.unsqueeze(0)
            action_edges = agent.action_edges.unsqueeze(0)
            
            print(f"Placement logits shape: {placement_logits.shape}")
            print(f"Attack logits shape: {attack_logits.shape}")
            print(f"Army logits shape: {army_logits.shape}")
            print(f"Action edges shape: {action_edges.shape}")
            print(f"Attacks tensor shape: {attacks_tensor.shape}")
            print(f"Placements tensor shape: {placements_tensor.shape}")
            
            # Compute log probs using the function
            try:
                computed_placement_log_probs, computed_attack_log_probs = compute_individual_log_probs(
                    attacks_tensor, attack_logits, army_logits, placements_tensor,
                    placement_logits, action_edges
                )
                
                print(f"Computed placement log probs: {computed_placement_log_probs}")
                print(f"Computed attack log probs: {computed_attack_log_probs}")
                
                # Compare with actual if available
                if hasattr(agent, 'actual_placement_log_probs'):
                    actual_placement = agent.actual_placement_log_probs
                    computed_placement = computed_placement_log_probs.squeeze()
                    
                    print(f"\nPLACEMENT COMPARISON:")
                    print(f"Actual: {actual_placement}")
                    print(f"Computed: {computed_placement}")
                    
                    if len(actual_placement) > 0 and len(computed_placement) > 0:
                        min_len = min(len(actual_placement), len(computed_placement))
                        differences = []
                        for i in range(min_len):
                            diff = abs(actual_placement[i] - computed_placement[i])
                            differences.append(diff)
                            print(f"  Action {i}: Actual={actual_placement[i]:.6f}, Computed={computed_placement[i]:.6f}, Diff={diff:.6f}")
                        
                        max_diff = max(differences) if differences else 0
                        print(f"Max placement difference: {max_diff:.6f}")
                
                if hasattr(agent, 'actual_attack_log_probs'):
                    actual_attack = agent.actual_attack_log_probs
                    computed_attack = computed_attack_log_probs.squeeze()
                    
                    print(f"\nATTACK COMPARISON:")
                    print(f"Actual: {actual_attack}")
                    print(f"Computed: {computed_attack}")
                    
                    if len(actual_attack) > 0 and len(computed_attack) > 0:
                        min_len = min(len(actual_attack), len(computed_attack))
                        differences = []
                        for i in range(min_len):
                            diff = abs(actual_attack[i] - computed_attack[i])
                            differences.append(diff)
                            print(f"  Action {i}: Actual={actual_attack[i]:.6f}, Computed={computed_attack[i]:.6f}, Diff={diff:.6f}")
                        
                        max_diff = max(differences) if differences else 0
                        print(f"Max attack difference: {max_diff:.6f}")
                        
                        if max_diff > 1.0:
                            print(f"⚠️  LARGE DIFFERENCE DETECTED: {max_diff:.6f}")
                        else:
                            print(f"✅ Attack differences within reasonable range")
                
            except Exception as e:
                print(f"Error computing log probs: {e}")
                import traceback
                traceback.print_exc()
    else:
        print("Missing logits data for comparison")

if __name__ == "__main__":
    test_actual_agent_log_probs()
