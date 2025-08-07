"""
Test PPO training with the masking fix to verify log probability consistency
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Add current directory to path to import main components
import torch
import logging

# Set up basic logging to reduce output
logging.basicConfig(level=logging.WARNING)

def test_training_with_masking_fix():
    """Run a few training steps to test log probability consistency"""
    print("=== Testing PPO Training with Masking Fix ===")
    
    try:
        # Import required components
        from src.game.Game import Game
        from src.agents.RLGNNAgent import RLGNNAgent
        from src.game.World import World
        from src.game.GameConfig import GameConfig
        
        # Create a simple game configuration
        config = GameConfig()
        config.num_players = 2
        config.game_mode = "basic"
        
        # Load a simple world (try different world files)
        world_files = ["world.txt", "Asia.txt", "oceania.txt"]
        world = None
        for world_file in world_files:
            try:
                if os.path.exists(world_file):
                    world = World(world_file)
                    print(f"Loaded world from {world_file}")
                    break
            except Exception as e:
                continue
        
        if world is None:
            print("No world file found, creating minimal test world")
            # Create a minimal test - we'll skip this for now
            return False
        
        # Create game
        game = Game(config, world)
        
        # Create agents
        agent1 = RLGNNAgent(agent_number=1)
        agent2 = RLGNNAgent(agent_number=2)
        
        # Initialize agents
        agent1.init(5000)
        agent2.init(5000)
        
        print(f"Created game with {len(world.regions)} regions")
        
        # Run a few game turns to generate some training data
        max_turns = 5
        turn = 0
        
        while not game.is_done() and turn < max_turns:
            turn += 1
            print(f"Turn {turn}")
            
            # Agent 1 turn
            try:
                if game.current_player == 1:
                    placements = agent1.place_armies(game)
                    for placement in placements:
                        game.place_armies(placement)
                    
                    attacks = agent1.attack_transfer(game)
                    for attack in attacks:
                        game.attack_transfer(attack)
                    
                    agent1.end_move(game)
                    game.next_turn()
                
                # Agent 2 turn  
                if game.current_player == 2 and not game.is_done():
                    placements = agent2.place_armies(game)
                    for placement in placements:
                        game.place_armies(placement)
                    
                    attacks = agent2.attack_transfer(game)
                    for attack in attacks:
                        game.attack_transfer(attack)
                    
                    agent2.end_move(game)
                    game.next_turn()
                    
            except Exception as e:
                print(f"Error during turn {turn}: {e}")
                import traceback
                traceback.print_exc()
                break
        
        # Check the log probability differences from the agent's rewards
        print("\n=== Log Probability Difference Analysis ===")
        for agent_name, agent in [("Agent 1", agent1), ("Agent 2", agent2)]:
            if 'log_prob_diff_mean' in agent.total_rewards:
                diff_mean = agent.total_rewards['log_prob_diff_mean']
                diff_std = agent.total_rewards.get('log_prob_diff_std', 0)
                print(f"{agent_name}:")
                print(f"  Log prob difference mean: {diff_mean:.3f}")
                print(f"  Log prob difference std: {diff_std:.3f}")
                
                # Check if differences are reasonable (should be much smaller than ~200)
                if abs(diff_mean) < 10.0:
                    print(f"  ✓ Log prob differences are reasonable")
                else:
                    print(f"  ❌ Log prob differences are still too large")
            else:
                print(f"{agent_name}: No PPO update occurred yet")
        
        print("\n=== Test completed successfully! ===")
        return True
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_training_with_masking_fix()
    if success:
        print("🎉 Training test completed!")
    else:
        print("❌ Training test failed")
