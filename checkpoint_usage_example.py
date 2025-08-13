#!/usr/bin/env python3
"""
Example of how to use the new comprehensive checkpoint system

This demonstrates:
1. Creating a config with checkpoint settings
2. Setting up automatic checkpoint resume
3. Configuring what to load from checkpoints
"""

import sys
sys.path.append('.')

from src.config.training_config import ConfigFactory

def create_checkpoint_resume_config():
    """Create a config that automatically resumes from the latest checkpoint"""
    
    print("🔧 Creating checkpoint resume configuration...")
    
    # Start with a base config
    config = ConfigFactory.create('sage_model_decisive')
    
    # Configure automatic checkpoint resume
    config.logging.auto_resume_latest = True
    config.logging.resume_experiment_name = "sage_decisive_training"  # Resume from this experiment
    
    # Configure what to load from checkpoints
    config.logging.load_model_state = True
    config.logging.load_optimizer_state = True
    config.logging.load_reward_normalizer = True  # Critical for continued training!
    config.logging.load_game_number = True
    config.logging.load_stat_trackers = True
    config.logging.load_training_state = True
    
    # Configure checkpoint saving
    config.logging.checkpoint_every_n_episodes = 50  # Save more frequently
    config.logging.keep_last_n_checkpoints = 10  # Keep more checkpoints
    
    # Set experiment name (checkpoints will be saved under this name)
    config.logging.experiment_name = "sage_decisive_training"
    
    print("✅ Checkpoint resume config created!")
    print("\n📋 Configuration Summary:")
    print(f"   Experiment name: {config.logging.experiment_name}")
    print(f"   Auto resume: {config.logging.auto_resume_latest}")
    print(f"   Resume from experiment: {config.logging.resume_experiment_name}")
    print(f"   Save frequency: every {config.logging.checkpoint_every_n_episodes} games")
    print(f"   Keep checkpoints: {config.logging.keep_last_n_checkpoints}")
    print(f"   Load reward normalizer: {config.logging.load_reward_normalizer}")
    
    return config

def create_specific_checkpoint_resume_config():
    """Create a config that resumes from a specific checkpoint file"""
    
    print("\n🎯 Creating specific checkpoint resume configuration...")
    
    # Start with a base config  
    config = ConfigFactory.create('sage_model_decisive')
    
    # Configure specific checkpoint resume
    config.logging.resume_from_checkpoint = True
    config.logging.checkpoint_path = "analysis/logs/checkpoints/sage_decisive_training/checkpoint_game_1000_20250813_143022.pth"
    
    # Configure what to load (be selective if needed)
    config.logging.load_model_state = True
    config.logging.load_optimizer_state = True  
    config.logging.load_reward_normalizer = True  # Usually want this
    config.logging.load_game_number = True  # Continue game numbering
    config.logging.load_stat_trackers = False  # Maybe start fresh stats
    config.logging.load_training_state = True
    
    print("✅ Specific checkpoint resume config created!")
    print(f"   Resume from: {config.logging.checkpoint_path}")
    print(f"   Load reward normalizer: {config.logging.load_reward_normalizer}")
    print(f"   Load game number: {config.logging.load_game_number}")
    print(f"   Load stat trackers: {config.logging.load_stat_trackers}")
    
    return config

def main():
    print("🚀 Comprehensive Checkpoint System Usage Examples\n")
    
    # Example 1: Auto-resume from latest checkpoint
    auto_config = create_checkpoint_resume_config()
    
    # Example 2: Resume from specific checkpoint  
    specific_config = create_specific_checkpoint_resume_config()
    
    print("\n📖 Usage Instructions:")
    print("1. For auto-resume: Use the auto_config when starting training")
    print("2. For specific resume: Use the specific_config with the correct checkpoint path")
    print("3. The system will automatically handle loading all training state")
    print("4. Reward normalizer continuity is preserved!")
    print("\n🎉 Ready to use comprehensive checkpointing!")

if __name__ == "__main__":
    main()
