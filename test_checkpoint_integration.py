#!/usr/bin/env python3
"""
Test script to verify the checkpoint system integration
"""

import sys
import os
sys.path.append('.')

def test_checkpoint_integration():
    """Test basic checkpoint system integration"""
    
    print("🧪 Testing checkpoint system integration...")
    
    try:
        # Test 1: Import all required modules
        from src.agents.RLUtils.CheckpointManager import CheckpointManager
        from src.config.training_config import ConfigFactory
        print("✅ All modules imported successfully")
        
        # Test 2: Create config
        config = ConfigFactory.create('sage_model')
        print("✅ Config created successfully")
        
        # Test 3: Initialize CheckpointManager
        checkpoint_manager = CheckpointManager(config, 'integration_test')
        print(f"✅ CheckpointManager initialized successfully")
        print(f"   📁 Directory: {checkpoint_manager.checkpoint_dir}")
        print(f"   💾 Save frequency: every {checkpoint_manager.save_frequency} games")
        
        # Test 4: Test checkpoint directory creation
        if os.path.exists(checkpoint_manager.checkpoint_dir):
            print("✅ Checkpoint directory created successfully")
        else:
            print("❌ Checkpoint directory not created")
            return False
            
        # Test 5: Test checkpoint config settings
        print("\n📝 Checkpoint configuration:")
        print(f"   resume_from_checkpoint: {config.logging.resume_from_checkpoint}")
        print(f"   auto_resume_latest: {config.logging.auto_resume_latest}")
        print(f"   checkpoint_every_n_episodes: {config.logging.checkpoint_every_n_episodes}")
        print(f"   keep_last_n_checkpoints: {config.logging.keep_last_n_checkpoints}")
        
        # Test 6: Test checkpoint resume configuration
        print("\n⚙️  Checkpoint loading configuration:")
        print(f"   load_model_state: {config.logging.load_model_state}")
        print(f"   load_optimizer_state: {config.logging.load_optimizer_state}")
        print(f"   load_reward_normalizer: {config.logging.load_reward_normalizer}")
        print(f"   load_game_number: {config.logging.load_game_number}")
        print(f"   load_stat_trackers: {config.logging.load_stat_trackers}")
        print(f"   load_training_state: {config.logging.load_training_state}")
        
        print("\n🎉 All checkpoint system tests passed!")
        print("\n💡 To use the checkpoint system:")
        print("1. Set resume_from_checkpoint=True and checkpoint_path='path/to/checkpoint.pth' to resume from a specific checkpoint")
        print("2. Set auto_resume_latest=True to automatically resume from the latest checkpoint")
        print("3. Use checkpoint_every_n_episodes to control save frequency")
        print("4. Use the load_* flags to control what gets loaded from checkpoints")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_checkpoint_integration()
    if success:
        print("\n✅ Checkpoint system integration test PASSED")
        sys.exit(0)
    else:
        print("\n❌ Checkpoint system integration test FAILED")
        sys.exit(1)
