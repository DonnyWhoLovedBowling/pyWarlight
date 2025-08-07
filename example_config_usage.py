"""
Example usage of the new configuration system for PPO training.

This demonstrates how to use the comprehensive configuration system
to easily switch between different training modes.
"""

from src.config.training_config import ConfigFactory, VerificationConfig
from src.agents.RLUtils.PPOAgent import PPOAgent
import torch

def example_debug_training():
    """Example: Full debug mode with all verifications enabled"""
    print("=== DEBUG TRAINING EXAMPLE ===")
    
    # Get debug configuration with all verifications enabled
    config = ConfigFactory.get_debug_config()
    
    # Create PPO agent with debug verification
    # Assuming you have a policy and optimizer
    # ppo_agent = PPOAgent(
    #     policy=your_policy,
    #     optimizer=your_optimizer,
    #     verification_config=config.verification,
    #     **config.ppo.__dict__
    # )
    
    print(f"Debug config has {sum(config.verification.__dict__.values())} verifications enabled")
    print("Enabled verifications:", [k for k, v in config.verification.__dict__.items() if v])

def example_production_training():
    """Example: Production mode with minimal overhead"""
    print("\n=== PRODUCTION TRAINING EXAMPLE ===")
    
    # Get production configuration with verifications disabled
    config = ConfigFactory.get_production_config()
    
    # Create PPO agent for production
    # ppo_agent = PPOAgent(
    #     policy=your_policy,
    #     optimizer=your_optimizer,
    #     verification_config=config.verification,
    #     **config.ppo.__dict__
    # )
    
    print(f"Production config has {sum(config.verification.__dict__.values())} verifications enabled")
    print("This configuration is optimized for maximum performance")

def example_custom_verification():
    """Example: Custom verification configuration"""
    print("\n=== CUSTOM VERIFICATION EXAMPLE ===")
    
    # Create custom verification config - only enable specific checks
    custom_verification = VerificationConfig(
        verify_structural_integrity=True,
        verify_model_outputs=False,
        verify_single_vs_batch=False,
        verify_buffer_integrity=True,
        verify_action_data=False,
        verify_old_log_probs=False,
        check_extreme_log_probs=True,
        check_extreme_attack_diffs=False,
        analyze_gradients=True,
        analyze_weight_changes=False,
        analyze_action_distribution=True
    )
    
    # Create PPO agent with custom verification
    # ppo_agent = PPOAgent(
    #     policy=your_policy,
    #     optimizer=your_optimizer,
    #     verification_config=custom_verification
    # )
    
    enabled_checks = [k for k, v in custom_verification.__dict__.items() if v]
    print(f"Custom config enables only: {enabled_checks}")

def example_analysis_mode():
    """Example: Analysis mode for detailed monitoring"""
    print("\n=== ANALYSIS MODE EXAMPLE ===")
    
    # Get analysis configuration
    config = ConfigFactory.get_analysis_config()
    
    # This configuration enables key analytical verifications
    # while disabling expensive structural checks
    print("Analysis mode enables:")
    enabled_checks = [k for k, v in config.verification.__dict__.items() if v]
    for check in enabled_checks:
        print(f"  - {check}")

def example_fast_debug():
    """Example: Fast debug mode for quick development iterations"""
    print("\n=== FAST DEBUG MODE EXAMPLE ===")
    
    # Get fast debug configuration
    config = ConfigFactory.get_fast_debug_config()
    
    print("Fast debug mode - minimal verification overhead:")
    enabled_checks = [k for k, v in config.verification.__dict__.items() if v]
    for check in enabled_checks:
        print(f"  - {check}")

if __name__ == "__main__":
    # Run all examples
    example_debug_training()
    example_production_training()
    example_custom_verification()
    example_analysis_mode()
    example_fast_debug()
    
    print("\n=== CONFIGURATION COMPARISON ===")
    configs = {
        'Debug': ConfigFactory.get_debug_config(),
        'Production': ConfigFactory.get_production_config(),
        'Analysis': ConfigFactory.get_analysis_config(),
        'Fast Debug': ConfigFactory.get_fast_debug_config()
    }
    
    print(f"{'Mode':<12} {'Verifications':<15} {'PPO Epochs':<12} {'Adaptive':<10}")
    print("-" * 50)
    
    for name, config in configs.items():
        verification_count = sum(config.verification.__dict__.values())
        print(f"{name:<12} {verification_count:<15} {config.ppo.epochs:<12} {config.ppo.adaptive_epochs}")
