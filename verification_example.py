"""
Example of how to enable enhanced verification and monitoring for PPO training

This shows how to enable various debugging and monitoring features:
1. Input verification between single-sample and batch inference  
2. Comprehensive gradient analysis and monitoring
3. Weight change tracking
4. Action distribution analysis  
5. Adaptive PPO epochs based on gradient norms
"""

# Example usage in your main training script:

def run_training_with_verification():
    from src.agents.RLGNNAgent import RLGNNAgent
    
    # Create agent with enhanced verification
    agent = RLGNNAgent()
    
    # Enable all verification systems for debugging
    agent.enable_verification(
        enable_batch_verification=True,   # Verify inputs match between inference and training
        enable_ppo_verification=True      # Enable comprehensive PPO training analysis
    )
    
    print("🔍 Enhanced verification enabled!")
    print("This will provide:")
    print("  • Input consistency verification")
    print("  • Gradient norm analysis and health checks")
    print("  • Weight change tracking")
    print("  • Action distribution analysis")  
    print("  • Adaptive PPO epoch adjustment")
    print("  • Comprehensive logging to TensorBoard")
    
    # Your existing training loop would continue normally
    # The verification will run automatically and provide detailed diagnostics
    
    return agent

def run_production_training():
    from src.agents.RLGNNAgent import RLGNNAgent
    
    # Create agent without verification for production (default)
    agent = RLGNNAgent()
    
    print("🚀 Production mode - verification disabled for performance")
    print("The agent will still log basic metrics to TensorBoard:")
    print("  • gradient_norm, weight_change_norm, relative_weight_change")
    print("  • adaptive_epochs, gradient_cv") 
    print("  • All existing reward and loss metrics")
    
    return agent

if __name__ == "__main__":
    # For debugging and analysis:
    # agent = run_training_with_verification()
    
    # For production training:
    agent = run_production_training()
