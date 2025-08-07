#!/usr/bin/env python3
"""
Model Architecture Testing Script

This script allows you to easily test different model architectures
and compare their gradient stability characteristics.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.RLUtils.ModelFactory import ModelFactory, print_model_comparison
from src.config.training_config import ConfigFactory
import torch

def test_model_creation():
    """Test that all model architectures can be created successfully"""
    print("🧪 TESTING MODEL CREATION")
    print("=" * 40)
    
    model_types = ['standard', 'residual', 'sage', 'transformer']
    
    for model_type in model_types:
        try:
            print(f"\n📋 Testing {model_type} model...")
            
            # Create model
            model = ModelFactory.create_model(
                model_type=model_type,
                node_feat_dim=8,
                embed_dim=64,
                max_army_send=50
            )
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"   ✅ Model created successfully")
            print(f"   📊 Total parameters: {total_params:,}")
            print(f"   🎯 Trainable parameters: {trainable_params:,}")
            
            # Test forward pass
            node_features = torch.randn(42, 8)  # 42 regions, 8 features each
            action_edges = torch.randint(0, 42, (30, 2))  # 30 possible edges
            army_counts = torch.randint(1, 20, (42,)).float()
            
            with torch.no_grad():
                placement_logits, attack_logits, army_logits = model(
                    node_features, action_edges, army_counts, action="Phase.PLACE_ARMIES"
                )
                
            print(f"   🔍 Forward pass successful")
            print(f"   📏 Placement logits shape: {placement_logits.shape}")
            
        except Exception as e:
            print(f"   ❌ Error creating {model_type} model: {e}")
    
    print("\n" + "=" * 40)

def test_configurations():
    """Test that all model-specific configurations work"""
    print("\n🔧 TESTING CONFIGURATIONS")
    print("=" * 40)
    
    config_names = ['residual_model', 'sage_model', 'transformer_model']
    
    for config_name in config_names:
        try:
            print(f"\n📋 Testing {config_name} config...")
            
            config = ConfigFactory.create(config_name)
            
            print(f"   ✅ Configuration created successfully")
            print(f"   🏗️  Model type: {config.model.model_type}")
            print(f"   📚 Learning rate: {config.ppo.learning_rate}")
            print(f"   🔄 PPO epochs: {config.ppo.ppo_epochs}")
            print(f"   📦 Batch size: {config.ppo.batch_size}")
            print(f"   ✂️  Gradient clip: {config.ppo.gradient_clip_norm}")
            print(f"   ⚖️  Value loss coeff: {config.ppo.value_loss_coeff}")
            
        except Exception as e:
            print(f"   ❌ Error creating {config_name} config: {e}")
    
    print("\n" + "=" * 40)

def show_recommendations():
    """Show recommendations for different use cases"""
    print("\n💡 RECOMMENDATIONS FOR YOUR USE CASE")
    print("=" * 50)
    
    print("""
🎯 FOR GRADIENT EXPLOSION ISSUES (Your Current Problem):
   
   1️⃣ IMMEDIATE SOLUTION - Residual Model:
      • Use: ConfigFactory.create('residual_model')
      • Benefits: Residual connections + LayerNorm prevent gradient explosion
      • Expected gradient norms: 10-100 (vs 60,000+ currently)
      • Can handle 2 epochs safely with higher learning rates
   
   2️⃣ ALTERNATIVE - SAGE Model:
      • Use: ConfigFactory.create('sage_model')  
      • Benefits: Sampling-based aggregation is inherently more stable
      • Good for large maps with many regions
      • Less memory intensive than residual connections
   
   3️⃣ ADVANCED - Transformer Model:
      • Use: ConfigFactory.create('transformer_model')
      • Benefits: Attention mechanism excellent for strategic reasoning
      • Handles long-range dependencies (continent control, etc.)
      • Requires more tuning but very powerful
   
🔧 QUICK START:
   Just change line 66 in RLGNNAgent.py from:
   config = ConfigFactory.create('value_regularized_multi_epoch')
   to:
   config = ConfigFactory.create('residual_model')
   
🎓 WHY THESE WORK BETTER:
   • Your current GCN has no normalization → gradient explosion
   • Residual: Skip connections prevent vanishing/exploding gradients  
   • SAGE: Sampling reduces sensitivity to graph structure changes
   • Transformer: Self-attention with proper normalization is very stable
   
📊 EXPECTED RESULTS:
   • Gradient norms: 10-300 instead of 60,000+
   • Stable multi-epoch training (2+ epochs)
   • Better learning from strategic patterns
   • Reduced army entropy (more decisive actions)
    """)

def main():
    """Main testing function"""
    print("🚀 MODEL ARCHITECTURE TESTING SUITE")
    print("=" * 60)
    
    # Show comparison
    print_model_comparison()
    
    # Test model creation
    test_model_creation()
    
    # Test configurations  
    test_configurations()
    
    # Show recommendations
    show_recommendations()
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    main()
