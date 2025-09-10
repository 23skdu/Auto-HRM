#!/usr/bin/env python3
"""
Simple system test for HRM Autonomous Agent
"""
import os
import sys
import torch
from transformers import T5Tokenizer

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test all major imports"""
    print("🧪 Testing System Imports...")
    
    try:
        from core.hrm_model import EnhancedHierarchicalReasoningModel
        print("✅ Core model import successful")
    except Exception as e:
        print(f"❌ Core model import failed: {e}")
        return False
    
    try:
        from data_collection.data_collector import DataCollector
        print("✅ Data collector import successful")
    except Exception as e:
        print(f"❌ Data collector import failed: {e}")
        return False
    
    try:
        from training.autonomous_trainer import AutonomousTrainer, TrainingConfig
        print("✅ Autonomous trainer import successful")
    except Exception as e:
        print(f"❌ Autonomous trainer import failed: {e}")
        return False
    
    try:
        from evaluation.evaluator import ModelEvaluator
        print("✅ Model evaluator import successful")
    except Exception as e:
        print(f"❌ Model evaluator import failed: {e}")
        return False
    
    return True

def test_model_creation():
    """Test model creation"""
    print("\n🧠 Testing Model Creation...")
    
    try:
        from core.hrm_model import EnhancedHierarchicalReasoningModel
        
        # Initialize tokenizer
        tokenizer = T5Tokenizer.from_pretrained("t5-small", use_fast=False)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
        
        # Model configuration
        model_config = {
            "vocab_size": len(tokenizer),
            "d_model": 128,  # Very small for testing
            "n_heads": 2,
            "d_ff": 256,
            "dropout": 0.1,
            "halt_max_steps": 4,
            "ponder_loss_weight": 1e-2,
            "num_tools": 10
        }
        
        # Create model
        model = EnhancedHierarchicalReasoningModel(model_config)
        
        # Test forward pass
        test_input = torch.randint(0, len(tokenizer), (1, 10))
        with torch.no_grad():
            outputs = model(test_input)
        
        print(f"✅ Model created successfully")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters())}")
        print(f"   Output shape: {outputs['logits'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

def test_basic_functionality():
    """Test basic system functionality"""
    print("\n⚙️  Testing Basic Functionality...")
    
    try:
        # Test configuration loading
        from main import create_default_config
        config = create_default_config()
        print("✅ Configuration system working")
        
        # Test utilities
        from utils.helpers import set_seed, get_device
        set_seed(42)
        device = get_device()
        print(f"✅ Utilities working (device: {device})")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 HRM AUTONOMOUS AGENT - SYSTEM TEST")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed")
        return 1
    
    # Test model creation
    if not test_model_creation():
        print("\n❌ Model creation tests failed")
        return 1
    
    # Test basic functionality
    if not test_basic_functionality():
        print("\n❌ Basic functionality tests failed")
        return 1
    
    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED!")
    print("=" * 50)
    print("\n📋 System is ready for use!")
    print("\n🚀 Next steps:")
    print("   1. Run: python quick_start.py")
    print("   2. Or: python main.py --mode web")
    print("   3. Or: python run_demo.py")
    
    return 0

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    exit_code = main()
    sys.exit(exit_code)