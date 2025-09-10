#!/usr/bin/env python3
"""
Demo script for HRM Autonomous Agent
Demonstrates key capabilities and provides interactive examples
"""
import os
import sys
import asyncio
import logging
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.hrm_model import EnhancedHierarchicalReasoningModel, create_enhanced_hrm_model
from training.autonomous_trainer import AutonomousTrainer, TrainingConfig
from data_collection.data_collector import DataCollector
from evaluation.evaluator import ModelEvaluator
from utils.helpers import set_seed, get_device, log_system_info, Timer
from transformers import T5Tokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HRMDemo:
    """Interactive demo of HRM Autonomous Agent capabilities"""
    
    def __init__(self):
        self.device = get_device()
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.data_collector = DataCollector()
        self.evaluator = ModelEvaluator()
        
        # Set seed for reproducibility
        set_seed(42)
    
    def setup_model(self):
        """Initialize model and tokenizer"""
        logger.info("Setting up model and tokenizer...")
        
        # Initialize tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained("t5-small", use_fast=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
        
        # Model configuration
        model_config = {
            "vocab_size": len(self.tokenizer),
            "d_model": 256,  # Smaller for demo
            "n_heads": 4,
            "d_ff": 1024,
            "dropout": 0.1,
            "halt_max_steps": 6,
            "ponder_loss_weight": 1e-2,
            "num_tools": 50
        }
        
        # Create model
        self.model = create_enhanced_hrm_model(model_config).to(self.device)
        
        logger.info(f"Model created with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    async def demo_data_collection(self):
        """Demonstrate data collection capabilities"""
        print("\n" + "="*60)
        print("🗂️  DATA COLLECTION DEMO")
        print("="*60)
        
        with Timer("Data Collection"):
            # Collect sample data
            collected_data = await self.data_collector.collect_all_data()
            stats = self.data_collector.get_collection_stats()
        
        print(f"\n📊 Collection Statistics:")
        print(f"   Total Sources: {stats.get('total_sources', 0)}")
        print(f"   Active Sources: {stats.get('active_sources', 0)}")
        
        for data_type, type_stats in stats.get('data_types', {}).items():
            print(f"   {data_type.title()}: {type_stats.get('samples', 0)} samples")
        
        return collected_data
    
    async def demo_model_capabilities(self):
        """Demonstrate model reasoning capabilities"""
        print("\n" + "="*60)
        print("🧠 MODEL REASONING DEMO")
        print("="*60)
        
        test_prompts = [
            "What is 15 + 27?",
            "Explain how photosynthesis works",
            "If I have 10 apples and eat 3, how many are left?",
            "List the steps to make a paper airplane"
        ]
        
        self.model.eval()
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n🔍 Test {i}: {prompt}")
            
            try:
                # Tokenize input
                inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=256, truncation=True)
                inputs = inputs.to(self.device)
                
                # Generate response
                with Timer(f"Inference {i}"):
                    outputs = self.model(inputs, return_tool_outputs=True)
                
                # Get logits and generate response (simplified)
                logits = outputs["logits"]
                predicted_ids = logits.argmax(dim=-1)
                response = self.tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
                
                # Remove input from response
                input_text = self.tokenizer.decode(inputs[0], skip_special_tokens=True)
                if response.startswith(input_text):
                    response = response[len(input_text):].strip()
                
                print(f"   Response: {response[:100]}...")
                
                # Show reasoning confidence if available
                if "reasoning_confidence" in outputs:
                    confidence = outputs["reasoning_confidence"]
                    if confidence:
                        avg_confidence = sum(c.item() if hasattr(c, 'item') else c for c in confidence) / len(confidence)
                        print(f"   Confidence: {avg_confidence:.3f}")
                
            except Exception as e:
                print(f"   Error: {e}")
    
    async def demo_evaluation(self):
        """Demonstrate model evaluation"""
        print("\n" + "="*60)
        print("📈 MODEL EVALUATION DEMO")
        print("="*60)
        
        with Timer("Model Evaluation"):
            results = await self.evaluator.comprehensive_evaluation(self.model, self.tokenizer)
        
        print(f"\n📊 Evaluation Results:")
        for metric, score in results.items():
            print(f"   {metric.replace('_', ' ').title()}: {score:.3f}")
        
        return results
    
    async def demo_training_setup(self):
        """Demonstrate training setup"""
        print("\n" + "="*60)
        print("🏋️  TRAINING SETUP DEMO")
        print("="*60)
        
        # Create training configuration
        config = TrainingConfig(
            batch_size=4,  # Small for demo
            learning_rate=1e-4,
            num_epochs=1,
            max_training_time=1,  # 1 hour max for demo
            performance_threshold=0.7
        )
        
        # Model configuration
        model_config = {
            "vocab_size": len(self.tokenizer),
            "d_model": 256,
            "n_heads": 4,
            "d_ff": 1024,
            "dropout": 0.1,
            "halt_max_steps": 6,
            "ponder_loss_weight": 1e-2,
            "num_tools": 50
        }
        
        # Initialize trainer
        self.trainer = AutonomousTrainer(config, model_config)
        self.trainer.initialize_model()
        
        print(f"✅ Training setup complete")
        print(f"   Batch Size: {config.batch_size}")
        print(f"   Learning Rate: {config.learning_rate}")
        print(f"   Model Parameters: {sum(p.numel() for p in self.trainer.model.parameters())}")
        
        # Show training status
        status = self.trainer.get_training_status()
        print(f"   Dataset Size: {status['dataset_size']}")
        print(f"   Current Epoch: {status['current_epoch']}")
        print(f"   Global Step: {status['global_step']}")
    
    def demo_self_improvement(self):
        """Demonstrate self-improvement mechanisms"""
        print("\n" + "="*60)
        print("🔄 SELF-IMPROVEMENT DEMO")
        print("="*60)
        
        if not self.model:
            print("❌ Model not initialized")
            return
        
        # Get improvement suggestions
        suggestions = self.model.get_improvement_suggestions()
        
        print(f"🎯 Current Improvement Suggestions:")
        if suggestions:
            for i, suggestion in enumerate(suggestions, 1):
                print(f"   {i}. {suggestion}")
        else:
            print("   No specific improvements suggested at this time")
        
        # Show performance history
        if hasattr(self.model, 'performance_history') and self.model.performance_history:
            recent_performance = self.model.performance_history[-3:]
            print(f"\n📈 Recent Performance History:")
            for i, perf in enumerate(recent_performance, 1):
                print(f"   {i}. Confidence: {perf.get('confidence', 0):.3f}, "
                      f"Error Likelihood: {perf.get('error_likelihood', 0):.3f}")
        else:
            print(f"\n📈 No performance history available yet")
    
    def demo_web_interface_info(self):
        """Show information about web interface"""
        print("\n" + "="*60)
        print("🌐 WEB INTERFACE INFO")
        print("="*60)
        
        print("The HRM Autonomous Agent includes a comprehensive web dashboard:")
        print("   • Real-time training progress monitoring")
        print("   • Interactive performance visualizations")
        print("   • Configuration management")
        print("   • Data collection controls")
        print("   • Model evaluation triggers")
        print("   • System logs and alerts")
        
        print(f"\nTo start the web interface:")
        print(f"   python main.py --mode web --port 5050")
        print(f"   Then visit: http://localhost:5050")
    
    async def run_full_demo(self):
        """Run complete demonstration"""
        print("🚀 HRM AUTONOMOUS AGENT DEMO")
        print("="*60)
        
        # Log system info
        log_system_info()
        
        # Setup model
        self.setup_model()
        
        # Run demonstrations
        await self.demo_data_collection()
        await self.demo_model_capabilities()
        await self.demo_evaluation()
        await self.demo_training_setup()
        self.demo_self_improvement()
        self.demo_web_interface_info()
        
        print("\n" + "="*60)
        print("✅ DEMO COMPLETED SUCCESSFULLY")
        print("="*60)
        print("Next steps:")
        print("1. Run 'python main.py --mode web' for the web interface")
        print("2. Run 'python main.py --mode autonomous' for full training")
        print("3. Check the README.md for detailed documentation")
        print("4. Explore the configuration options in config/default.json")

def main():
    """Main demo function"""
    print("Starting HRM Autonomous Agent Demo...")
    
    # Create demo instance
    demo = HRMDemo()
    
    # Run the demo
    try:
        asyncio.run(demo.run_full_demo())
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {e}")
        logger.exception("Demo failed")
        return 1
    
    return 0

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Run demo
    exit_code = main()
    sys.exit(exit_code)