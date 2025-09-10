#!/usr/bin/env python3
"""
Quick Start Script for HRM Autonomous Agent
Provides easy access to all major functionalities
"""
import os
import sys
import argparse
import asyncio

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def print_banner():
    """Print welcome banner"""
    print("=" * 70)
    print("🧠 HRM AUTONOMOUS AGENT - QUICK START")
    print("=" * 70)
    print("Advanced autonomous system for continuous HRM training")
    print("Features: Reasoning, Tool Use, Error Correction, Self-Improvement")
    print("=" * 70)

def print_menu():
    """Print main menu options"""
    print("\n📋 Available Options:")
    print("1. 🌐 Launch Web Dashboard (Recommended)")
    print("2. 🎮 Run Interactive Demo")
    print("3. 🤖 Start Autonomous Training")
    print("4. 📊 Collect Training Data")
    print("5. 📈 Evaluate Model Performance")
    print("6. ⚙️  Generate Configuration File")
    print("7. 📚 Show Documentation")
    print("8. 🚪 Exit")

def launch_web_dashboard():
    """Launch the web dashboard"""
    print("\n🌐 Launching Web Dashboard...")
    print("The dashboard will be available at: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    
    try:
        from web_interface.app import app, socketio
        socketio.run(app, host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        print("\n👋 Web dashboard stopped")
    except Exception as e:
        print(f"❌ Error launching web dashboard: {e}")

async def run_demo():
    """Run the interactive demo"""
    print("\n🎮 Running Interactive Demo...")
    
    try:
        from run_demo import HRMDemo
        demo = HRMDemo()
        await demo.run_full_demo()
    except Exception as e:
        print(f"❌ Demo failed: {e}")

async def start_autonomous_training():
    """Start autonomous training"""
    print("\n🤖 Starting Autonomous Training...")
    print("This will run continuous training with self-improvement")
    
    confirm = input("Continue? (y/N): ").lower().strip()
    if confirm != 'y':
        print("Training cancelled")
        return
    
    try:
        from training.autonomous_trainer import AutonomousTrainer, TrainingConfig
        from transformers import T5Tokenizer
        
        # Create configuration
        config = TrainingConfig(
            batch_size=16,
            learning_rate=2e-5,
            num_epochs=5,
            max_training_time=24,
            performance_threshold=0.85
        )
        
        # Model configuration
        tokenizer = T5Tokenizer.from_pretrained("t5-small", use_fast=False)
        model_config = {
            "vocab_size": len(tokenizer),
            "d_model": 512,
            "n_heads": 8,
            "d_ff": 2048,
            "dropout": 0.1,
            "halt_max_steps": 8,
            "ponder_loss_weight": 1e-2,
            "num_tools": 100
        }
        
        # Initialize and run trainer
        trainer = AutonomousTrainer(config, model_config)
        trainer.initialize_model()
        
        print("🚀 Starting autonomous training loop...")
        await trainer.autonomous_training_loop()
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"❌ Training failed: {e}")

async def collect_data():
    """Collect training data"""
    print("\n📊 Collecting Training Data...")
    
    try:
        from data_collection.data_collector import DataCollector
        
        collector = DataCollector()
        collected_data = await collector.collect_all_data()
        stats = collector.get_collection_stats()
        
        print(f"✅ Data collection completed!")
        print(f"📈 Statistics: {stats}")
        
    except Exception as e:
        print(f"❌ Data collection failed: {e}")

async def evaluate_model():
    """Evaluate model performance"""
    print("\n📈 Evaluating Model Performance...")
    
    try:
        from evaluation.evaluator import ModelEvaluator
        from core.hrm_model import EnhancedHierarchicalReasoningModel
        from transformers import T5Tokenizer
        
        # Create a simple model for evaluation
        tokenizer = T5Tokenizer.from_pretrained("t5-small", use_fast=False)
        model_config = {
            "vocab_size": len(tokenizer),
            "d_model": 256,  # Smaller for quick demo
            "n_heads": 4,
            "d_ff": 1024,
            "dropout": 0.1,
            "halt_max_steps": 6,
            "ponder_loss_weight": 1e-2,
            "num_tools": 50
        }
        
        model = EnhancedHierarchicalReasoningModel(model_config)
        evaluator = ModelEvaluator()
        
        print("🔍 Running comprehensive evaluation...")
        results = await evaluator.comprehensive_evaluation(model, tokenizer)
        
        print(f"✅ Evaluation completed!")
        print(f"📊 Results:")
        for metric, score in results.items():
            print(f"   {metric.replace('_', ' ').title()}: {score:.3f}")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")

def generate_config():
    """Generate configuration file"""
    print("\n⚙️  Generating Configuration File...")
    
    try:
        import json
        from main import create_default_config
        
        config = create_default_config()
        config_path = "config/my_config.json"
        
        os.makedirs("config", exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Configuration saved to: {config_path}")
        print("You can edit this file to customize training parameters")
        
    except Exception as e:
        print(f"❌ Config generation failed: {e}")

def show_documentation():
    """Show documentation"""
    print("\n📚 Documentation")
    print("=" * 50)
    print("📖 README.md - Complete documentation and setup guide")
    print("⚙️  config/default.json - Configuration options")
    print("🌐 Web Dashboard - Interactive monitoring and control")
    print("🎮 Demo Mode - Showcase of capabilities")
    print("🤖 Autonomous Mode - Fully automated training")
    print("")
    print("🔗 Key Files:")
    print("   • main.py - Main entry point")
    print("   • run_demo.py - Interactive demonstration")
    print("   • core/hrm_model.py - Enhanced HRM implementation")
    print("   • training/autonomous_trainer.py - Training system")
    print("   • data_collection/data_collector.py - Data gathering")
    print("   • evaluation/evaluator.py - Performance assessment")
    print("   • web_interface/app.py - Web dashboard")
    print("")
    print("🚀 Quick Commands:")
    print("   python main.py --mode web")
    print("   python main.py --mode autonomous")
    print("   python run_demo.py")

async def main():
    """Main interactive loop"""
    print_banner()
    
    while True:
        print_menu()
        
        try:
            choice = input("\n🎯 Select option (1-8): ").strip()
            
            if choice == '1':
                launch_web_dashboard()
            elif choice == '2':
                await run_demo()
            elif choice == '3':
                await start_autonomous_training()
            elif choice == '4':
                await collect_data()
            elif choice == '5':
                await evaluate_model()
            elif choice == '6':
                generate_config()
            elif choice == '7':
                show_documentation()
            elif choice == '8':
                print("\n👋 Goodbye!")
                break
            else:
                print("❌ Invalid option. Please select 1-8.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('config', exist_ok=True)
    
    # Run main loop
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)