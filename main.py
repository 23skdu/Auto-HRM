"""
Main entry point for the HRM Autonomous Agent System
Provides both CLI and web interface options
"""
import os
import sys
import argparse
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.hrm_model import EnhancedHierarchicalReasoningModel
from training.autonomous_trainer import AutonomousTrainer, TrainingConfig
from data_collection.data_collector import DataCollector
from evaluation.evaluator import ModelEvaluator
from web_interface.app import app, socketio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hrm_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HRMAutonomousAgent:
    """Main autonomous agent controller"""
    
    def __init__(self):
        self.trainer = None
        self.data_collector = DataCollector()
        self.evaluator = ModelEvaluator()
        
    async def run_autonomous_mode(self, config: Dict[str, Any]):
        """Run the agent in fully autonomous mode"""
        logger.info("Starting HRM Autonomous Agent in autonomous mode...")
        
        # Create training configuration
        training_config = TrainingConfig(
            batch_size=config.get('batch_size', 16),
            learning_rate=config.get('learning_rate', 2e-5),
            num_epochs=config.get('num_epochs', 3),
            max_training_time=config.get('max_training_time', 24),
            performance_threshold=config.get('performance_threshold', 0.85),
            mixed_precision=config.get('mixed_precision', True)
        )
        
        # Model configuration
        model_config = {
            "d_model": config.get('d_model', 512),
            "n_heads": config.get('n_heads', 8),
            "d_ff": config.get('d_ff', 2048),
            "dropout": config.get('dropout', 0.1),
            "halt_max_steps": config.get('halt_max_steps', 8),
            "ponder_loss_weight": config.get('ponder_weight', 1e-2),
            "num_tools": config.get('num_tools', 100)
        }
        
        # Initialize trainer
        self.trainer = AutonomousTrainer(training_config, model_config)
        self.trainer.initialize_model(config.get('checkpoint_path'))
        
        # Start autonomous training loop
        await self.trainer.autonomous_training_loop()
        
        logger.info("Autonomous training completed")
    
    async def run_data_collection(self):
        """Run data collection only"""
        logger.info("Starting data collection...")
        
        collected_data = await self.data_collector.collect_all_data()
        stats = self.data_collector.get_collection_stats()
        
        logger.info(f"Data collection completed. Stats: {stats}")
        return collected_data, stats
    
    async def run_evaluation(self, model_path: str = None):
        """Run model evaluation only"""
        logger.info("Starting model evaluation...")
        
        if model_path and os.path.exists(model_path):
            # Load model from checkpoint
            # This would need to be implemented based on your checkpoint format
            logger.info(f"Loading model from {model_path}")
        
        if self.trainer and self.trainer.model:
            results = await self.evaluator.comprehensive_evaluation(
                self.trainer.model, self.trainer.tokenizer
            )
            logger.info(f"Evaluation results: {results}")
            return results
        else:
            logger.error("No model available for evaluation")
            return None
    
    def run_web_interface(self, host: str = '0.0.0.0', port: int = 5000):
        """Run the web interface"""
        logger.info(f"Starting web interface on {host}:{port}")
        socketio.run(app, host=host, port=port, debug=False)

def create_default_config() -> Dict[str, Any]:
    """Create default configuration"""
    return {
        # Training parameters
        'batch_size': 16,
        'learning_rate': 2e-5,
        'num_epochs': 3,
        'max_training_time': 24,
        'performance_threshold': 0.85,
        'mixed_precision': True,
        
        # Model parameters
        'd_model': 512,
        'n_heads': 8,
        'd_ff': 2048,
        'dropout': 0.1,
        'halt_max_steps': 8,
        'ponder_weight': 1e-2,
        'num_tools': 100,
        
        # Paths
        'checkpoint_path': None,
        'data_dir': 'data',
        'output_dir': 'outputs'
    }

def load_config_from_file(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    import json
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {e}")
        return create_default_config()

def save_config_to_file(config: Dict[str, Any], config_path: str):
    """Save configuration to JSON file"""
    import json
    
    try:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Saved configuration to {config_path}")
    except Exception as e:
        logger.error(f"Error saving config to {config_path}: {e}")

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='HRM Autonomous Agent')
    parser.add_argument('--mode', choices=['autonomous', 'web', 'collect', 'evaluate'], 
                       default='web', help='Operation mode')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--checkpoint', type=str, help='Model checkpoint path')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Web interface host')
    parser.add_argument('--port', type=int, default=5000, help='Web interface port')
    parser.add_argument('--save-config', type=str, help='Save default config to file')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--max-time', type=int, default=24, help='Max training time (hours)')
    
    # Model parameters
    parser.add_argument('--d-model', type=int, default=512, help='Model dimension')
    parser.add_argument('--n-heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--d-ff', type=int, default=2048, help='Feed forward dimension')
    parser.add_argument('--max-halt-steps', type=int, default=8, help='Max halt steps')
    
    args = parser.parse_args()
    
    # Load or create configuration
    if args.config and os.path.exists(args.config):
        config = load_config_from_file(args.config)
    else:
        config = create_default_config()
    
    # Override config with command line arguments
    config.update({
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'num_epochs': args.epochs,
        'max_training_time': args.max_time,
        'd_model': args.d_model,
        'n_heads': args.n_heads,
        'd_ff': args.d_ff,
        'halt_max_steps': args.max_halt_steps,
        'checkpoint_path': args.checkpoint
    })
    
    # Save config if requested
    if args.save_config:
        save_config_to_file(config, args.save_config)
        print(f"Default configuration saved to {args.save_config}")
        return
    
    # Initialize agent
    agent = HRMAutonomousAgent()
    
    try:
        if args.mode == 'autonomous':
            logger.info("Running in autonomous mode...")
            await agent.run_autonomous_mode(config)
        
        elif args.mode == 'collect':
            logger.info("Running data collection...")
            collected_data, stats = await agent.run_data_collection()
            print(f"Data collection completed. Stats: {stats}")
        
        elif args.mode == 'evaluate':
            logger.info("Running model evaluation...")
            results = await agent.run_evaluation(args.checkpoint)
            if results:
                print(f"Evaluation results: {results}")
        
        elif args.mode == 'web':
            logger.info("Starting web interface...")
            agent.run_web_interface(args.host, args.port)
        
        else:
            logger.error(f"Unknown mode: {args.mode}")
            return 1
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Run main function
    exit_code = asyncio.run(main())
    sys.exit(exit_code)