"""
Autonomous Training System for HRM
Handles continuous learning, on-the-fly retraining, and model improvement
"""
import os
import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import pickle
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch.optim import AdamW
from transformers import T5Tokenizer, get_linear_schedule_with_warmup
from tqdm.auto import tqdm
import numpy as np

from huggingface_hub import HfApi, HfFolder
import wandb

try:
    from ..core.hrm_model import EnhancedHierarchicalReasoningModel, LLMDataset
    from ..data_collection.data_collector import DataCollector
    from ..evaluation.evaluator import ModelEvaluator
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.hrm_model import EnhancedHierarchicalReasoningModel, LLMDataset
    from data_collection.data_collector import DataCollector
    from evaluation.evaluator import ModelEvaluator

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    # Basic training parameters
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Advanced parameters
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    save_steps: int = 500
    eval_steps: int = 250
    logging_steps: int = 50
    
    # Model parameters
    block_size: int = 512
    max_halt_steps: int = 8
    ponder_weight: float = 1e-2
    
    # Autonomous training parameters
    performance_threshold: float = 0.85
    improvement_threshold: float = 0.02
    max_training_time: int = 24  # hours
    early_stopping_patience: int = 5
    
    # Data parameters
    max_samples_per_type: int = 5000
    data_mixing_ratios: Dict[str, float] = None
    
    def __post_init__(self):
        if self.data_mixing_ratios is None:
            self.data_mixing_ratios = {
                "reasoning": 0.4,
                "instruction": 0.3,
                "tool_use": 0.2,
                "error_correction": 0.1
            }

class ContinuousDataset(Dataset):
    """Dataset that can be updated during training"""
    def __init__(self, tokenizer, block_size=512):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.data = []
        self.processed_data = []
        
    def add_data(self, new_data: List[Dict]):
        """Add new data to the dataset"""
        self.data.extend(new_data)
        self._process_new_data(new_data)
    
    def _process_new_data(self, new_data: List[Dict]):
        """Process new data and add to processed_data"""
        for item in new_data:
            processed = self._process_item(item)
            if processed:
                self.processed_data.extend(processed)
    
    def _process_item(self, item: Dict) -> List[torch.Tensor]:
        """Process individual data item into training examples"""
        text_parts = []
        
        # Handle different data types
        if item.get("type") == "reasoning" or "math" in item.get("type", ""):
            if "question" in item and "answer" in item:
                text = f"Question: {item['question']}\nAnswer: {item['answer']}"
            elif "problem" in item and "solution" in item:
                text = f"Problem: {item['problem']}\nSolution: {item['solution']}"
            else:
                return []
        
        elif item.get("type") == "instruction" or "instruction" in item.get("type", ""):
            instruction = item.get("instruction", "")
            input_text = item.get("input", "")
            output = item.get("output", "")
            
            if input_text:
                text = f"Instruction: {instruction}\nInput: {input_text}\nOutput: {output}"
            else:
                text = f"Instruction: {instruction}\nOutput: {output}"
        
        elif item.get("type") == "tool_use":
            query = item.get("query", "")
            solution = item.get("solution", "")
            tools = item.get("tools", [])
            
            tools_text = ", ".join([t.get("name", "") for t in tools if isinstance(t, dict)])
            text = f"Query: {query}\nAvailable tools: {tools_text}\nSolution: {solution}"
        
        elif item.get("type") == "error_correction":
            incorrect = item.get("incorrect_response", "")
            correct = item.get("correct_response", "")
            text = f"Incorrect: {incorrect}\nCorrect: {correct}"
        
        else:
            # Generic text processing
            text = str(item.get("text", ""))
            if not text:
                return []
        
        # Tokenize and create training examples
        try:
            tokens = self.tokenizer.encode(text, add_special_tokens=True, max_length=self.block_size, truncation=True)
            if len(tokens) < 10:  # Skip very short examples
                return []
            
            # Pad to block_size
            if len(tokens) < self.block_size:
                tokens.extend([self.tokenizer.pad_token_id] * (self.block_size - len(tokens)))
            
            return [torch.tensor(tokens, dtype=torch.long)]
        
        except Exception as e:
            logger.warning(f"Error processing item: {e}")
            return []
    
    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, idx):
        return self.processed_data[idx]
    
    def get_stats(self):
        """Get dataset statistics"""
        return {
            "total_items": len(self.data),
            "processed_examples": len(self.processed_data),
            "data_types": {}
        }

class AutonomousTrainer:
    def __init__(self, config: TrainingConfig, model_config: Dict[str, Any]):
        self.config = config
        self.model_config = model_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.tokenizer = T5Tokenizer.from_pretrained("t5-small", use_fast=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
        
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.mixed_precision)
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_performance = 0.0
        self.training_history = []
        self.performance_plateau_count = 0
        
        # Data components
        self.data_collector = DataCollector()
        self.evaluator = ModelEvaluator()
        self.continuous_dataset = ContinuousDataset(self.tokenizer, config.block_size)
        
        # Autonomous learning state
        self.learning_objectives = {
            "reasoning_accuracy": 0.0,
            "instruction_following": 0.0,
            "tool_use_success": 0.0,
            "error_correction": 0.0,
            "overall_confidence": 0.0
        }
        
        self.improvement_strategies = []
        self.last_data_collection = None
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging and monitoring"""
        try:
            wandb.init(
                project="hrm-autonomous-training",
                config=self.config.__dict__,
                name=f"autonomous_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        except Exception as e:
            logger.warning(f"Could not initialize wandb: {e}")
    
    def initialize_model(self, checkpoint_path: Optional[str] = None):
        """Initialize or load model"""
        # Update model config with tokenizer vocab size
        self.model_config["vocab_size"] = len(self.tokenizer)
        
        # Create model
        self.model = EnhancedHierarchicalReasoningModel(self.model_config).to(self.device)
        
        # Load checkpoint if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_checkpoint(checkpoint_path)
            logger.info(f"Loaded model from checkpoint: {checkpoint_path}")
        else:
            # Initialize weights
            self._initialize_weights()
            logger.info("Initialized new model")
        
        # Setup optimizer and scheduler
        self._setup_optimizer()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        with torch.no_grad():
            for module in self.model.modules():
                if isinstance(module, nn.Linear):
                    module.weight.normal_(0.0, 0.02)
                    if module.bias is not None:
                        module.bias.zero_()
                elif isinstance(module, nn.Embedding):
                    module.weight.normal_(0.0, 0.02)
    
    def _setup_optimizer(self):
        """Setup optimizer and scheduler"""
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if any(nd in name.lower() for nd in ["bias", "norm", "rmsnorm", "layernorm"]):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        self.optimizer = AdamW([
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0}
        ], lr=self.config.learning_rate)
        
        # Scheduler will be updated when we know the dataset size
        self.scheduler = None
    
    async def autonomous_training_loop(self):
        """Main autonomous training loop"""
        logger.info("Starting autonomous training loop...")
        
        training_start_time = datetime.now()
        max_training_duration = timedelta(hours=self.config.max_training_time)
        
        while datetime.now() - training_start_time < max_training_duration:
            try:
                # 1. Collect new data
                await self._collect_and_update_data()
                
                # 2. Evaluate current performance
                current_performance = await self._evaluate_model()
                
                # 3. Determine if training is needed
                if self._should_continue_training(current_performance):
                    # 4. Perform training epoch
                    await self._training_epoch()
                    
                    # 5. Self-evaluation and improvement
                    await self._self_improvement_cycle()
                    
                    # 6. Save checkpoint
                    self._save_checkpoint()
                
                else:
                    logger.info("Performance threshold met, entering monitoring mode...")
                    await asyncio.sleep(3600)  # Wait 1 hour before next check
                
                # 7. Check for fatal errors or intervention needed
                if self._needs_human_intervention():
                    logger.critical("Human intervention required!")
                    break
                
            except Exception as e:
                logger.error(f"Error in training loop: {e}")
                if self._is_fatal_error(e):
                    logger.critical(f"Fatal error encountered: {e}")
                    break
                else:
                    # Continue with next iteration
                    await asyncio.sleep(300)  # Wait 5 minutes before retry
        
        logger.info("Autonomous training loop completed")
    
    async def _collect_and_update_data(self):
        """Collect new data and update dataset"""
        # Check if we need to collect new data
        if (self.last_data_collection is None or 
            datetime.now() - self.last_data_collection > timedelta(hours=6)):
            
            logger.info("Collecting new training data...")
            collected_data = await self.data_collector.collect_all_data()
            
            # Process and add to continuous dataset
            for data_type, data_list in collected_data.items():
                if data_list:
                    # Apply mixing ratios
                    ratio = self.config.data_mixing_ratios.get(data_type, 0.1)
                    max_samples = int(self.config.max_samples_per_type * ratio)
                    
                    if len(data_list) > max_samples:
                        # Sample diverse subset
                        sampled_data = self._sample_diverse_data(data_list, max_samples)
                    else:
                        sampled_data = data_list
                    
                    self.continuous_dataset.add_data(sampled_data)
                    logger.info(f"Added {len(sampled_data)} {data_type} samples")
            
            self.last_data_collection = datetime.now()
    
    def _sample_diverse_data(self, data_list: List[Dict], max_samples: int) -> List[Dict]:
        """Sample diverse subset of data"""
        if len(data_list) <= max_samples:
            return data_list
        
        # Simple diversity sampling - in practice, you'd use embeddings
        sampled = []
        step = len(data_list) // max_samples
        
        for i in range(0, len(data_list), step):
            if len(sampled) >= max_samples:
                break
            sampled.append(data_list[i])
        
        return sampled
    
    async def _evaluate_model(self) -> Dict[str, float]:
        """Evaluate current model performance"""
        if self.model is None:
            return {"overall": 0.0}
        
        try:
            # Use the evaluator to assess performance
            performance = await self.evaluator.comprehensive_evaluation(self.model, self.tokenizer)
            
            # Update learning objectives
            self.learning_objectives.update(performance)
            
            # Log performance
            if wandb.run:
                wandb.log(performance, step=self.global_step)
            
            return performance
        
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            return {"overall": 0.0}
    
    def _should_continue_training(self, current_performance: Dict[str, float]) -> bool:
        """Determine if training should continue"""
        overall_performance = current_performance.get("overall", 0.0)
        
        # Check if we've reached the performance threshold
        if overall_performance >= self.config.performance_threshold:
            return False
        
        # Check if we're making progress
        if overall_performance <= self.best_performance + self.config.improvement_threshold:
            self.performance_plateau_count += 1
        else:
            self.performance_plateau_count = 0
            self.best_performance = overall_performance
        
        # Stop if we've plateaued for too long
        if self.performance_plateau_count >= self.config.early_stopping_patience:
            logger.info("Performance plateau detected, stopping training")
            return False
        
        return True
    
    async def _training_epoch(self):
        """Perform one training epoch"""
        if len(self.continuous_dataset) == 0:
            logger.warning("No training data available")
            return
        
        # Create data loader
        dataloader = DataLoader(
            self.continuous_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True
        )
        
        # Update scheduler if needed
        if self.scheduler is None:
            total_steps = len(dataloader) * self.config.num_epochs
            warmup_steps = int(total_steps * self.config.warmup_ratio)
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer, warmup_steps, total_steps
            )
        
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Training Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                input_ids = batch.to(self.device)
                attention_mask = torch.ones_like(input_ids)
                labels = input_ids.clone()
                
                # Forward pass
                with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
                    outputs = self.model(input_ids, labels=labels, attention_mask=attention_mask)
                    loss = outputs["loss"]
                
                if loss is None or not torch.isfinite(loss):
                    continue
                
                # Backward pass
                loss = loss / self.config.gradient_accumulation_steps
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    
                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    self.global_step += 1
                
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "avg_loss": f"{total_loss/num_batches:.4f}"
                })
                
                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    if wandb.run:
                        wandb.log({
                            "train_loss": loss.item(),
                            "learning_rate": self.scheduler.get_last_lr()[0],
                            "global_step": self.global_step
                        })
                
            except Exception as e:
                logger.error(f"Error in training step: {e}")
                continue
        
        avg_loss = total_loss / max(num_batches, 1)
        logger.info(f"Epoch {self.current_epoch} completed. Average loss: {avg_loss:.4f}")
        
        self.current_epoch += 1
        
        # Record training history
        self.training_history.append({
            "epoch": self.current_epoch,
            "loss": avg_loss,
            "global_step": self.global_step,
            "timestamp": datetime.now().isoformat()
        })
    
    async def _self_improvement_cycle(self):
        """Perform self-improvement analysis and adjustments"""
        logger.info("Starting self-improvement cycle...")
        
        # Get improvement suggestions from the model
        suggestions = self.model.get_improvement_suggestions()
        
        # Analyze training history for patterns
        if len(self.training_history) >= 3:
            recent_losses = [h["loss"] for h in self.training_history[-3:]]
            if all(recent_losses[i] >= recent_losses[i+1] for i in range(len(recent_losses)-1)):
                suggestions.append("Loss not decreasing - consider adjusting learning rate")
        
        # Apply improvements
        for suggestion in suggestions:
            await self._apply_improvement(suggestion)
        
        # Update improvement strategies
        self.improvement_strategies.extend(suggestions)
        
        logger.info(f"Applied {len(suggestions)} improvement strategies")
    
    async def _apply_improvement(self, suggestion: str):
        """Apply specific improvement strategy"""
        try:
            if "learning rate" in suggestion.lower():
                # Adjust learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                new_lr = current_lr * 0.9  # Reduce by 10%
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = new_lr
                logger.info(f"Adjusted learning rate from {current_lr} to {new_lr}")
            
            elif "reasoning tasks" in suggestion.lower():
                # Collect more reasoning data
                reasoning_data = await self.data_collector._collect_from_dataset(
                    self.data_collector.data_sources[0]  # GSM8K
                )
                if reasoning_data:
                    self.continuous_dataset.add_data(reasoning_data[:100])
                    logger.info("Added additional reasoning training data")
            
            elif "error correction" in suggestion.lower():
                # Generate more error correction examples
                error_data = self.data_collector._generate_error_correction_data()
                self.continuous_dataset.add_data(error_data[:50])
                logger.info("Added additional error correction training data")
            
            elif "halt mechanism" in suggestion.lower():
                # Adjust halt mechanism parameters
                if hasattr(self.model, 'max_steps') and self.model.max_steps > 4:
                    self.model.max_steps -= 1
                    logger.info(f"Reduced max halt steps to {self.model.max_steps}")
        
        except Exception as e:
            logger.error(f"Error applying improvement '{suggestion}': {e}")
    
    def _save_checkpoint(self):
        """Save training checkpoint"""
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{self.global_step}.pt")
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "best_performance": self.best_performance,
            "training_history": self.training_history,
            "learning_objectives": self.learning_objectives,
            "config": self.config.__dict__,
            "model_config": self.model_config
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Keep only recent checkpoints
        self._cleanup_old_checkpoints(checkpoint_dir)
    
    def _cleanup_old_checkpoints(self, checkpoint_dir: str, keep_last: int = 5):
        """Remove old checkpoints to save space"""
        try:
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_")]
            checkpoints.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
            
            if len(checkpoints) > keep_last:
                for checkpoint in checkpoints[:-keep_last]:
                    os.remove(os.path.join(checkpoint_dir, checkpoint))
        except Exception as e:
            logger.warning(f"Error cleaning up checkpoints: {e}")
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        self.global_step = checkpoint.get("global_step", 0)
        self.current_epoch = checkpoint.get("current_epoch", 0)
        self.best_performance = checkpoint.get("best_performance", 0.0)
        self.training_history = checkpoint.get("training_history", [])
        self.learning_objectives = checkpoint.get("learning_objectives", {})
    
    def _needs_human_intervention(self) -> bool:
        """Check if human intervention is needed"""
        # Check for critical performance degradation
        if len(self.training_history) >= 5:
            recent_losses = [h["loss"] for h in self.training_history[-5:]]
            if all(loss > 10.0 for loss in recent_losses):  # Very high losses
                return True
        
        # Check for memory issues
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            if memory_used > 0.95:  # Using >95% of GPU memory
                return True
        
        # Check for training stagnation
        if self.performance_plateau_count >= self.config.early_stopping_patience * 2:
            return True
        
        return False
    
    def _is_fatal_error(self, error: Exception) -> bool:
        """Determine if an error is fatal and requires stopping"""
        fatal_error_types = [
            torch.cuda.OutOfMemoryError,
            RuntimeError,  # Often CUDA errors
        ]
        
        return any(isinstance(error, error_type) for error_type in fatal_error_types)
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status"""
        return {
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "best_performance": self.best_performance,
            "learning_objectives": self.learning_objectives,
            "dataset_size": len(self.continuous_dataset),
            "improvement_strategies": self.improvement_strategies[-10:],  # Last 10
            "training_history": self.training_history[-5:],  # Last 5 epochs
            "performance_plateau_count": self.performance_plateau_count
        }

# Example usage
async def main():
    config = TrainingConfig()
    model_config = {
        "d_model": 512,
        "n_heads": 8,
        "d_ff": 2048,
        "dropout": 0.1,
        "halt_max_steps": 8,
        "ponder_loss_weight": 1e-2,
        "num_tools": 100
    }
    
    trainer = AutonomousTrainer(config, model_config)
    trainer.initialize_model()
    
    await trainer.autonomous_training_loop()

if __name__ == "__main__":
    asyncio.run(main())