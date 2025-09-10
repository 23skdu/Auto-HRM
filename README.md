# HRM Autonomous Agent System

An advanced autonomous agent system that continuously trains and improves a Hierarchical Reasoning Model (HRM) with enhanced capabilities for reasoning, instruction following, tool use, and error correction.

## Features

### 🧠 Enhanced Hierarchical Reasoning Model
- **Adaptive Computation Time (ACT)**: Dynamic reasoning steps with halt mechanism
- **Tool Use Integration**: Built-in function calling and tool selection capabilities
- **Scratchpad Memory**: Persistent memory across reasoning steps
- **Error Detection & Correction**: Self-monitoring and improvement mechanisms
- **Confidence Calibration**: Uncertainty estimation for better decision making

### 🔄 Autonomous Training System
- **Continuous Learning**: On-the-fly data collection and model updates
- **Self-Improvement**: Automatic strategy adjustment based on performance
- **Multi-objective Optimization**: Balances reasoning, efficiency, and accuracy
- **Adaptive Hyperparameters**: Dynamic learning rate and architecture adjustments
- **Human-in-the-loop**: Intervention system for critical errors

### 📊 Comprehensive Data Collection
- **Multi-source Aggregation**: Reasoning, instruction, tool use, and conversation data
- **Quality Filtering**: Automatic data cleaning and validation
- **Diversity Sampling**: Ensures balanced training distribution
- **Real-time Collection**: API integration for fresh data sources
- **Synthetic Generation**: Creates targeted training examples

### 🎯 Advanced Evaluation System
- **Multi-dimensional Assessment**: Reasoning, instruction following, tool use, error correction
- **Benchmark Integration**: GSM8K, MATH, Alpaca, and custom evaluations
- **Confidence Calibration**: Measures uncertainty alignment
- **Efficiency Metrics**: Reasoning steps vs. performance trade-offs
- **Continuous Monitoring**: Real-time performance tracking

### 🌐 Interactive Web Interface
- **Real-time Dashboard**: Live training progress and metrics
- **Configuration Management**: Dynamic parameter adjustment
- **Visualization**: Training curves, performance radar charts, data distribution
- **Control Panel**: Start/stop training, trigger evaluations, collect data
- **Alert System**: Notifications for critical events and interventions

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM
- 50GB+ storage space

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd hrm_autonomous_agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p data/{raw,processed,reasoning,instruction_following,tool_use,error_correction}
mkdir -p checkpoints outputs logs
```

### Configuration
```bash
# Generate default configuration
python main.py --save-config config/default.json

# Edit configuration as needed
nano config/default.json
```

## Usage

### Web Interface (Recommended)
```bash
# Start the web dashboard
python main.py --mode web --host 0.0.0.0 --port 5000

# Access dashboard at http://localhost:5000
```

### Command Line Interface

#### Autonomous Training
```bash
# Start autonomous training with default settings
python main.py --mode autonomous

# With custom configuration
python main.py --mode autonomous --config config/my_config.json

# With specific parameters
python main.py --mode autonomous \
    --batch-size 32 \
    --learning-rate 1e-5 \
    --epochs 5 \
    --max-time 48
```

#### Data Collection Only
```bash
# Collect training data
python main.py --mode collect
```

#### Model Evaluation Only
```bash
# Evaluate existing model
python main.py --mode evaluate --checkpoint checkpoints/best_model.pt
```

## Architecture

### Core Components

#### 1. Enhanced HRM Model (`core/hrm_model.py`)
- **HRMBlock**: Transformer block with RMSNorm and SwiGLU
- **ToolUseHead**: Function calling decision mechanism
- **ScratchpadModule**: Working memory for multi-step reasoning
- **Self-Evaluation**: Performance monitoring and improvement suggestions

#### 2. Autonomous Trainer (`training/autonomous_trainer.py`)
- **ContinuousDataset**: Dynamic dataset that grows during training
- **Performance Monitoring**: Real-time metrics and plateau detection
- **Self-Improvement Cycle**: Automatic strategy adjustments
- **Checkpoint Management**: Automatic saving and recovery

#### 3. Data Collector (`data_collection/data_collector.py`)
- **Multi-source Integration**: HuggingFace datasets, APIs, web scraping
- **Quality Assurance**: Content filtering and validation
- **Diversity Sampling**: Balanced data distribution
- **Synthetic Generation**: Targeted example creation

#### 4. Evaluator (`evaluation/evaluator.py`)
- **Comprehensive Metrics**: 7 evaluation dimensions
- **Benchmark Integration**: Standard and custom test sets
- **Confidence Assessment**: Uncertainty calibration
- **Trend Analysis**: Performance trajectory monitoring

### Training Process

1. **Initialization**: Load model, setup optimizer, initialize components
2. **Data Collection**: Gather diverse training examples from multiple sources
3. **Training Epoch**: Process batches with gradient accumulation and mixed precision
4. **Evaluation**: Assess performance across multiple dimensions
5. **Self-Improvement**: Analyze results and adjust strategies
6. **Checkpoint**: Save model state and training progress
7. **Monitoring**: Check for intervention needs or completion criteria

### Data Flow

```
Data Sources → Data Collector → Quality Filter → Continuous Dataset
                                                        ↓
Web Interface ← Status Updates ← Autonomous Trainer ← Model Training
                                        ↓
                                   Evaluator → Performance Metrics
```

## Configuration

### Training Parameters
```json
{
  "batch_size": 16,
  "learning_rate": 2e-5,
  "num_epochs": 3,
  "max_training_time": 24,
  "performance_threshold": 0.85,
  "mixed_precision": true,
  "gradient_accumulation_steps": 1
}
```

### Model Parameters
```json
{
  "d_model": 512,
  "n_heads": 8,
  "d_ff": 2048,
  "dropout": 0.1,
  "halt_max_steps": 8,
  "ponder_loss_weight": 1e-2,
  "num_tools": 100
}
```

### Data Collection
```json
{
  "max_daily_samples": 10000,
  "quality_threshold": 0.7,
  "data_mixing_ratios": {
    "reasoning": 0.4,
    "instruction": 0.3,
    "tool_use": 0.2,
    "error_correction": 0.1
  }
}
```

## Monitoring and Control

### Web Dashboard Features
- **Real-time Status**: Training progress, current metrics, system health
- **Interactive Charts**: Loss curves, performance radar, data distribution
- **Control Panel**: Start/stop training, trigger evaluations, collect data
- **Configuration**: Dynamic parameter adjustment
- **Logs**: Real-time system messages and error tracking
- **Alerts**: Notifications for critical events

### Performance Metrics
- **Reasoning Accuracy**: Mathematical and logical problem solving
- **Instruction Following**: Task completion and format adherence
- **Tool Use Success**: Function calling and tool selection accuracy
- **Error Correction**: Detection and fixing of mistakes
- **Overall Confidence**: Uncertainty calibration quality
- **Reasoning Efficiency**: Steps vs. performance trade-off
- **Response Quality**: Coherence, relevance, and completeness

## Advanced Features

### Self-Improvement Mechanisms
- **Performance Analysis**: Identifies weaknesses and improvement opportunities
- **Strategy Adjustment**: Modifies training approach based on results
- **Hyperparameter Adaptation**: Dynamic learning rate and architecture changes
- **Data Augmentation**: Generates targeted training examples
- **Error Pattern Recognition**: Learns from common mistakes

### Human-in-the-Loop
- **Intervention Triggers**: Automatic detection of critical issues
- **Manual Override**: Web interface controls for human guidance
- **Feedback Integration**: Incorporates human corrections and preferences
- **Safety Mechanisms**: Prevents harmful or incorrect outputs

### Deployment Options
- **Local Training**: Single machine with GPU acceleration
- **Cloud Integration**: Supports cloud-based training infrastructure
- **Distributed Training**: Multi-GPU and multi-node capabilities
- **Edge Deployment**: Optimized models for resource-constrained environments

## Troubleshooting

### Common Issues

#### CUDA Out of Memory
```bash
# Reduce batch size
python main.py --mode autonomous --batch-size 8

# Enable gradient checkpointing (add to config)
"gradient_checkpointing": true
```

#### Data Collection Failures
```bash
# Check internet connection and API keys
# Review logs for specific error messages
tail -f hrm_agent.log
```

#### Training Stagnation
- Check learning rate (may be too high/low)
- Verify data quality and diversity
- Review performance plateau settings
- Consider architecture adjustments

#### Web Interface Issues
```bash
# Check port availability
netstat -tulpn | grep :5000

# Restart with different port
python main.py --mode web --port 8080
```

### Performance Optimization

#### Memory Usage
- Use mixed precision training
- Implement gradient checkpointing
- Optimize batch size for your hardware
- Clear cache regularly

#### Training Speed
- Use multiple GPUs if available
- Optimize data loading with multiple workers
- Use compiled models (torch.compile)
- Profile bottlenecks with torch.profiler

## Contributing

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Format code
black .
flake8 .
```

### Adding New Features
1. **Data Sources**: Extend `DataCollector` with new collection methods
2. **Evaluation Metrics**: Add new assessment dimensions to `ModelEvaluator`
3. **Model Components**: Enhance `EnhancedHierarchicalReasoningModel`
4. **Training Strategies**: Modify `AutonomousTrainer` improvement cycles
5. **Web Interface**: Add new dashboard components and visualizations

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this system in your research, please cite:

```bibtex
@software{hrm_autonomous_agent,
  title={HRM Autonomous Agent: Continuous Learning System for Hierarchical Reasoning},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/hrm-autonomous-agent}
}
```

## Acknowledgments

- Based on the Hierarchical Reasoning Model architecture
- Inspired by Adaptive Computation Time mechanisms
- Built with PyTorch, Transformers, and modern ML tools
- Web interface powered by Flask and Plotly

## Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the documentation
- Review the troubleshooting guide
- Contact the development team

---

**Note**: This system is designed for research and educational purposes. Ensure proper safety measures and human oversight when deploying in production environments.