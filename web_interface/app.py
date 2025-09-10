"""
Web Interface for HRM Autonomous Agent
Interactive dashboard for monitoring and controlling the training process
"""
import os
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import threading

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
import plotly.graph_objs as go
import plotly.utils
import pandas as pd

try:
    from ..training.autonomous_trainer import AutonomousTrainer, TrainingConfig
    from ..data_collection.data_collector import DataCollector
    from ..evaluation.evaluator import ModelEvaluator
    from ..core.hrm_model import EnhancedHierarchicalReasoningModel
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from training.autonomous_trainer import AutonomousTrainer, TrainingConfig
    from data_collection.data_collector import DataCollector
    from evaluation.evaluator import ModelEvaluator
    from core.hrm_model import EnhancedHierarchicalReasoningModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hrm_autonomous_agent_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global state
trainer = None
training_thread = None
training_active = False
system_status = {
    "status": "idle",
    "last_update": datetime.now().isoformat(),
    "training_progress": 0,
    "current_epoch": 0,
    "global_step": 0,
    "performance_metrics": {},
    "data_collection_stats": {},
    "error_messages": []
}

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/status')
def get_status():
    """Get current system status"""
    global trainer, system_status
    
    if trainer:
        training_status = trainer.get_training_status()
        system_status.update({
            "global_step": training_status["global_step"],
            "current_epoch": training_status["current_epoch"],
            "best_performance": training_status["best_performance"],
            "performance_metrics": training_status["learning_objectives"],
            "dataset_size": training_status["dataset_size"],
            "improvement_strategies": training_status["improvement_strategies"],
            "performance_plateau_count": training_status["performance_plateau_count"]
        })
    
    return jsonify(system_status)

@app.route('/api/start_training', methods=['POST'])
def start_training():
    """Start autonomous training"""
    global trainer, training_thread, training_active
    
    try:
        if training_active:
            return jsonify({"error": "Training already active"}), 400
        
        # Get configuration from request
        config_data = request.json or {}
        
        # Create training configuration
        config = TrainingConfig(
            batch_size=config_data.get('batch_size', 16),
            learning_rate=config_data.get('learning_rate', 2e-5),
            num_epochs=config_data.get('num_epochs', 3),
            max_training_time=config_data.get('max_training_time', 24),
            performance_threshold=config_data.get('performance_threshold', 0.85)
        )
        
        # Model configuration
        model_config = {
            "d_model": config_data.get('d_model', 512),
            "n_heads": config_data.get('n_heads', 8),
            "d_ff": config_data.get('d_ff', 2048),
            "dropout": config_data.get('dropout', 0.1),
            "halt_max_steps": config_data.get('halt_max_steps', 8),
            "ponder_loss_weight": config_data.get('ponder_weight', 1e-2),
            "num_tools": config_data.get('num_tools', 100)
        }
        
        # Initialize trainer
        trainer = AutonomousTrainer(config, model_config)
        trainer.initialize_model(config_data.get('checkpoint_path'))
        
        # Start training in background thread
        training_thread = threading.Thread(target=run_training_loop)
        training_thread.daemon = True
        training_thread.start()
        
        training_active = True
        system_status["status"] = "training"
        system_status["last_update"] = datetime.now().isoformat()
        
        return jsonify({"message": "Training started successfully"})
    
    except Exception as e:
        logger.error(f"Error starting training: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/stop_training', methods=['POST'])
def stop_training():
    """Stop autonomous training"""
    global training_active, system_status
    
    training_active = False
    system_status["status"] = "stopping"
    system_status["last_update"] = datetime.now().isoformat()
    
    return jsonify({"message": "Training stop requested"})

@app.route('/api/collect_data', methods=['POST'])
def trigger_data_collection():
    """Trigger manual data collection"""
    try:
        collector = DataCollector()
        
        # Run data collection in background
        def collect_data():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            collected_data = loop.run_until_complete(collector.collect_all_data())
            stats = collector.get_collection_stats()
            
            system_status["data_collection_stats"] = stats
            system_status["last_update"] = datetime.now().isoformat()
            
            # Emit update to connected clients
            socketio.emit('data_collection_complete', stats)
        
        thread = threading.Thread(target=collect_data)
        thread.daemon = True
        thread.start()
        
        return jsonify({"message": "Data collection started"})
    
    except Exception as e:
        logger.error(f"Error triggering data collection: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/evaluate_model', methods=['POST'])
def trigger_evaluation():
    """Trigger model evaluation"""
    global trainer
    
    try:
        if not trainer or not trainer.model:
            return jsonify({"error": "No model available for evaluation"}), 400
        
        def run_evaluation():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            evaluator = ModelEvaluator()
            results = loop.run_until_complete(
                evaluator.comprehensive_evaluation(trainer.model, trainer.tokenizer)
            )
            
            system_status["performance_metrics"] = results
            system_status["last_update"] = datetime.now().isoformat()
            
            # Emit results to connected clients
            socketio.emit('evaluation_complete', results)
        
        thread = threading.Thread(target=run_evaluation)
        thread.daemon = True
        thread.start()
        
        return jsonify({"message": "Evaluation started"})
    
    except Exception as e:
        logger.error(f"Error triggering evaluation: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/training_history')
def get_training_history():
    """Get training history for visualization"""
    global trainer
    
    if not trainer:
        return jsonify({"error": "No trainer available"}), 400
    
    history = trainer.training_history
    
    # Prepare data for plotting
    epochs = [h["epoch"] for h in history]
    losses = [h["loss"] for h in history]
    timestamps = [h["timestamp"] for h in history]
    
    # Create plotly figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epochs,
        y=losses,
        mode='lines+markers',
        name='Training Loss',
        line=dict(color='blue', width=2),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title='Training Loss Over Time',
        xaxis_title='Epoch',
        yaxis_title='Loss',
        hovermode='x unified'
    )
    
    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return jsonify({
        "graph": graph_json,
        "history": history
    })

@app.route('/api/performance_metrics')
def get_performance_metrics():
    """Get performance metrics visualization"""
    global trainer
    
    if not trainer:
        return jsonify({"error": "No trainer available"}), 400
    
    metrics = trainer.learning_objectives
    
    # Create radar chart for performance metrics
    categories = list(metrics.keys())
    values = list(metrics.values())
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Current Performance'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Model Performance Metrics"
    )
    
    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return jsonify({"graph": graph_json, "metrics": metrics})

@app.route('/api/data_stats')
def get_data_stats():
    """Get data collection statistics"""
    try:
        collector = DataCollector()
        stats = collector.get_collection_stats()
        
        # Create visualization of data distribution
        data_types = list(stats.get("data_types", {}).keys())
        sample_counts = [stats["data_types"][dt]["samples"] for dt in data_types]
        
        fig = go.Figure(data=[
            go.Bar(x=data_types, y=sample_counts, name='Sample Counts')
        ])
        
        fig.update_layout(
            title='Data Distribution by Type',
            xaxis_title='Data Type',
            yaxis_title='Number of Samples'
        )
        
        graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        return jsonify({
            "graph": graph_json,
            "stats": stats
        })
    
    except Exception as e:
        logger.error(f"Error getting data stats: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/model_config', methods=['GET', 'POST'])
def model_config():
    """Get or update model configuration"""
    global trainer
    
    if request.method == 'GET':
        if trainer:
            return jsonify(trainer.model_config)
        else:
            return jsonify({
                "d_model": 512,
                "n_heads": 8,
                "d_ff": 2048,
                "dropout": 0.1,
                "halt_max_steps": 8,
                "ponder_loss_weight": 1e-2,
                "num_tools": 100
            })
    
    elif request.method == 'POST':
        try:
            new_config = request.json
            
            if trainer:
                # Update existing trainer configuration
                trainer.model_config.update(new_config)
                return jsonify({"message": "Configuration updated"})
            else:
                return jsonify({"message": "Configuration saved for next training session"})
        
        except Exception as e:
            return jsonify({"error": str(e)}), 500

@app.route('/api/logs')
def get_logs():
    """Get recent log messages"""
    # In a real implementation, you'd read from log files
    # For now, return system status messages
    logs = system_status.get("error_messages", [])
    return jsonify({"logs": logs[-100:]})  # Last 100 log entries

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    emit('status_update', system_status)

@socketio.on('request_status')
def handle_status_request():
    """Handle status request from client"""
    emit('status_update', system_status)

def run_training_loop():
    """Run the autonomous training loop in background"""
    global trainer, training_active, system_status
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Run training with periodic status updates
        async def training_with_updates():
            training_start_time = datetime.now()
            max_training_duration = timedelta(hours=trainer.config.max_training_time)
            
            while training_active and datetime.now() - training_start_time < max_training_duration:
                try:
                    # Collect new data
                    await trainer._collect_and_update_data()
                    
                    # Evaluate current performance
                    current_performance = await trainer._evaluate_model()
                    
                    # Update system status
                    system_status.update({
                        "performance_metrics": current_performance,
                        "last_update": datetime.now().isoformat()
                    })
                    
                    # Emit status update
                    socketio.emit('status_update', system_status)
                    
                    # Check if training should continue
                    if trainer._should_continue_training(current_performance):
                        # Perform training epoch
                        await trainer._training_epoch()
                        
                        # Self-improvement cycle
                        await trainer._self_improvement_cycle()
                        
                        # Save checkpoint
                        trainer._save_checkpoint()
                        
                        # Update status
                        training_status = trainer.get_training_status()
                        system_status.update({
                            "global_step": training_status["global_step"],
                            "current_epoch": training_status["current_epoch"],
                            "best_performance": training_status["best_performance"],
                            "dataset_size": training_status["dataset_size"]
                        })
                        
                        # Emit training progress
                        socketio.emit('training_progress', {
                            "epoch": training_status["current_epoch"],
                            "step": training_status["global_step"],
                            "performance": training_status["best_performance"]
                        })
                    
                    else:
                        logger.info("Performance threshold met, entering monitoring mode...")
                        await asyncio.sleep(3600)  # Wait 1 hour
                    
                    # Check for fatal errors
                    if trainer._needs_human_intervention():
                        system_status["status"] = "intervention_needed"
                        system_status["error_messages"].append({
                            "timestamp": datetime.now().isoformat(),
                            "level": "CRITICAL",
                            "message": "Human intervention required!"
                        })
                        socketio.emit('intervention_needed', {
                            "message": "Human intervention required!"
                        })
                        break
                
                except Exception as e:
                    logger.error(f"Error in training loop: {e}")
                    system_status["error_messages"].append({
                        "timestamp": datetime.now().isoformat(),
                        "level": "ERROR",
                        "message": str(e)
                    })
                    
                    if trainer._is_fatal_error(e):
                        system_status["status"] = "error"
                        socketio.emit('fatal_error', {"message": str(e)})
                        break
                    else:
                        await asyncio.sleep(300)  # Wait 5 minutes before retry
            
            # Training completed
            system_status["status"] = "completed"
            system_status["last_update"] = datetime.now().isoformat()
            socketio.emit('training_complete', system_status)
        
        loop.run_until_complete(training_with_updates())
    
    except Exception as e:
        logger.error(f"Fatal error in training loop: {e}")
        system_status["status"] = "error"
        system_status["error_messages"].append({
            "timestamp": datetime.now().isoformat(),
            "level": "FATAL",
            "message": str(e)
        })
        socketio.emit('fatal_error', {"message": str(e)})
    
    finally:
        training_active = False

if __name__ == '__main__':
    # Create templates directory and basic template
    os.makedirs('templates', exist_ok=True)
    
    # Run the Flask app
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)