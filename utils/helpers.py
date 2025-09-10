"""
Utility functions and helpers for the HRM Autonomous Agent
"""
import os
import json
import logging
import random
import numpy as np
import torch
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import hashlib

logger = logging.getLogger(__name__)

def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device(device_str: str = "auto") -> torch.device:
    """Get the appropriate device for training"""
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using MPS device (Apple Silicon)")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU device")
    else:
        device = torch.device(device_str)
        logger.info(f"Using specified device: {device}")
    
    return device

def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total": total_params,
        "trainable": trainable_params,
        "non_trainable": total_params - trainable_params
    }

def format_number(num: Union[int, float]) -> str:
    """Format large numbers with appropriate suffixes"""
    if num >= 1e9:
        return f"{num/1e9:.1f}B"
    elif num >= 1e6:
        return f"{num/1e6:.1f}M"
    elif num >= 1e3:
        return f"{num/1e3:.1f}K"
    else:
        return str(num)

def calculate_model_size(model: torch.nn.Module) -> Dict[str, Union[int, str]]:
    """Calculate model size in bytes and formatted string"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    total_size = param_size + buffer_size
    
    return {
        "param_size_bytes": param_size,
        "buffer_size_bytes": buffer_size,
        "total_size_bytes": total_size,
        "param_size_mb": param_size / (1024 ** 2),
        "buffer_size_mb": buffer_size / (1024 ** 2),
        "total_size_mb": total_size / (1024 ** 2),
        "formatted_size": format_size(total_size)
    }

def format_size(size_bytes: int) -> str:
    """Format byte size to human readable string"""
    if size_bytes >= 1024**3:
        return f"{size_bytes / (1024**3):.1f} GB"
    elif size_bytes >= 1024**2:
        return f"{size_bytes / (1024**2):.1f} MB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes} B"

def save_json(data: Dict[str, Any], filepath: str, indent: int = 2):
    """Save data to JSON file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent, default=str)

def load_json(filepath: str) -> Dict[str, Any]:
    """Load data from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def create_timestamp() -> str:
    """Create timestamp string for filenames"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def parse_timestamp(timestamp_str: str) -> datetime:
    """Parse timestamp string back to datetime"""
    return datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")

def time_since(timestamp: Union[str, datetime]) -> str:
    """Get human readable time since timestamp"""
    if isinstance(timestamp, str):
        timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
    
    now = datetime.now()
    if timestamp.tzinfo is not None:
        now = now.replace(tzinfo=timestamp.tzinfo)
    
    delta = now - timestamp
    
    if delta.days > 0:
        return f"{delta.days} days ago"
    elif delta.seconds > 3600:
        hours = delta.seconds // 3600
        return f"{hours} hours ago"
    elif delta.seconds > 60:
        minutes = delta.seconds // 60
        return f"{minutes} minutes ago"
    else:
        return "Just now"

def hash_string(text: str) -> str:
    """Create hash of string for deduplication"""
    return hashlib.md5(text.encode()).hexdigest()

def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split list into chunks of specified size"""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """Flatten nested dictionary"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def unflatten_dict(d: Dict[str, Any], sep: str = '.') -> Dict[str, Any]:
    """Unflatten dictionary with separator"""
    result = {}
    for key, value in d.items():
        parts = key.split(sep)
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    return result

def merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dictionaries"""
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result

def get_memory_usage() -> Dict[str, Union[int, str]]:
    """Get current memory usage"""
    import psutil
    
    process = psutil.Process()
    memory_info = process.memory_info()
    
    result = {
        "rss_bytes": memory_info.rss,
        "vms_bytes": memory_info.vms,
        "rss_mb": memory_info.rss / (1024 ** 2),
        "vms_mb": memory_info.vms / (1024 ** 2),
        "rss_formatted": format_size(memory_info.rss),
        "vms_formatted": format_size(memory_info.vms)
    }
    
    # Add GPU memory if available
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated()
        gpu_memory_cached = torch.cuda.memory_reserved()
        
        result.update({
            "gpu_allocated_bytes": gpu_memory,
            "gpu_cached_bytes": gpu_memory_cached,
            "gpu_allocated_mb": gpu_memory / (1024 ** 2),
            "gpu_cached_mb": gpu_memory_cached / (1024 ** 2),
            "gpu_allocated_formatted": format_size(gpu_memory),
            "gpu_cached_formatted": format_size(gpu_memory_cached)
        })
    
    return result

def cleanup_old_files(directory: str, max_age_days: int = 7, pattern: str = "*"):
    """Clean up old files in directory"""
    import glob
    from pathlib import Path
    
    cutoff_time = datetime.now() - timedelta(days=max_age_days)
    pattern_path = os.path.join(directory, pattern)
    
    removed_count = 0
    for filepath in glob.glob(pattern_path):
        file_path = Path(filepath)
        if file_path.is_file():
            file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
            if file_time < cutoff_time:
                try:
                    file_path.unlink()
                    removed_count += 1
                    logger.debug(f"Removed old file: {filepath}")
                except Exception as e:
                    logger.warning(f"Could not remove file {filepath}: {e}")
    
    if removed_count > 0:
        logger.info(f"Cleaned up {removed_count} old files from {directory}")

def ensure_directory(path: str):
    """Ensure directory exists, create if not"""
    os.makedirs(path, exist_ok=True)

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division that returns default if denominator is zero"""
    return numerator / denominator if denominator != 0 else default

def moving_average(values: List[float], window_size: int) -> List[float]:
    """Calculate moving average of values"""
    if len(values) < window_size:
        return values
    
    result = []
    for i in range(len(values)):
        start_idx = max(0, i - window_size + 1)
        window = values[start_idx:i + 1]
        result.append(sum(window) / len(window))
    
    return result

def exponential_moving_average(values: List[float], alpha: float = 0.1) -> List[float]:
    """Calculate exponential moving average"""
    if not values:
        return []
    
    result = [values[0]]
    for i in range(1, len(values)):
        ema = alpha * values[i] + (1 - alpha) * result[-1]
        result.append(ema)
    
    return result

def calculate_metrics(predictions: List[float], targets: List[float]) -> Dict[str, float]:
    """Calculate common regression metrics"""
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have same length")
    
    n = len(predictions)
    if n == 0:
        return {}
    
    # Convert to numpy arrays
    pred = np.array(predictions)
    true = np.array(targets)
    
    # Calculate metrics
    mae = np.mean(np.abs(pred - true))
    mse = np.mean((pred - true) ** 2)
    rmse = np.sqrt(mse)
    
    # R-squared
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "count": n
    }

def log_system_info():
    """Log system information"""
    import platform
    import psutil
    
    logger.info("=== System Information ===")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python: {platform.python_version()}")
    logger.info(f"CPU: {platform.processor()}")
    logger.info(f"CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
    
    memory = psutil.virtual_memory()
    logger.info(f"RAM: {format_size(memory.total)} total, {format_size(memory.available)} available")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU Memory: {format_size(torch.cuda.get_device_properties(0).total_memory)}")
    else:
        logger.info("CUDA: Not available")
    
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info("=" * 30)

class Timer:
    """Context manager for timing code execution"""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.now()
        duration = self.end_time - self.start_time
        logger.info(f"{self.name} completed in {duration.total_seconds():.2f} seconds")
    
    @property
    def elapsed(self) -> Optional[timedelta]:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

class ProgressTracker:
    """Track progress of long-running operations"""
    
    def __init__(self, total: int, name: str = "Progress"):
        self.total = total
        self.name = name
        self.current = 0
        self.start_time = datetime.now()
    
    def update(self, increment: int = 1):
        self.current += increment
        
        if self.current % max(1, self.total // 20) == 0:  # Log every 5%
            self.log_progress()
    
    def log_progress(self):
        elapsed = datetime.now() - self.start_time
        progress_pct = (self.current / self.total) * 100
        
        if self.current > 0:
            eta = elapsed * (self.total - self.current) / self.current
            logger.info(f"{self.name}: {progress_pct:.1f}% ({self.current}/{self.total}) - ETA: {eta}")
        else:
            logger.info(f"{self.name}: {progress_pct:.1f}% ({self.current}/{self.total})")
    
    def finish(self):
        elapsed = datetime.now() - self.start_time
        logger.info(f"{self.name} completed: {self.total} items in {elapsed.total_seconds():.2f} seconds")