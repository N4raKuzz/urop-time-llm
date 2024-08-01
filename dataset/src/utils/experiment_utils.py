import yaml
import torch
import random
import numpy as np
from typing import Dict, Any, List, Tuple
import torch.nn as nn
import torch.optim as optim

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def process_config(config: Dict[str, Any]) -> Dict[str, Any]:
    # Deep copy the config to avoid modifying the original
    processed_config = {**config}
    
    # Process train config
    train_config = processed_config['train_config']
    loss_func_name = train_config['loss_fn']
    train_config['loss_fn'] = getattr(nn, loss_func_name)()
    optimizer_class = train_config['optimizer']['class']
    train_config['optimizer']['class'] = getattr(optim, optimizer_class)
    train_config['checkpoint_dir'] = processed_config['checkpoint_dir']
    train_config['log_dir'] = processed_config['log_dir']
    
    return processed_config

def sanitize_conifgs(config):
    sanitized = {}
    for key, value in config.items():
        if isinstance(value, (int, float, str, bool)):
            sanitized[key] = value
        elif isinstance(value, torch.Tensor):
            sanitized[key] = value.item()
        else:
            sanitized[key] = str(value)
    return sanitized

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_average_metrics(all_metrics: List[Dict[str, float]]) -> Tuple[Dict[str, float], Dict[str, float]]:
    avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0].keys()}
    std_metrics = {k: np.std([m[k] for m in all_metrics]) for k in all_metrics[0].keys()}
    return avg_metrics, std_metrics

def print_metrics(avg_metrics: Dict[str, float], std_metrics: Dict[str, float]):
    print("Average metrics:")
    for k, v in avg_metrics.items():
        print(f"{k}: {v:.4f} ± {std_metrics[k]:.4f}")

def print_header(message, PRINT_WIDTH=80):
    padding = (PRINT_WIDTH - len(message) - 2) // 2  # -2 for the spaces around the message
    print('=' * padding + f' {message} ' + '=' * padding)