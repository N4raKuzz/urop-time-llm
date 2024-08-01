import yaml
from typing import Dict, Any
from src.utils.experiment_utils import load_config, process_config

class ExperimentConfig:
    def __init__(self, config: Dict[str, Any]):
        processed_config = process_config(config)
        self.seed = processed_config.get('seed')  # Seed is handled globally, here just for storage
        self.task = processed_config.get('task')
        self.results_dir = processed_config.get('results_dir')
        self.log_dir = processed_config['log_dir']
        self.data_config = processed_config['data_config']
        self.model_config = processed_config['model_config']
        self.train_config = processed_config['train_config']
        self.data_config['task'] = self.task

    @classmethod
    def from_yaml(cls, config_path: str):
        config = load_config(config_path)
        return cls(config)
