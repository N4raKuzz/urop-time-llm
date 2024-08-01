import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
sys.dont_write_bytecode = True

import pandas as pd
import numpy as np
import yaml
from typing import Dict, List
import pytorch_lightning as pl
from src.data.ventilation_dataset import VentilationDataset
from src.trainers.base_trainer import BaseTrainer
from src.utils.experiment_utils import print_header

class DynamicPredictionDataset(VentilationDataset):
    def __init__(self, config: Dict, data_fold: str = 'train'):
        super().__init__(config, data_fold)

    def filter_data(self, data: pd.DataFrame) -> np.ndarray:
        mask = np.ones(len(data), dtype=bool)
        
        # Filter out subjects failed first extubation
        censoring_mask = (((data.reintubation_after_first_extub==1) | 
                        (data.NIV_4h_after_first_extub==1) | 
                        (data.HF_4h_after_first_extub==1)) | 
                        data['event_description'].str.startswith('Not Extubated'))
        # censoring_mask = ((data.reintubation_after_first_extub==1) | data['event_description'].str.startswith('Not Extubated'))
                
        if censoring_mask.any():
            return np.zeros(len(data), dtype=bool)
        
        # Keep only data between vent_start_time and first_extubation_time
        mask &= (data.index >= data['vent_start_time'].iloc[0])
        mask &= (data.index < data['first_extubation_time'].iloc[0])
        
        return mask

    def compute_targets(self, data: pd.DataFrame) -> pd.DataFrame:
        delta_t = 24
        data[f'successful_extubation_next_{delta_t}h'] = (
            (data.index >= (data['vent_end_time'] - delta_t)) &
            (data['event_description'].isin(['No 48h Observation - Alive', 'Successful Extubation']) &
            (data.reintubation_after_first_extub!=1) 
            & (data.NIV_4h_after_first_extub!=1) & (data.HF_4h_after_first_extub!=1)
            )
        ).astype(int)
        return data

def run_experiment(config: Dict):
    data_seed = int(config['data']['preprocessed_dir'].split('_seed')[-1])

    # Update config seed if it doesn't match data seed
    if 'seed' not in config or config['seed'] != data_seed:
        print_header(f"Updating seed in config to match data seed: {data_seed}")
        config['seed'] = data_seed

    # Set up seed for reproducibility
    pl.seed_everything(config['seed'])

    # Create experiment directory
    experiment_dir = os.path.join(config['logging']['base_dir'], f"{config['experiment_name']}_{config['seed']}")
    os.makedirs(experiment_dir, exist_ok=False)     

    # Update log directory in config
    config['logging']['log_dir'] = experiment_dir

    # Save config
    with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as file:
        yaml.dump(config, file)

    # Run experiment
    trainer = BaseTrainer(config)
    trainer.run()

def update_config(base_config: Dict, params: Dict) -> Dict:
    config = base_config.copy()
    for key, value in params.items():
        if key == 'experiment_name':
            config['experiment_name'] = value
        else:
            keys = key.split('.')
            d = config
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            if value is None:
                d.pop(keys[-1], None)
            else:
                d[keys[-1]] = value
    return config

if __name__ == "__main__":
    # Automatically find and load the config file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, 'config.yaml')
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, 'r') as file:
        base_config = yaml.safe_load(file)

    import argparse
    parser = argparse.ArgumentParser(description="Run dynamic prediction experiment.")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning", default=True)
    args = parser.parse_args()

    if args.tune:
        param_grid = [
            {
                'experiment_name': 'dynamic_prediction_NIV_window_1', 
                'data.window_size': 1,
                'model.params.encoder': None,
                'model.params.classifier.input_dim': 43
                },
            {
                'experiment_name': 'dynamic_prediction_NIV_window_8',
                'data.window_size': 8,
                },
            {
                'experiment_name': 'dynamic_prediction_NIV_window_24',
                'data.window_size': 24,
                }
        ]
        
        for params in param_grid:
            print_header(f"Running experiment: {params['experiment_name']}")
            config = update_config(base_config, params)
            run_experiment(config)
    else:
        run_experiment(base_config)