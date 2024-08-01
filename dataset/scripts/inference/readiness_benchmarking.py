import os
import sys
import torch
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.trainers.base_trainer import BaseTrainer
from src.metrics.classification_metrics import ClassificationMetrics
from src.utils.experiment_utils import load_config
from scripts.experiments.ef_prediction.main import EFPredictionDataset
from scripts.experiments.dynamic_prediction.main import DynamicPredictionDataset

def load_model(experiment_path: str, model_name: str):
    config = load_config(os.path.join(experiment_path, 'config.yaml'))
    if 'dynamic_prediction' in experiment_path:
        model_path = os.path.join(experiment_path, 'checkpoints', f'{model_name}.ckpt')
        model = BaseTrainer.load_from_checkpoint(model_path, config=config).to('cpu').model
        model.eval()
    elif 'ef_prediction' in experiment_path:
        model_path = os.path.join(experiment_path, 'checkpoints', f'{model_name}_best_model.joblib')
        model = joblib.load(model_path)
    else:
        raise ValueError("Unknown experiment type")
    return model, config

def run_inference(experiment_path: str, model_name: str, ef_indices: List[Tuple[str, int, int]], flip_prediction: bool):
    model, config = load_model(experiment_path, model_name)
    dataset_config = config['dataset']
    dataset_class = globals()[dataset_config['class']]
    config['data']['stride'] = 1
    test_dataset = dataset_class(config, data_fold='test')

    all_predictions = []
    all_targets = []
    ef_predictions = []
    ef_targets = []

    # Create sets for faster lookups
    ef_indices_set = set((subject_id, orig_idx) for subject_id, _, orig_idx in ef_indices)
    valid_indices_set = set((subject_id, orig_idx) for subject_id, _, orig_idx in test_dataset.valid_indices)

    # Find the union of valid indices across all datasets
    indices_to_process = list(ef_indices_set.union(valid_indices_set))

    # Collect all data and masks
    all_sequences = []
    all_targets = []
    task_mask = []
    ef_mask = []

    for subject_id, orig_idx in indices_to_process:
        sequence, target = test_dataset.get_data_by_index(subject_id, orig_idx)
        all_sequences.append(sequence)
        all_targets.append(target)
        task_mask.append((subject_id, orig_idx) in valid_indices_set)
        ef_mask.append((subject_id, orig_idx) in ef_indices_set)

    # Convert to appropriate format for batch processing
    if isinstance(model, torch.nn.Module):
        all_sequences = torch.stack(all_sequences, dim=0)
        all_targets = [t.numpy() for t in all_targets]
    else:
        all_sequences = np.vstack(all_sequences)

    all_targets = np.array(all_targets)

    # Batch inference
    if isinstance(model, torch.nn.Module):
        with torch.no_grad():
            all_predictions = model(all_sequences).squeeze().numpy()
    else:
        all_predictions = model.predict_proba(all_sequences.reshape(len(all_sequences), -1))[:, 1]

    # Apply masks
    task_mask = np.array(task_mask)
    ef_mask = np.array(ef_mask)

    task_predictions = all_predictions[task_mask]
    task_targets = all_targets[task_mask]

    ef_predictions = all_predictions[ef_mask]
    ef_targets = all_targets[ef_mask]

    # Handle prediction flipping for EF
    if flip_prediction:
        ef_predictions = 1 - ef_predictions
        ef_targets = 1 - ef_targets

    # Calculate metrics for all valid test data
    task_metrics = ClassificationMetrics.calculate_metrics(task_targets, task_predictions)
    
    # Calculate metrics for EF comparison
    ef_metrics = ClassificationMetrics.calculate_metrics(ef_targets, ef_predictions)

    return task_metrics, ef_metrics, task_predictions, task_targets, ef_predictions, ef_targets

if __name__ == "__main__":
    experiments = {
        'ef_prediction': {
            'experiment': './results/ef_prediction_12',
            'best_model': 'Random Forest',
            'flip_prediction': False
        },
        'dynamic_prediction': {
            'experiment': './results/dynamic_prediction_12',
            'best_model': 'model-epoch=68-val_loss=0.43',
            'flip_prediction': True
        }
    }

    # Load EF dataset to get indices
    config_path = os.path.join(experiments['ef_prediction']['experiment'], 'config.yaml')
    ef_config = load_config(config_path)
    ef_dataset = EFPredictionDataset(ef_config, data_fold='test')

    results = {}
    ef_comparison = {}
    for exp_name, exp_info in experiments.items():
        all_metrics, ef_metrics, _, _, _, _ = run_inference(
            exp_info['experiment'], 
            exp_info['best_model'], 
            ef_dataset.valid_indices, 
            exp_info['flip_prediction']
        )
        results[exp_name] = all_metrics
        ef_comparison[exp_name] = ef_metrics

    # Create DataFrames for comparison
    all_metrics_df = pd.DataFrame(results).T
    ef_comparison_df = pd.DataFrame(ef_comparison).T

    print("Metrics on all test data:")
    print(all_metrics_df)
    print("\nComparison of metrics on EF data:")
    print(ef_comparison_df)
