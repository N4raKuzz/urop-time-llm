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

def run_inference(model, test_dataset, indices_to_process):
    all_sequences = []
    all_targets = []

    for subject_id, orig_idx in indices_to_process:
        sequence, target = test_dataset.get_data_by_index(subject_id, orig_idx)
        all_sequences.append(sequence)
        all_targets.append(target)

    if isinstance(model, torch.nn.Module):
        all_sequences = torch.stack(all_sequences, dim=0)
        with torch.no_grad():
            all_predictions = model(all_sequences).squeeze().numpy()
    else:
        all_sequences = np.vstack(all_sequences)
        all_predictions = model.predict_proba(all_sequences.reshape(len(all_sequences), -1))[:, 1]

    all_targets = np.array([t.numpy() if isinstance(t, torch.Tensor) else t for t in all_targets])

    return all_predictions, all_targets

def evaluate_on_task(predictions, targets, flip_prediction):
    if flip_prediction:
        predictions = 1 - predictions
        targets = 1 - targets
    return ClassificationMetrics.calculate_metrics(targets, predictions)

if __name__ == "__main__":
    output_file_path = './results/inferences_readiness_comparisons_NIV_aligned.txt'

    experiments = {
        'ef_prediction_baseline': {
            'experiment': './results/ef_prediction_NIV_window_1_12',
            'best_model': 'Random Forest',
            'dataset_class': EFPredictionDataset,
        },
        'ef_prediction_8hrs': {
            'experiment': './results/ef_prediction_NIV_window_8_12',
            'best_model': 'Random Forest',
            'dataset_class': EFPredictionDataset,
        },
        'ef_prediction_24hrs': {
            'experiment': './results/ef_prediction_NIV_window_24_12',
            'best_model': 'Random Forest',
            'dataset_class': EFPredictionDataset,
        },
        'dynamic_prediction_baseline': {
            'experiment': './results/dynamic_prediction_NIV_window_1_12',
            'best_model': 'model-epoch=62-val_loss=0.45',
            'dataset_class': DynamicPredictionDataset,
        },
        'dynamic_prediction_8hrs': {
            'experiment': './results/dynamic_prediction_NIV_window_8_12',
            'best_model': 'model-epoch=53-val_loss=0.45',
            'dataset_class': DynamicPredictionDataset,
        },
        'dynamic_prediction_24hrs': {
            'experiment': './results/dynamic_prediction_NIV_window_24_12',
            'best_model': 'model-epoch=64-val_loss=0.45',
            'dataset_class': DynamicPredictionDataset,
        }
    }

    # Load all datasets and get their indices
    datasets = {}
    task_indices = {}
    all_indices = set()
    for exp_name, exp_info in experiments.items():
        config = load_config(os.path.join(exp_info['experiment'], 'config.yaml'))
        config['data']['stride'] = 1  # Ensure stride is 1 for all datasets
        datasets[exp_name] = exp_info['dataset_class'](config, data_fold='test')
        task_indices[exp_name] = set((subject_id, orig_idx) for subject_id, _, orig_idx in datasets[exp_name].valid_indices)
        all_indices.update(task_indices[exp_name])

    # Run inference for all models on the union of all valid indices
    results = {}
    for exp_name, exp_info in experiments.items():
        model, _ = load_model(exp_info['experiment'], exp_info['best_model'])
        indices_to_process = sorted(list(all_indices))
        predictions, targets = run_inference(model, datasets[exp_name], indices_to_process)
        results[exp_name] = (predictions, targets, indices_to_process)

    # Evaluate each model on each task
    task_results = {task: {} for task in experiments.keys()}
    for task_name, task_dataset in datasets.items():
        task_valid_indices = task_indices[task_name]
        for model_name, (predictions, targets, model_indices) in results.items():
            # Create a mask for the task-specific indices
            task_mask = [idx in task_valid_indices for idx in model_indices]
            
            task_predictions = predictions[task_mask]
            task_targets = targets[task_mask]

            # Define flip_prediction logic here (hardcoded for now)
            flip_prediction = (task_name.startswith('ef_prediction') and model_name.startswith('dynamic_prediction')) or \
                              (task_name.startswith('dynamic_prediction') and model_name.startswith('ef_prediction'))
            
            metrics = evaluate_on_task(task_predictions, task_targets, flip_prediction)
            task_results[task_name][model_name] = metrics

    # Create and print DataFrames for each task
    for task_name, task_result in task_results.items():
        df = pd.DataFrame(task_result).T
        print(f"\nMetrics for {task_name} task:")
        print(df)

# Open the file in write mode
with open(output_file_path, 'w') as file:
    # Create and write DataFrames for each task to the file
    for task_name, task_result in task_results.items():
        df = pd.DataFrame(task_result).T
        file.write(f"\nMetrics for {task_name} task:\n")
        file.write(df.to_string())
        file.write("\n")  # Add a newline for better readability between tasks

# Example print statement for confirmation (optional)
print(f"Metrics have been written to {output_file_path}")

