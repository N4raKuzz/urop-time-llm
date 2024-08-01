
import sys
from pathlib import Path
# Add the project root directory to the Python path
project_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, project_root)

import argparse
from typing import List
from src.trainers.experiment_config import ExperimentConfig
from src.utils.experiment_utils import load_config, calculate_average_metrics, print_metrics
from scripts.experiments.dynamic_prediction import DynamicPrediction
from ef_prediction import EFPrediction
from src.trainers.binary_classification import BinaryClassifier

def parse_seeds(seeds_str: str) -> List[int]:
    try:
        return [int(seed) for seed in seeds_str.split()]
    except ValueError:
        raise argparse.ArgumentTypeError("Seeds must be space-separated integers")

def run_experiment(config_path: str, seeds: List[int]):
    config_dict = load_config(config_path)
    config = ExperimentConfig(config_dict)

    if config.task == 'dynamic_prediction':
        experiment = DynamicPrediction(config)
    elif config.task == 'ef_prediction':
        experiment = EFPrediction(config)
    else:
        raise ValueError(f"Unsupported task: {config.task}")

    all_results = experiment.run_multiple_seeds(seeds)
    # Calculate and print average metrics
    avg_metrics = {}
    std_metrics = {}
    for model_name in all_results[0]['metrics'].keys():
        model_metrics = [r['metrics'][model_name] for r in all_results]
                
        # Separate numeric and non-numeric metrics
        numeric_metrics = []

        for metrics in model_metrics:
            numeric = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
            numeric_metrics.append(numeric)
    
        avg_metrics[model_name], std_metrics[model_name] = calculate_average_metrics(numeric_metrics)

    print("\nAverage metrics across all seeds:")
    for model_name in avg_metrics.keys():
        print(f"\nModel: {model_name}")
        for metric_name in avg_metrics[model_name].keys():
            avg = avg_metrics[model_name][metric_name]
            std = std_metrics[model_name][metric_name]
            print(f"  {metric_name}: {avg:.4f} ± {std:.4f}")

    print(f"Experiments completed. Results saved in {config.results_dir}")
    print(f"To view TensorBoard logs, run: tensorboard --logdir={config.log_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments with multiple seeds")
    parser.add_argument("config_path", type=str, help="Path to the experiment configuration file")
    parser.add_argument("--seeds", type=parse_seeds, default="42 123 456 789 1010", 
                        help="Seeds for the experiments (space-separated integers)")
    # parser.add_argument("--task", type=str, choices=['dynamic_prediction', 'binary_classification'],
    #                     default='binary_classification', help="Task to run")
    args = parser.parse_args()

    run_experiment(args.config_path, args.seeds)
