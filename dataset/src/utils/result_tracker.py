import json
import os
from datetime import datetime
import torch
import pickle

def make_serializable(obj):
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    elif hasattr(obj, '__dict__'):
        return make_serializable(obj.__dict__)
    else:
        return str(obj)

class ResultTracker:
    def __init__(self, results_dir='results'):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

    def save_result(self, result, config, seed):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = os.path.join(self.results_dir, f"{config.task}_{timestamp}_seed{seed}")
        os.makedirs(result_dir, exist_ok=True)

        # Save model
        if result.get('model') is not None:
            model = result['model']
            if isinstance(model, torch.nn.Module):
                torch.save(model.state_dict(), os.path.join(result_dir, 'model.pth'))
            else:
                with open(os.path.join(result_dir, 'models.pkl'), 'wb') as f:
                    pickle.dump(result['model'], f)            
            # else:
            #     print(f"Warning: Unknown model type {type(model)}. Model not saved.")

        # Save metrics
        metrics = {
            'main_metrics': result['metrics'],
            'sanity_check': result.get('sanity_check', {})
        }
        with open(os.path.join(result_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)

        # Save config
        serializable_config = make_serializable(config.__dict__)
        serializable_config['seed'] = seed
        with open(os.path.join(result_dir, 'config.json'), 'w') as f:
            json.dump(serializable_config, f, indent=2)

        # Save preprocessor
        torch.save(result['preprocessor'], os.path.join(result_dir, 'preprocessor.pth'))

    def get_all_results(self):
        all_results = []
        for result_dir in os.listdir(self.results_dir):
            result_path = os.path.join(self.results_dir, result_dir)
            if os.path.isdir(result_path):
                with open(os.path.join(result_path, 'metrics.json'), 'r') as f:
                    metrics = json.load(f)
                with open(os.path.join(result_path, 'config.json'), 'r') as f:
                    config = json.load(f)
                all_results.append({
                    'metrics': metrics,
                    'config': config,
                    'result_dir': result_path
                })
        return all_results