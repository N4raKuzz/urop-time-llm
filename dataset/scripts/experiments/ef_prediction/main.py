import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
import yaml
from typing import Dict, Any, Tuple, List
import pytorch_lightning as pl
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, average_precision_score
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from src.data.ventilation_dataset import VentilationDataset
from src.trainers.base_trainer import BaseTrainer
from src.models.classic_ml import get_classic_ml_models
from src.metrics.classification_metrics import ClassificationMetrics
from src.utils.experiment_utils import print_header
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils.class_weight import compute_sample_weight
import joblib

class EFPredictionDataset(VentilationDataset):
    def __init__(self, config: Dict, data_fold: str = 'train'):
        super().__init__(config, data_fold)

    def filter_data(self, data: pd.DataFrame) -> np.ndarray:
        mask = np.ones(len(data), dtype=bool)
        
        cutoff_start_time = max(data['first_extubation_time'].iloc[0] - 1, data['vent_start_time'].iloc[0])
        cutoff_end_time = data['first_extubation_time'].iloc[0]
        
        # Keep only data between vent_start_time and first_extubation_time
        mask &= (data.index >= cutoff_start_time)
        mask &= (data.index < cutoff_end_time)
        
        return mask

    def compute_targets(self, data: pd.DataFrame) -> pd.DataFrame:
        delta_t = 24
        data[f'successful_extubation_next_{delta_t}h'] = (
            (data.index >= (data['vent_end_time'] - delta_t)) &
            (data['event_description'].isin(['No 48h Observation - Alive', 'Successful Extubation']) &
            (data.reintubation_after_first_extub!=1) 
            # & (data.NIV_4h_after_first_extub!=1) & (data.HF_4h_after_first_extub!=1)
            )
        ).astype(int)
        return data

    def __getitem__(self, idx: int):
        subject_id, seq_idx, _ = self.valid_indices[idx]
        sequence, target, _, _ = self.all_data[subject_id][seq_idx]
        return sequence.flatten(), 1-target

    def get_data_by_index(self, subject_id: str, orig_idx: int) -> Tuple[np.array, np.array]:
        for _, (sequence, target, _, seq_orig_idx) in enumerate(self.all_data[subject_id]):
            if seq_orig_idx == orig_idx:
                return sequence.flatten(), 1-target
        raise ValueError(f"No data found for subject {subject_id} at index {orig_idx}")

class EFTrainer(BaseTrainer):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = self.setup_model()
        self.prepare_data()

    def setup_model(self):
        self.scorer = make_scorer(average_precision_score, needs_proba=True)
        return get_classic_ml_models()

    def _get_numpy_data(self, dataset):
        X, y = zip(*[dataset[i] for i in range(len(dataset))])
        return np.vstack(X), np.hstack(y)

    def prepare_data(self):
        super().prepare_data()
        X_train, y_train = self._get_numpy_data(self.train_dataset)
        X_val, y_val = self._get_numpy_data(self.val_dataset)
        self.X_train_val = np.vstack((X_train, X_val))
        self.y_train_val = np.hstack((y_train, y_val))

        if self.config['data'].get('resampling'):
            self.X_train_val, self.y_train_val = self._apply_resampling(self.X_train_val, self.y_train_val)

        if self.config['data'].get('feature_selection'):
            self.X_train_val, selected_features = self._apply_feature_selection(self.X_train_val, self.y_train_val)
            self.config['data']['feature_columns'] = selected_features
            print(f"Selected features: {selected_features}")

    def _apply_resampling(self, X, y):
        resampling_config = self.config['data']['resampling']
        if resampling_config['method'] == 'undersampling':
            resampler = RandomUnderSampler(sampling_strategy=resampling_config['ratio'])
        elif resampling_config['method'] == 'oversampling':
            resampler = RandomOverSampler(sampling_strategy=resampling_config['ratio'])
        return resampler.fit_resample(X, y)

    def _apply_feature_selection(self, X, y):
        selector = RFECV(estimator=LogisticRegression(max_iter=5000), 
                         min_features_to_select=30, 
                         scoring=self.scorer, 
                         step=1, 
                         cv=5)
        X_selected = selector.fit_transform(X, y)
        selected_features = [self.train_dataset.feature_columns[i] for i in range(len(selector.support_)) if selector.support_[i]]
        return X_selected, selected_features

    def run(self):
        print_header('Starting training')

        best_models = {}
        summary = []

        for name, (model, params) in self.model.items():
            print(f"Training {name}...")
            
            grid_search = GridSearchCV(model(), params, cv=5, scoring=self.scorer, n_jobs=-1)
            
            if self.config['training'].get('class_weight'):
                sample_weight = compute_sample_weight('balanced', self.y_train_val)
                grid_search.fit(self.X_train_val, self.y_train_val, sample_weight=sample_weight)
            else:
                grid_search.fit(self.X_train_val, self.y_train_val)
            
            best_models[name] = grid_search.best_estimator_
            print(f"Best parameters for {name}: {grid_search.best_params_}")
            print(f"Best cross-validation score for {name}: {grid_search.best_score_:.4f}")

            # Calculate and log average and std of test scores
            test_scores = grid_search.cv_results_['mean_test_score']
            avg_score = np.mean(test_scores)
            std_score = np.std(test_scores)
            print(f"Average CV score: {avg_score:.4f} ± {std_score:.4f}")

            summary.append(f"{name}:\n"
                           f"  Best parameters: {grid_search.best_params_}\n"
                           f"  Best CV score: {grid_search.best_score_:.4f}\n"
                           f"  Average CV score: {avg_score:.4f} ± {std_score:.4f}\n")

        # Save best models
        checkpoint_dir = os.path.join(self.config['logging']['log_dir'], 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        for name, model in best_models.items():
            joblib.dump(model, os.path.join(checkpoint_dir, f'{name}_best_model.joblib'))

        # Save summary
        with open(os.path.join(self.config['logging']['log_dir'], 'summary.txt'), 'w') as f:
            f.write("\n".join(summary))

        print_header('Training completed')

def run_experiment(config: Dict):
    data_seed = int(config['data']['preprocessed_dir'].split('_seed')[-1])
    if 'seed' not in config or config['seed'] != data_seed:
        print_header(f"Updating seed in config to match data seed: {data_seed}")
        config['seed'] = data_seed

    pl.seed_everything(config['seed'])

    experiment_dir = os.path.join(config['logging']['base_dir'], f"{config['experiment_name']}_{config['seed']}")
    os.makedirs(experiment_dir, exist_ok=False)
    config['logging']['log_dir'] = experiment_dir

    trainer = EFTrainer(config)
    trainer.run()

    # Save updated config after experiment
    with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as file:
        yaml.dump(config, file)

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
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, 'config.yaml')
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, 'r') as file:
        base_config = yaml.safe_load(file)

    import argparse
    parser = argparse.ArgumentParser(description="Run EF prediction experiment.")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning", default=True)
    args = parser.parse_args()

    if args.tune:
        param_grid = [
            {'experiment_name': 'ef_prediction_NIV_window_1', 'data.window_size': 1},
            {'experiment_name': 'ef_prediction_NIV_window_8', 'data.window_size': 8},
            {'experiment_name': 'ef_prediction_NIV_window_24', 'data.window_size': 24}
        ]
        
        for params in param_grid:
            print_header(f"Running experiment: {params['experiment_name']}")
            config = update_config(base_config, params)
            run_experiment(config)
    else:
        run_experiment(base_config)