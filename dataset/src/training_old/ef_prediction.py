from typing import Any, Dict, Tuple
from src.trainers.binary_classification import BinaryClassifier
from src.evaluation.metrics import ClassificationMetrics
from data.base import CustomTensorDataset
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, average_precision_score
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

class EFPrediction(BinaryClassifier):
    def _get_metric_fn(self):
        def metric_fn(y_true, y_proba) -> Dict[str, float]:
            return ClassificationMetrics.calculate_metrics(y_true, y_proba)
        return metric_fn

    def setup_model(self) -> Dict[str, Any]:
        from src.models.classic_ml import get_classic_ml_models
        return get_classic_ml_models()

    def prepare_data_loader(self, data: Tuple[Dict, Dict, Dict]) -> Tuple[DataLoader, DataLoader, DataLoader]:
        train_data, val_data, test_data = data

        X_train, y_train = train_data['X'], train_data['y']
        X_val, y_val = val_data['X'], val_data['y']
        X_train_val = np.concatenate([X_train, X_val], axis=0)
        y_train_val = np.concatenate([y_train, y_val], axis=0)    

        # balancing positive/negative samples
        rus = RandomUnderSampler(sampling_strategy=.5)
        X_train_val_resampled, y_train_val_resampled = rus.fit_resample(X_train_val, y_train_val.ravel())

        ros = RandomOverSampler()
        X_train_val_resampled, y_train_val_resampled = ros.fit_resample(X_train_val_resampled, y_train_val_resampled)

        # using full list of variables
        X_train_val_selected = X_train_val_resampled
        X_test_selected = test_data['X']

        # # Feature Selection using RFE with cross-validation:
        # print('='*100)
        # selector = RFECV(estimator=LogisticRegression(max_iter=5000, random_state=seed_value), 
        #                  min_features_to_select=30, scoring=scorer, step=1, cv=5)
        # selector = RFECV(estimator=RandomForestClassifier(random_state=seed_value), step=1, cv=5)
        # selector = selector.fit(X_train_resampled, y_train_resampled)
        # selected_features = X_train_resampled.columns[selector.support_]
        # print(f'Total of {len(selected_features)} features has been selected:\n', selected_features)

        # # Use only selected features for training and testing
        # X_train_selected = X_train_resampled[selected_features]
        # X_test_selected = X_test[selected_features]

        train_val_loader = CustomTensorDataset(X_train_val_selected, y_train_val_resampled)
        test_loader = CustomTensorDataset(X_test_selected, test_data['y'])

        return train_val_loader, test_loader

    def train(self, models, data, reweight_class=False):
        X_train_val, y_train_val = data[0].X.numpy(), data[0].y.numpy()

        scorer = make_scorer(average_precision_score, needs_proba=True) 
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train_val), y=y_train_val)
        class_weight_dict = dict(enumerate(class_weights))
        if not reweight_class:
            class_weight_dict = {key: class_weight_dict[key] * 0.5 for key in class_weight_dict}

        trained_models = {}

        for name, (model, params) in models.items():
            print(f"Tuning hyperparameters for {name}...")
            model_to_tune = model(class_weight=class_weight_dict) if 'class_weight' in model().get_params() else model()
            grid_search = GridSearchCV(model_to_tune, params, cv=5, scoring=scorer, refit=True)
            grid_search.fit(X_train_val, y_train_val)
            best_model = grid_search.best_estimator_
            trained_models[name] = best_model
            print(f"Best params for {name}: {grid_search.best_params_}")
            print(f"Best score for {name}: {grid_search.best_score_}")
        
        return trained_models, None

    def evaluate(self, trained_models, data):
        X_test, y_test = data[-1].X.numpy(), data[-1].y.numpy()

        evaluation_results = {}
        for name, model in trained_models.items():
            y_proba = model.predict_proba(X_test)[:, 1]
            test_metrics = self.metric_fn(y_test, y_proba)   
            evaluation_results[name] = test_metrics

        return {
            'metrics': evaluation_results,
            'model': trained_models,
            'preprocessor': self.data_handler.get_preprocessor()
        }
