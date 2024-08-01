from typing import List, Dict, Any, Tuple
import torch
from src.trainers.binary_classification import BinaryClassifier
from src.trainers.experiment_config import ExperimentConfig
from data.base import CustomTensorDataset
from src.evaluation.metrics import ClassificationMetrics

class DynamicPrediction(BinaryClassifier):
    def __init__(self, config: ExperimentConfig):
        super().__init__(config)

    def _get_last_observations(self, test_data):
        indices = test_data['indices']
        assert len(indices) == len(test_data['X']) == len(test_data['y']), "Indices and data lengths do not match"

        # Get column names dynamically
        id_col, index_col = indices.columns

        # Create a boolean mask for the last observations
        mask = indices.groupby(id_col)[index_col].transform(max) == indices[index_col]

        return test_data['X'][mask], test_data['y'][mask]

    def _negate_metric_fn(self, y_true: torch.Tensor, y_logit: torch.Tensor) -> Dict[str, float]:
        y_prob = torch.sigmoid(y_logit)
        return ClassificationMetrics.calculate_metrics(1-y_true.cpu().numpy(), 1-y_prob.cpu().numpy())

    def _calculate_class_weights(self, y):
        class_counts = torch.bincount(torch.tensor(y, dtype=torch.int64))
        total_samples = len(y)
        class_weights = total_samples / (2 * class_counts)
        return class_weights

    def setup_model(self) -> torch.nn.Module:
        model = super().setup_model()

        if self.config.train_config['weighted_loss']:
            # Calculate class weights
            class_weights = self._calculate_class_weights(self.data_dict[0]['y'])
            
            # Set up loss function with class weights
            self.config.train_config['loss_fn'] = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights[1])
        return model

    def run_sanity_check(self, model: torch.nn.Module, test_data: Dict[str, Any]) -> Dict[str, Any]:
        last_obs_X, last_obs_y = self._get_last_observations(test_data)
        test_loader = CustomTensorDataset(last_obs_X, last_obs_y)
        model.eval()
        with torch.no_grad():
            test_logit = model(test_loader.X)
        sanity_check_metrics = self._negate_metric_fn(test_loader.y, test_logit)
        
        # Log sanity check metrics
        for metric_name, metric_value in sanity_check_metrics.items():
            if isinstance(metric_value, (int, float)):
                self.logger.experiment.add_scalar(f'sanity_check_{metric_name}', metric_value, self.config.seed)

        return {
            'metrics': {
                f"{self.config.model_config['type']}_sanity_check": sanity_check_metrics
            }
        }

    def run(self, seed: int) -> Tuple[Dict[str, Any], Any]:
        results, logger = super().run(seed)
        sanity_check_results = self.run_sanity_check(results['model'], self.data_dict[-1])  # data[2] is test_data
        results['metrics'].update(sanity_check_results['metrics'])
        return results, logger
  
    
