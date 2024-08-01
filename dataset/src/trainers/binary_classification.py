from typing import Any, Dict, Tuple
from .base_trainer import BaseExperiment
from .experiment_config import ExperimentConfig
from ..data.data_handler import DataHandler
from ..data.base import CustomTensorDataset
from ..models import get_encoder
from ..training_old.trainer import Trainer
from ..evaluation.metrics import ClassificationMetrics
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger


class BinaryClassifier(BaseExperiment):
    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.data_handler = DataHandler(config.data_config)
        self.metric_fn = self._get_metric_fn()
        self.logger = TensorBoardLogger(save_dir=config.log_dir, name='lightning_logs')

    def _get_metric_fn(self):
        def metric_fn(y_true: torch.Tensor, y_logit: torch.Tensor) -> Dict[str, float]:
            y_prob = torch.sigmoid(y_logit)
            return ClassificationMetrics.calculate_metrics(y_true.cpu().numpy(), y_prob.cpu().numpy())
        return metric_fn

    def prepare_data(self):
        return self.data_handler.prepare_data()    

    def prepare_data_loader(self, data: Tuple[Dict, Dict, Dict]) -> Tuple[DataLoader, DataLoader, DataLoader]:
        train_data, val_data, test_data = data
        
        train_loader = DataLoader(
            CustomTensorDataset(train_data['X'], train_data['y']),
            batch_size=self.config.train_config['batch_size'],
            shuffle=True,
            num_workers=8
        )
        val_loader = DataLoader(
            CustomTensorDataset(val_data['X'], val_data['y']),
            batch_size=self.config.train_config['batch_size'],
            num_workers=8
        )
        test_loader = CustomTensorDataset(test_data['X'], test_data['y'])
        
        return train_loader, val_loader, test_loader

    def setup_model(self) -> torch.nn.Module:
        return get_encoder(self.config.model_config)

    def train(self, model: torch.nn.Module, data: Tuple[DataLoader, DataLoader, Any]) -> Tuple[torch.nn.Module, TensorBoardLogger]:
        train_loader, val_loader, _ = data
        trainer = Trainer(model, self.config.train_config, self.metric_fn, logger=self.logger)
        trainer.train(train_loader, val_loader)
        return trainer.get_best_model(), self.logger

    def evaluate(self, model: torch.nn.Module, data: Tuple[DataLoader, DataLoader, DataLoader]) -> Dict[str, Any]:
        _, _, test_loader = data
        model.eval()    
        with torch.no_grad():
            test_pred = model(test_loader.X)
        test_metrics = self.metric_fn(test_loader.y, test_pred)
        
        # Log test metrics
        for metric_name, metric_value in test_metrics.items():
            if isinstance(metric_value, (int, float)):
                self.logger.experiment.add_scalar(f'test_{metric_name}', metric_value, self.config.seed)

        return {
            'model': model,
            'metrics': {
                self.config.model_config['type']: test_metrics  
            },
            'preprocessor': self.data_handler.get_preprocessor()
        }
 
