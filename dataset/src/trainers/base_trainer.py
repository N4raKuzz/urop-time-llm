# src/trainers/base_trainer.py
import os
import torch
import torch.nn as nn
from typing import Dict, Any
import importlib
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from src.utils.callbacks import CSVLogger, CustomPlotCallback
from src.utils.experiment_utils import print_header

class BaseTrainer(pl.LightningModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.model = self.setup_model()
        self.loss_fn = self.get_loss_function()
        self.metric_fns = nn.ModuleDict()
        self.callbacks = self.create_callbacks()
        self.custom_logger = self.create_logger()

    def setup(self, stage=None):
        self.setup_metrics()

    def setup_model(self) -> nn.Module:
        model_config = self.config['model']
        model_class = getattr(importlib.import_module(f"src.models.{model_config['name']}"), model_config['class'])
        return model_class(model_config['params'])

    def get_loss_function(self):
        loss_name = self.config['loss']['name']
        if hasattr(nn, loss_name):
            return getattr(nn, loss_name)()
        else:
            raise ValueError(f"Unsupported loss function: {loss_name}")

    def setup_metrics(self):
        for metric_name, metric_config in self.config['metrics'].items():
            module = importlib.import_module(metric_config['module'])
            metric_class = getattr(module, metric_config['class'])
            
            self.metric_fns[metric_name] = metric_class(**metric_config.get('params', {}))

    def create_callbacks(self):
        callbacks = []
        
        # ModelCheckpoint
        callbacks.append(ModelCheckpoint(
            dirpath=os.path.join(self.config['logging']['log_dir'], 'checkpoints'),
            filename='model-{epoch:02d}-{val_loss:.2f}',
            save_top_k=1,
            monitor='val_loss',
            every_n_epochs=self.config['logging'].get('save_every_n_epochs', 1)
        ))
        
        # CSVLogger
        callbacks.append(CSVLogger(
            os.path.join(self.config['logging']['log_dir'], 'metrics.csv'),
            self.config['logging'].get('log_every_n_epochs', 1)
        ))
        
        # CustomPlotCallback
        callbacks.append(CustomPlotCallback(
            self.config['logging'].get('plot_every_n_epochs', 1),
            os.path.join(self.config['logging']['log_dir'], 'plots')
        ))
        
        # EarlyStopping
        if self.config['training'].get('early_stopping_patience'):
            callbacks.append(EarlyStopping(
                monitor='val_loss',
                patience=self.config['training']['early_stopping_patience']
            ))
        
        return callbacks

    def create_logger(self):
        return TensorBoardLogger(
            save_dir=self.config['logging']['log_dir'],
            name='lightning_logs',
            log_graph=True
        )

    def prepare_data(self):
        print_header('Preparing data')
        dataset_config = self.config['dataset']
        dataset_class = getattr(importlib.import_module(dataset_config['module']), dataset_config['class'])
        self.train_dataset = dataset_class(self.config, data_fold='train')
        self.val_dataset = dataset_class(self.config, data_fold='val')

        self.example_input_array = torch.randn(
            1,  # batch size
            self.config['data']['window_size'],
            len(self.train_dataset.feature_columns)
        )  # as example array for tensorboard

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config['training']['train_batch_size'], shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config['training']['val_batch_size'])

    def configure_optimizers(self):
        optimizer_config = self.config['optimizer']
        optimizer_class = getattr(torch.optim, optimizer_config['class'])
        return optimizer_class(self.model.parameters(), **optimizer_config['params'])

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x).squeeze()
        if y_hat.shape != y.shape:
            y = y.view(y_hat.shape)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x).squeeze()
        if y_hat.shape != y.shape:
            y = y.view(y_hat.shape)
        loss = self.loss_fn(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # Calculate and log metrics
        for metric_name, metric_fn in self.metric_fns.items():
            metric_config = self.config['metrics'][metric_name]
            if metric_config.get('target_dtype') == 'int':
                y = y.long()
            metric_value = metric_fn(y_hat, y)
            self.log(f'val_{metric_name}', metric_value, on_step=False, on_epoch=True, sync_dist=True)
        
        return loss

    def forward(self, x):
        return self.model(x)
        
    def run(self):
        print_header('Starting training')

        trainer = pl.Trainer(
            max_epochs=self.config['training'].get('max_epochs', 1000),
            gradient_clip_val=self.config['training']['grad_clip_val'],
            callbacks=self.callbacks,
            logger=self.custom_logger,
            log_every_n_steps=self.config['logging'].get('log_every_n_steps', 50),
            accelerator='auto', 
            devices='auto',
            num_sanity_val_steps=0  # Skip sanity check
        )
        
        trainer.fit(self)
        print_header('Training completed')
        return trainer.callback_metrics