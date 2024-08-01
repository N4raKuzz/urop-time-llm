import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.utils.data import DataLoader
from src.utils.callbacks import MetricPrinterCallback
from typing import Dict, Any, Callable
import torch
import torch.nn as nn

class EncoderModule(pl.LightningModule):
    def __init__(self, encoder: nn.Module, train_config: Dict[str, Any],
                 metric_fn: Callable[[torch.Tensor, torch.Tensor], Dict[str, float]]):
        super().__init__()
        self.encoder = encoder
        self.train_config = train_config
        self.metric_fn = metric_fn
        self.loss_fn = self.train_config['loss_fn']
        self.grad_clip_val = self.train_config.get('grad_clip_val', 0.0)

    def forward(self, x):
        return self.encoder(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        metrics = self.metric_fn(y, y_hat)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                self.log(f'val_{name}', value, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_after_backward(self):
        # Gradient clipping
        if self.grad_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.grad_clip_val)

    def on_train_epoch_end(self):
        # Log histograms of model parameters
        for name, param in self.named_parameters():
            self.logger.experiment.add_histogram(name, param, self.current_epoch)

    def configure_optimizers(self):
        optimizer_class = self.train_config['optimizer']['class']
        optimizer = optimizer_class(self.parameters(), **self.train_config['optimizer']['params'])
        return optimizer
    
class Trainer:
    def __init__(self, encoder: nn.Module, train_config: Dict[str, Any], 
                 metric_fn: Callable[[torch.Tensor, torch.Tensor], Dict[str, float]],
                 logger: TensorBoardLogger = None):
        self.encoder = encoder
        self.train_config = train_config
        self.metric_fn = metric_fn
        self.pl_module = None
        self.logger = logger

    def setup_module(self):
        self.pl_module = EncoderModule(self.encoder, self.train_config, self.metric_fn)

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        self.setup_module()
        if self.logger is None:
            self.logger = TensorBoardLogger(save_dir=self.train_config['log_dir'], name="lightning_logs")

        trainer = pl.Trainer(
            max_epochs=self.train_config['max_epochs'],
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=self.train_config['early_stopping_patience'], mode='min'),
                ModelCheckpoint(monitor='val_loss', dirpath=self.train_config['checkpoint_dir'], 
                                filename='model-{epoch:02d}-{val_loss:.2f}', save_top_k=1, mode='min'),
                LearningRateMonitor(logging_interval='step'),
                MetricPrinterCallback(print_every_n_epochs=1)
            ],
            logger=self.logger,
            log_every_n_steps=self.train_config['log_every_n_steps'],
            accelerator='auto', 
            devices='auto'
        )
        
        trainer.fit(self.pl_module, train_loader, val_loader)

    def get_best_model(self):
        return self.pl_module.encoder

    def get_logger(self):
        return self.logger