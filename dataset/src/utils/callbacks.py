import pytorch_lightning as pl
import pandas as pd
import os
import matplotlib.pyplot as plt
from src.utils.data_utils import to_numpy
import numpy as np

class MetricPrinterCallback(pl.Callback):
    def __init__(self, print_every_n_epochs=1):
        self.print_every_n_epochs = print_every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.print_every_n_epochs == 0:
            metrics = trainer.callback_metrics
            print(f"\n*****Epoch {trainer.current_epoch}*****")
            
            for name, value in metrics.items():
                value = to_numpy(value)
                if isinstance(value, (float, int, np.number)):
                    print(f"  {name}: {value:.4f}")
                else:
                    print(f"  {name}: {value}")
            print("\n")

class CSVLogger(pl.Callback):
    def __init__(self, filepath, log_every_n_epochs=1):
        self.filepath = filepath
        self.log_every_n_epochs = log_every_n_epochs
        self.metrics = []

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.log_every_n_epochs == 0:
            metrics = {k: to_numpy(v) for k, v in trainer.callback_metrics.items()}
            self.metrics.append({**metrics, 'epoch': trainer.current_epoch})
            df = pd.DataFrame(self.metrics)
            df.to_csv(self.filepath, index=False)

class CustomPlotCallback(pl.Callback):
    def __init__(self, plot_every_n_epochs=1, plot_dir='plots'):
        self.plot_every_n_epochs = plot_every_n_epochs
        self.plot_dir = plot_dir
        os.makedirs(self.plot_dir, exist_ok=True)
        self.metrics = {}
        self.figures = {}

    def on_validation_epoch_end(self, trainer, pl_module):
        self._update_metrics(trainer.callback_metrics)
        if trainer.current_epoch % self.plot_every_n_epochs == 0:
            self.plot_metrics(trainer)

    def _update_metrics(self, callback_metrics):
        for name, value in callback_metrics.items():
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append(to_numpy(value))

    def _group_metrics(self):
        metric_groups = {}
        for metric_name, values in self.metrics.items():
            group_name = metric_name.split('_')[-1] if '_' in metric_name else metric_name
            if group_name not in metric_groups:
                metric_groups[group_name] = {}
            metric_groups[group_name][metric_name] = values
        return metric_groups

    def plot_metrics(self, trainer):
        metric_groups = self._group_metrics()
        
        for group_name, metrics in metric_groups.items():
            fig = plt.figure(figsize=(10, 5))
            
            # Find the maximum length of metric values
            max_len = max(len(values) for values in metrics.values())
            
            for metric_name, values in metrics.items():
                label = metric_name.split('_')[0] if '_' in metric_name else metric_name
                color = 'b' if 'train' in metric_name else 'r'
                
                # Align metrics to the right
                aligned_values = [None] * (max_len - len(values)) + values
            
                plt.plot(range(len(aligned_values)), aligned_values, f'{color}-o', label=label)
            
            plt.xlabel('Epoch')
            plt.ylabel(group_name)
            plt.legend()
            plt.title(f'{group_name} vs Epoch')
            plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            plt.tight_layout()
            plt.savefig(os.path.join(self.plot_dir, f'{group_name}_plot.png'))
            plt.close(fig)