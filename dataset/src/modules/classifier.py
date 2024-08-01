import torch
import torch.nn as nn
from .base_module import BaseModule

class Classifier(BaseModule):
    def __init__(self, config):
        super().__init__(config)
        self.input_dim = config['input_dim'] 
        self.output_dims = config['output_dims'] if isinstance(config['output_dims'], list) else [config['output_dims']]  # Changed from output_channels
        self.dropouts = config.get('dropouts', {})
        self.activation = config.get('activation')
        self.out_activation = config.get('out_activation')
        self.batch_norm = config.get('batch_norm', False)
        self.bias = config.get('bias', True)
        self.weight_normalization = config.get('weight_normalization', False)
        self.n_layers = len(self.output_dims)

        self.layers = self.build_classifier()

    def build_classifier(self):
        layers = nn.ModuleList()
        in_features = self.input_dim

        for layer_idx, out_features in enumerate(self.output_dims):
            layer = self.build_layer(in_features, out_features, layer_idx)
            layers.extend(layer)
            in_features = out_features

        return nn.Sequential(*layers)

    def build_layer(self, in_features, out_features, layer_idx):
        layer = []
        
        # Linear layer
        if self.weight_normalization:
            linear = nn.utils.weight_norm(nn.Linear(in_features, out_features, bias=self.bias), name='weight', dim=0)
        else:
            linear = nn.Linear(in_features, out_features, bias=self.bias)
        layer.append(linear)

        # Batch normalization
        if self.batch_norm:
            layer.append(nn.BatchNorm1d(out_features))

        # Activation
        if layer_idx < self.n_layers - 1:
            activation = self.get_activation(layer_idx)
            if activation:
                layer.append(activation)

        # Dropout
        dropout = self.dropouts.get(str(layer_idx))
        if dropout:
            layer.append(nn.Dropout(dropout))

        # Output activation
        if layer_idx == self.n_layers - 1 and self.out_activation:
            layer.append(self.get_activation(layer_idx, out=True))

        return layer

    def get_activation(self, idx, out=False):
        if out:
            act_config = self.out_activation
        else:
            act_config = self.activation
        
        if not act_config:
            return None

        activation_fns = {
            'relu': nn.ReLU(),
            'leakyrelu': nn.LeakyReLU(negative_slope=act_config.get('negative_slope', 0.01)),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'softmax': nn.Softmax(dim=act_config.get('dim', -1)),
            'logsoftmax': nn.LogSoftmax(dim=act_config.get('dim', -1)),
        }

        act_type = act_config['type'].lower() if isinstance(act_config, dict) else act_config.lower()
        return activation_fns.get(act_type)

    def forward(self, x):
        # x shape: (batch_size, seq_len, n_features) or (batch_size, n_features)
        if x.dim() == 3:
            # If input is (batch_size, seq_len, n_features), we'll use the last time step
            x = x[:, -1, :]
        return self.layers(x)