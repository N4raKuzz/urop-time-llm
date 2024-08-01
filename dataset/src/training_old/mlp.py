from typing import List, Optional, Union, Callable
import torch
import torch.nn as nn
from .base_module import BaseEncoder, EncoderConfig

class MLPEncoder(BaseEncoder):
    def __init__(self, config: EncoderConfig):
        super().__init__(config)
        self.layers = self._build_layers()

    def _build_layers(self) -> nn.Sequential:
        layers = []
        in_features = self.config.input_dim
        
        for dim, dropout in zip(self.config.hidden_dims, self.config.dropout):
            layers.append(self._create_block(in_features, dim, dropout))
            in_features = dim
        
        layers.append(nn.Linear(in_features, self.config.output_dim))
        return nn.Sequential(*layers)

    def _create_block(self, in_features: int, out_features: int, dropout: float) -> nn.Sequential:
        block = [nn.Linear(in_features, out_features)]
        
        if self.config.batch_norm:
            block.append(nn.BatchNorm1d(out_features))
        elif self.config.layer_norm:
            block.append(nn.LayerNorm(out_features))
        
        block.append(self._get_activation())
        
        if dropout > 0:
            block.append(nn.Dropout(dropout))
        
        return nn.Sequential(*block)
        
    def _get_activation(self) -> nn.Module:
        if isinstance(self.config.activation, str):
            activation_name = self.config.activation.lower()
            if activation_name == 'relu':
                return nn.ReLU()
            elif activation_name == 'leakyrelu':
                return nn.LeakyReLU()
            elif activation_name == 'tanh':
                return nn.Tanh()
            elif activation_name == 'sigmoid':
                return nn.Sigmoid()
            else:
                raise ValueError(f"Unsupported activation function: {self.config.activation}")
        return self.config.activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.layers(x)
        return output.squeeze(-1) 
