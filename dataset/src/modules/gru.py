# src/modules/rnn.py
import torch
import torch.nn as nn
from .base_module import BaseModule

class GRU(BaseModule):
    def __init__(self, config):
        super().__init__(config)
        
        self.input_size = config['input_dim']
        self.hidden_size = config['hidden_dim']
        self.num_layers = config.get('num_layers', 1)
        self.bias = config.get('bias', True)
        self.batch_first = config.get('batch_first', False)
        self.dropout = config.get('dropout', 0)
        self.bidirectional = config.get('bidirectional', False)

        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=self.bias,
            batch_first=self.batch_first,
            dropout=self.dropout,
            bidirectional=self.bidirectional
        )

    def forward(self, x, h0=None):
        output, hn = self.gru(x, h0)
        return output, hn