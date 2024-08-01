from typing import List, Optional, Union, Callable
import torch
import torch.nn as nn
from .base_module import BaseEncoder, EncoderConfig

class LSTMEncoder(BaseEncoder):
    def __init__(self, config: EncoderConfig):
        super().__init__(config)
        
        if isinstance(config.dropout, list):
            dropout = config.dropout[0] if config.num_layers > 1 else 0
        else:
            dropout = config.dropout if config.num_layers > 1 else 0
        
        self.lstm = nn.LSTM(
            input_size=config.input_dim,
            hidden_size=config.hidden_dims[0],
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=config.bidirectional,
            dropout=dropout
        )
        lstm_output_dim = config.hidden_dims[0] * (2 if config.bidirectional else 1)
        self.fc = nn.Linear(lstm_output_dim, config.output_dim)
        self.last_dropout = nn.Dropout(config.dropout if isinstance(config.dropout, float) else config.dropout[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        if self.config.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1]
            
        hidden = self.last_dropout(hidden)
        output = self.fc(hidden)
        return output.squeeze(-1)
