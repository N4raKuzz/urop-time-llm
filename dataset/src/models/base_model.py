# src/models/base_model.py
import torch.nn as nn
from typing import Dict, Any
import importlib

class BaseModel(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.encoder = self._build_encoder()
        self.classifier = self._build_classifier()

    def _build_encoder(self):
        encoder_config = self.config.get('encoder')
        if encoder_config is None:
            return None
        module = importlib.import_module(f"src.modules.{encoder_config['type'].lower()}")
        encoder_class = getattr(module, encoder_config['type'])
        return encoder_class(encoder_config)

    def _build_classifier(self):
        classifier_config = self.config['classifier']
        module = importlib.import_module("src.modules.classifier")
        return module.Classifier(classifier_config)

    def forward(self, x):
        # x shape: (batch_size, seq_len, n_features)
        if self.encoder:
            encoded = self.encoder(x)
            if isinstance(encoded, tuple):  # For RNN, which returns (output, hidden)
                encoded = encoded[0]
            if self.config['encoder']['type'] == 'RNN' and self.config['encoder'].get('batch_first', False):
                encoded = encoded[:, -1, :]  # Take the last output for each sequence
        else:
            encoded = x  # If no encoder, pass the input directly to the classifier
        return self.classifier(encoded)
