# src/modules/base.py
from abc import ABC, abstractmethod
import torch.nn as nn

class BaseModule(nn.Module, ABC):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.name = config.get('name', self.__class__.__name__)
        self.freeze = config.get('freeze', False)

    @abstractmethod
    def forward(self, x):
        pass

    def train(self, mode=True):
        if self.freeze:
            return super().train(False)
        return super().train(mode)
