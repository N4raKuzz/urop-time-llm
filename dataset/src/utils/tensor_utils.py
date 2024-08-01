import torch
import numpy as np

def convert_tensors_to_numpy(*tensors):
    return tuple(t.cpu().numpy() if isinstance(t, torch.Tensor) else t for t in tensors)