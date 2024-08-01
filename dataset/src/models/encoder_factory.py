from typing import Dict, Any
from ..encoders import BaseEncoder, EncoderConfig, MLPEncoder, LSTMEncoder

def get_encoder(config: Dict[str, Any]) -> BaseEncoder:
    encoder_type = config['type'] 
    encoder_config = EncoderConfig(**{k: v for k, v in config.items() if k != 'type'})

    if encoder_type == 'mlp':
        return MLPEncoder(encoder_config)
    elif encoder_type == 'lstm':
        return LSTMEncoder(encoder_config)
    else:
        raise ValueError(f"Unsupported encoder type: {encoder_type}")