from mindspore import nn

from models.embeddings.sinusoidal import build_sinusoidal_embedding

def build_embedding(name: str, *args, **kwargs) -> nn.cell:
    if 'sinusoidal' in name:
        return build_sinusoidal_embedding(*args, **kwargs)