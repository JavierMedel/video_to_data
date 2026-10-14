"""Reuse Torch Hub's existing DINOv2 source without a GitHub branch lookup."""
from pathlib import Path


def load_dinov2(model_name, *, hub=None):
    if hub is None:
        import torch
        hub = torch.hub
    cached = Path(hub.get_dir()) / 'facebookresearch_dinov2_main'
    if (cached / 'hubconf.py').is_file():
        return hub.load(str(cached), model_name, source='local')
    # An explicit branch also avoids PyTorch's main/master discovery request.
    return hub.load('facebookresearch/dinov2:main', model_name)
