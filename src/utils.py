import glob

import torch
from PIL import Image
from torch import nn


def swish(x: torch.Tensor):
    """
    ### Swish activation

    $$x \cdot \sigma(x)$$
    """
    return x * torch.sigmoid(x)


def normalization(channels: int):
    """
    ### Group normalization

    This is a helper function, with fixed number of groups and `eps`.
    """
    return nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6)


def zero_module(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        p.detach().zero_()
    return module


def get_entries(path):
    return sorted(glob.glob(path))


def resize_max(img, max_size):
    w, h = img.size
    if w > max_size or h > max_size:
        if w <= h:
            h = int(float(h) * float(max_size) / float(w))
            w = max_size
        else:
            w = int(float(w) * float(max_size) / float(h))
            h = max_size
        img = img.resize((w, h), Image.Resampling.LANCZOS)

    return img


def resize(img, size):
    w, h = img.size
    if w != size or h != size:
        if w <= h:
            h = int(float(h) * float(size) / float(w))
            w = size
        else:
            w = int(float(w) * float(size) / float(h))
            h = size
        img = img.resize((w, h), Image.Resampling.LANCZOS)

    return img
