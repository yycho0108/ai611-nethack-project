#!/usr/bin/env python3

from pathlib import Path
from os import PathLike
from typing import Union
import torch as th


def get_device(device: str = 'auto') -> th.device:
    if device == 'auto':
        if th.cuda.is_available():
            device = th.device('cuda')
        else:
            device = th.device('cpu')
    return th.device(device)


def ensure_dir(path: Union[str, PathLike]) -> Path:
    """ensure that directory exists."""
    path = Path(path).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_new_dir(root: Union[str, PathLike]) -> Path:
    """Get new runtime directory."""
    root = Path(root).expanduser()
    index = len([d for d in root.glob('run-*') if d.is_dir()])
    path = ensure_dir(root / F'run-{index:03d}')
    return path
