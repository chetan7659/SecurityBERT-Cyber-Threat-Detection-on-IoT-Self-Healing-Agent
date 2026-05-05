"""
Shared utility functions for SecurityBERT pipeline.
Covers: config loading, seeding, device, class weights,
        checkpointing, logging, and metrics helpers.
"""

import os
import json
import random
import logging
import time
from pathlib import Path
from typing  import Dict, Optional, Union

import numpy  as np
import torch
import yaml


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Create a formatted logger.

    Parameters
    ----------
    name  : str — logger name (usually __name__)
    level : int — logging level

    Returns
    -------
    logging.Logger
    """
    logger    = logging.getLogger(name)
    if not logger.handlers:
        handler   = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt  = '[%(asctime)s] %(levelname)s — %(name)s — %(message)s',
            datefmt= '%Y-%m-%d %H:%M:%S',
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: Union[str, Path] = 'configs/config.yaml') -> dict:
    """
    Load YAML configuration file.

    Parameters
    ----------
    config_path : path to config.yaml

    Returns
    -------
    dict — full configuration dictionary
    """
    config_path = Path(config_path)
    assert config_path.exists(), (
        f'❌ Config not found: {config_path.resolve()}'
    )
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f'Config loaded from {config_path}')
    return config


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """
    Set all random seeds for full reproducibility.

    Parameters
    ----------
    seed : int — random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
    logger.info(f'Random seed set to {seed}')


# ─────────────────────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """
    Return best available device.

    Returns
    -------
    torch.device — cuda if available, else cpu
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        logger.info(
            f'Device: {device}  '
            f'GPU: {torch.cuda.get_device_name(0)}  '
            f'VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB'
        )
    else:
        logger.info(f'Device: {device}  (CPU mode)')
    return device


# ─────────────────────────────────────────────────────────────────────────────
# Class Imbalance
# ─────────────────────────────────────────────────────────────────────────────

def compute_class_weights(
    labels     : torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """
    Compute per-class weights for imbalanced datasets.

    Formula (sklearn 'balanced'):
        weight_i = total_samples / (num_classes × count_i)

    High weight → rare class → model penalized heavily for missing it.
    Directly addresses the 1,615× imbalance in Edge-IIoTset.

    Parameters
    ----------
    labels      : 1D tensor of integer class labels
    num_classes : total number of classes

    Returns
    -------
    torch.Tensor of shape (num_classes,) — per-class weights
    """
    counts       = torch.bincount(labels, minlength=num_classes).float()
    total        = len(labels)
    weights      = total / (num_classes * counts)

    # Handle classes with zero samples
    weights      = torch.where(
        counts > 0,
        weights,
        torch.zeros_like(weights)
    )
    logger.info(
        f'Class weights computed: '
        f'min={weights[counts>0].min():.3f}  '
        f'max={weights.max():.3f}'
    )
    return weights


# ─────────────────────────────────────────────────────────────────────────────
# Checkpointing
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    state    : dict,
    path     : Union[str, Path],
    is_best  : bool = False,
    best_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Save model checkpoint.

    Parameters
    ----------
    state     : dict — checkpoint contents
    path      : save path for this checkpoint
    is_best   : if True, also copy to best_path
    best_path : path for best model (only used if is_best=True)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    logger.info(f'Checkpoint saved → {path}')

    if is_best and best_path is not None:
        import shutil
        shutil.copy(str(path), str(best_path))
        logger.info(f'Best model updated → {best_path}')


def load_checkpoint(
    path  : Union[str, Path],
    device: torch.device,
) -> dict:
    """
    Load model checkpoint.

    Parameters
    ----------
    path   : checkpoint file path
    device : target device

    Returns
    -------
    dict — checkpoint contents
    """
    path = Path(path)
    assert path.exists(), f'❌ Checkpoint not found: {path}'
    ckpt = torch.load(path, map_location=device, weights_only=False)
    logger.info(f'Checkpoint loaded from {path}')
    return ckpt


# ─────────────────────────────────────────────────────────────────────────────
# Metrics helpers
# ─────────────────────────────────────────────────────────────────────────────

def format_metrics(metrics: dict, prefix: str = '') -> str:
    """
    Format metrics dictionary into a readable string.

    Parameters
    ----------
    metrics : dict of metric_name → float
    prefix  : optional prefix string

    Returns
    -------
    str — formatted metrics line
    """
    parts = [f'{prefix}' if prefix else '']
    for k, v in metrics.items():
        if isinstance(v, float):
            parts.append(f'{k}: {v:.4f}')
        elif isinstance(v, int):
            parts.append(f'{k}: {v:,}')
    return '  |  '.join(p for p in parts if p)


def save_json(data: dict, path: Union[str, Path]) -> None:
    """Save dictionary to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    logger.info(f'JSON saved → {path}')


def load_json(path: Union[str, Path]) -> dict:
    """Load JSON file to dictionary."""
    with open(Path(path), 'r') as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Timer
# ─────────────────────────────────────────────────────────────────────────────

class Timer:
    """Simple context manager timer."""
    def __enter__(self):
        self.start = time.time()
        return self
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
    def __str__(self):
        return f'{self.elapsed:.2f}s'
