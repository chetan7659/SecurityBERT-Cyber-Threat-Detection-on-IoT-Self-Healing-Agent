"""
PyTorch Dataset and DataLoader builders for SecurityBERT.
Implements class-imbalance strategy (Option 3):
    - WeightedRandomSampler (stratified sampling)
    - Compatible with FocalLoss + class weights
"""

from typing  import Optional, Tuple
from pathlib import Path

import numpy  as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class SecurityBERTDataset(Dataset):
    """
    PyTorch Dataset wrapping tokenized SecurityBERT inputs.

    Parameters
    ----------
    input_ids       : (N, seq_len) — BBPE token IDs
    attention_masks : (N, seq_len) — attention masks
    labels          : (N,)         — integer class labels
    """

    def __init__(
        self,
        input_ids      : torch.Tensor,
        attention_masks: torch.Tensor,
        labels         : torch.Tensor,
    ):
        assert len(input_ids) == len(attention_masks) == len(labels), \
            'All tensors must have the same length'
        self.input_ids       = input_ids
        self.attention_masks = attention_masks
        self.labels          = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            'input_ids'      : self.input_ids[idx],
            'attention_mask' : self.attention_masks[idx],
            'labels'         : self.labels[idx],
        }


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader builder
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(
    input_ids       : torch.Tensor,
    attention_masks : torch.Tensor,
    labels          : torch.Tensor,
    class_weights   : torch.Tensor,
    batch_size      : int   = 128,
    train_ratio     : float = 0.80,
    random_state    : int   = 42,
    num_workers     : int   = 0,
) -> Tuple[DataLoader, DataLoader, dict]:
    """
    Build train and val DataLoaders with stratified split
    and WeightedRandomSampler for class imbalance (Option 3).

    Parameters
    ----------
    input_ids       : full tokenized input IDs
    attention_masks : full attention masks
    labels          : full label tensor
    class_weights   : per-class weights (from compute_class_weights)
    batch_size      : training batch size (paper: 128)
    train_ratio     : train/val split ratio (paper: 0.80)
    random_state    : random seed
    num_workers     : DataLoader workers

    Returns
    -------
    train_loader : DataLoader with WeightedRandomSampler
    val_loader   : DataLoader sequential
    split_info   : dict with train/val indices and sizes
    """
    N       = len(labels)
    indices = np.arange(N)

    # ── Stratified 80/20 split ────────────────────────────────────────────────
    train_idx, val_idx = train_test_split(
        indices,
        test_size    = 1 - train_ratio,
        stratify     = labels.numpy(),
        random_state = random_state,
    )

    train_ids    = input_ids [train_idx]
    train_masks  = attention_masks[train_idx]
    train_labels = labels    [train_idx]

    val_ids      = input_ids [val_idx]
    val_masks    = attention_masks[val_idx]
    val_labels   = labels    [val_idx]

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_dataset = SecurityBERTDataset(train_ids, train_masks, train_labels)
    val_dataset   = SecurityBERTDataset(val_ids,   val_masks,   val_labels)

    # ── WeightedRandomSampler — Component 2 of Option 3 ──────────────────────
    # Every batch guaranteed to contain all 15 classes
    sample_weights = class_weights[train_labels]
    sampler        = WeightedRandomSampler(
        weights     = sample_weights,
        num_samples = len(train_labels),
        replacement = True,
    )

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_dataset,
        batch_size  = batch_size,
        sampler     = sampler,
        num_workers = num_workers,
        pin_memory  = torch.cuda.is_available(),
        drop_last   = True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size  = batch_size * 2,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = torch.cuda.is_available(),
    )

    split_info = {
        'n_train'    : len(train_idx),
        'n_val'      : len(val_idx),
        'train_idx'  : train_idx,
        'val_idx'    : val_idx,
        'train_labels': train_labels,
        'val_labels'  : val_labels,
    }

    return train_loader, val_loader, split_info
