"""
SecurityBERT Training & Evaluation.
Implements Option 3 class imbalance strategy:
    Component 1 — Class Weights
    Component 2 — WeightedRandomSampler (in dataset.py)
    Component 3 — Focal Loss

Evaluation metrics (matching paper Section IV):
    Primary   : Weighted F1
    Secondary : Macro F1
    Critical  : Per-class Recall (cybersecurity priority)
    Visual    : Confusion Matrix, ROC-AUC
"""

import time
import json
from pathlib import Path
from typing  import Dict, List, Optional, Tuple, Union

import numpy  as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim      import AdamW
from transformers     import get_linear_schedule_with_warmup

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


# ─────────────────────────────────────────────────────────────────────────────
# Focal Loss — Component 3 of Option 3
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss for extreme class imbalance.
    Lin et al. (2017).

    FL(pt) = -α · (1 - pt)^γ · log(pt)

    Where:
        pt  = model probability for the correct class
        α   = class weight (Component 1 from compute_class_weights)
        γ   = focusing parameter (paper default = 2.0)

    Effect with γ=2:
        Well-classified (pt=0.99)  → (1-0.99)² = 0.0001 → tiny contribution
        Misclassified   (pt=0.10)  → (1-0.10)² = 0.81   → large contribution
        → Rare attack classes DOMINATE the gradient
        → Easy Normal class contributes almost NOTHING

    Parameters
    ----------
    alpha    : Tensor (num_classes,) — per-class weights
    gamma    : float — focusing parameter (default 2.0)
    reduction: str   — 'mean' | 'sum' | 'none'
    """

    def __init__(
        self,
        alpha    : torch.Tensor,
        gamma    : float = 2.0,
        reduction: str   = 'mean',
    ):
        super().__init__()
        self.register_buffer('alpha', alpha.float())
        self.gamma     = gamma
        self.reduction = reduction

    def forward(
        self,
        logits : torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Focal Loss.

        Parameters
        ----------
        logits  : (batch, num_classes) — raw model output
        targets : (batch,)             — ground truth class indices

        Returns
        -------
        torch.Tensor — scalar loss
        """
        # Per-sample cross-entropy
        ce_loss      = F.cross_entropy(logits, targets, reduction='none')

        # pt = probability of the correct class
        pt           = torch.exp(-ce_loss)

        # Focal weight
        focal_weight = (1.0 - pt) ** self.gamma

        # Per-sample class weight
        alpha_t      = self.alpha[targets]

        # Focal Loss
        focal_loss   = alpha_t * focal_weight * ce_loss

        if self.reduction == 'mean': return focal_loss.mean()
        if self.reduction == 'sum' : return focal_loss.sum()
        return focal_loss


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    model      : nn.Module,
    loader     : DataLoader,
    criterion  : nn.Module,
    device     : torch.device,
    label_map  : dict,
    num_classes: int,
) -> dict:
    """
    Full model evaluation.

    Computes all metrics matching paper Section IV:
    - Overall accuracy
    - Weighted F1 (primary metric)
    - Macro F1 (secondary metric)
    - Per-class Precision, Recall, F1 (Table VI)
    - Per-class ROC-AUC (Figure 7)
    - Raw predictions + probabilities

    Parameters
    ----------
    model       : SecurityBERT or SecurityBERTWithSoftmax
    loader      : validation DataLoader
    criterion   : FocalLoss instance
    device      : torch.device
    label_map   : {str(idx): class_name}
    num_classes : total number of classes

    Returns
    -------
    dict with all evaluation metrics
    """
    model.eval()
    all_preds  = []
    all_labels = []
    all_probs  = []
    total_loss = 0.0
    n_batches  = 0

    with torch.no_grad():
        for batch in loader:
            ids   = batch['input_ids'].to(device)
            masks = batch['attention_mask'].to(device)
            lbls  = batch['labels'].to(device)

            out    = model(ids, masks)
            logits = out['logits']
            loss   = criterion(logits, lbls)

            # Softmax probabilities for ROC-AUC
            probs  = F.softmax(logits, dim=-1)

            total_loss += loss.item()
            n_batches  += 1

            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(lbls.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.vstack(all_probs)

    class_names = [label_map[str(i)] for i in range(num_classes)]

    # ── Core metrics ──────────────────────────────────────────────────────────
    accuracy    = accuracy_score(all_labels, all_preds)
    weighted_f1 = f1_score(all_labels, all_preds,
                           average='weighted', zero_division=0)
    macro_f1    = f1_score(all_labels, all_preds,
                           average='macro', zero_division=0)

    # ── Per-class report ──────────────────────────────────────────────────────
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    # ── Per-class recall (most critical for cybersecurity) ────────────────────
    per_class_recall = {
        cls: report[cls]['recall']
        for cls in class_names
        if cls in report
    }

    # ── ROC-AUC per class (Figure 7) ─────────────────────────────────────────
    y_bin     = label_binarize(all_labels, classes=list(range(num_classes)))
    auc_scores= {}
    for i, cls in enumerate(class_names):
        if y_bin[:, i].sum() > 0:
            try:
                auc_scores[cls] = roc_auc_score(y_bin[:, i], all_probs[:, i])
            except Exception:
                auc_scores[cls] = 0.0
        else:
            auc_scores[cls] = 0.0

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds)

    return {
        'loss'             : total_loss / n_batches,
        'accuracy'         : accuracy,
        'weighted_f1'      : weighted_f1,
        'macro_f1'         : macro_f1,
        'report'           : report,
        'per_class_recall' : per_class_recall,
        'auc_scores'       : auc_scores,
        'mean_auc'         : np.mean(list(auc_scores.values())),
        'confusion_matrix' : cm,
        'all_preds'        : all_preds,
        'all_labels'       : all_labels,
        'all_probs'        : all_probs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    """
    SecurityBERT training pipeline.

    Implements paper training setup (Section III-G, III-H):
        - AdamW optimizer with weight decay
        - Linear warmup + linear decay scheduler
        - Focal Loss (Component 3)
        - Gradient clipping (max_norm=1.0)
        - Best model tracking by Weighted F1

    Parameters
    ----------
    model         : SecurityBERT or SecurityBERTWithSoftmax
    train_loader  : training DataLoader (with WeightedRandomSampler)
    val_loader    : validation DataLoader
    criterion     : FocalLoss instance
    device        : torch.device
    label_map     : {str(idx): class_name}
    num_classes   : int
    learning_rate : float (pre-training: 2e-5, fine-tuning: 1e-5)
    num_epochs    : int (paper: 4)
    warmup_ratio  : float (default: 0.10)
    max_grad_norm : float (default: 1.0)
    checkpoint_dir: path to save checkpoints
    """

    def __init__(
        self,
        model         : nn.Module,
        train_loader  : DataLoader,
        val_loader    : DataLoader,
        criterion     : FocalLoss,
        device        : torch.device,
        label_map     : dict,
        num_classes   : int,
        learning_rate : float = 2e-5,
        num_epochs    : int   = 4,
        warmup_ratio  : float = 0.10,
        max_grad_norm : float = 1.0,
        checkpoint_dir: Union[str, Path] = 'checkpoints',
    ):
        self.model          = model
        self.train_loader   = train_loader
        self.val_loader     = val_loader
        self.criterion      = criterion
        self.device         = device
        self.label_map      = label_map
        self.num_classes    = num_classes
        self.num_epochs     = num_epochs
        self.max_grad_norm  = max_grad_norm
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # ── Optimizer (separate weight decay from bias/LN) ────────────────────
        no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
        grouped  = [
            {'params': [p for n, p in model.named_parameters()
                        if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters()
                        if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
        ]
        self.optimizer = AdamW(grouped, lr=learning_rate, eps=1e-8)

        # ── Scheduler: linear warmup → linear decay ───────────────────────────
        steps_per_epoch    = len(train_loader)
        total_steps        = steps_per_epoch * num_epochs
        warmup_steps       = int(total_steps * warmup_ratio)
        self.scheduler     = get_linear_schedule_with_warmup(
            self.optimizer, warmup_steps, total_steps
        )
        self.steps_per_epoch = steps_per_epoch
        self.total_steps     = total_steps

        # ── History ───────────────────────────────────────────────────────────
        self.history = {
            'train_loss'      : [],
            'train_acc'       : [],
            'val_loss'        : [],
            'val_acc'         : [],
            'val_weighted_f1' : [],
            'val_macro_f1'    : [],
            'lr'              : [],
        }
        self.best_weighted_f1 = 0.0
        self.best_epoch       = 0

    def train_epoch(self, epoch: int) -> dict:
        """
        Run one training epoch.

        Returns
        -------
        dict — train_loss, train_acc, train_weighted_f1
        """
        self.model.train()
        epoch_loss  = 0.0
        epoch_preds = []
        epoch_lbls  = []
        t_epoch     = time.time()

        for step, batch in enumerate(self.train_loader, 1):
            ids   = batch['input_ids'].to(self.device)
            masks = batch['attention_mask'].to(self.device)
            lbls  = batch['labels'].to(self.device)

            # Forward
            out    = self.model(ids, masks)
            loss   = self.criterion(out['logits'], lbls)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )
            self.optimizer.step()
            self.scheduler.step()

            epoch_loss += loss.item()
            preds = out['logits'].detach().argmax(-1).cpu().numpy()
            epoch_preds.extend(preds)
            epoch_lbls.extend(lbls.cpu().numpy())

            # Progress every 10% of epoch
            if step % max(1, self.steps_per_epoch // 10) == 0:
                avg_loss = epoch_loss / step
                cur_lr   = self.scheduler.get_last_lr()[0]
                elapsed  = time.time() - t_epoch
                eta      = elapsed / step * (self.steps_per_epoch - step)
                print(
                    f'   Epoch {epoch}/{self.num_epochs}  '
                    f'Step {step:>5}/{self.steps_per_epoch}  '
                    f'Loss: {avg_loss:.4f}  '
                    f'LR: {cur_lr:.2e}  '
                    f'ETA: {eta:.0f}s'
                )

        train_loss = epoch_loss / self.steps_per_epoch
        train_acc  = accuracy_score(epoch_lbls, epoch_preds)
        train_wf1  = f1_score(epoch_lbls, epoch_preds,
                              average='weighted', zero_division=0)

        return {
            'train_loss'      : train_loss,
            'train_acc'       : train_acc,
            'train_weighted_f1': train_wf1,
            'epoch_time'      : time.time() - t_epoch,
        }

    def fit(self, save_best: bool = True) -> dict:
        """
        Full training loop — paper: 4 epochs.

        Parameters
        ----------
        save_best : save best model by Weighted F1

        Returns
        -------
        dict — complete training history
        """
        print(f'🚀 Training SecurityBERT …')
        print(f'   Epochs      : {self.num_epochs}')
        print(f'   Steps/epoch : {self.steps_per_epoch:,}')
        print(f'   Total steps : {self.total_steps:,}')
        print(f'   Device      : {self.device}')
        print('=' * 70)

        t_total = time.time()

        for epoch in range(1, self.num_epochs + 1):
            # ── Train ─────────────────────────────────────────────────────────
            train_m = self.train_epoch(epoch)

            # ── Validate ──────────────────────────────────────────────────────
            val_m = evaluate(
                self.model, self.val_loader, self.criterion,
                self.device, self.label_map, self.num_classes,
            )

            # ── Log ───────────────────────────────────────────────────────────
            self.history['train_loss'].append(train_m['train_loss'])
            self.history['train_acc'].append(train_m['train_acc'])
            self.history['val_loss'].append(val_m['loss'])
            self.history['val_acc'].append(val_m['accuracy'])
            self.history['val_weighted_f1'].append(val_m['weighted_f1'])
            self.history['val_macro_f1'].append(val_m['macro_f1'])
            self.history['lr'].append(self.scheduler.get_last_lr()[0])

            # ── Print summary ─────────────────────────────────────────────────
            print(f'\n{"─"*70}')
            print(f'  Epoch {epoch}/{self.num_epochs}  '
                  f'({train_m["epoch_time"]:.0f}s)')
            print(f'  Train  →  Loss: {train_m["train_loss"]:.4f}  '
                  f'Acc: {train_m["train_acc"]*100:.2f}%')
            print(f'  Val    →  Loss: {val_m["loss"]:.4f}  '
                  f'Acc: {val_m["accuracy"]*100:.2f}%')
            print(f'  Val F1 →  Weighted: {val_m["weighted_f1"]:.4f}  '
                  f'Macro: {val_m["macro_f1"]:.4f}')

            # Recall on critical minority classes
            for cls in ['MITM', 'Fingerprinting', 'Ransomware']:
                recall = val_m['per_class_recall'].get(cls, 0.0)
                print(f'  Recall {cls}: {recall:.4f}')

            # ── Save best ─────────────────────────────────────────────────────
            if save_best and val_m['weighted_f1'] > self.best_weighted_f1:
                self.best_weighted_f1 = val_m['weighted_f1']
                self.best_epoch       = epoch
                state = {
                    'epoch'            : epoch,
                    'model_state_dict' : self.model.state_dict(),
                    'optimizer_state'  : self.optimizer.state_dict(),
                    'val_weighted_f1'  : val_m['weighted_f1'],
                    'val_macro_f1'     : val_m['macro_f1'],
                    'val_accuracy'     : val_m['accuracy'],
                    'config'           : self.model.bert.config.to_dict()
                                         if hasattr(self.model, 'bert') else {},
                    'label_map'        : self.label_map,
                    'history'          : self.history,
                }
                best_path = self.checkpoint_dir / 'best_model.pt'
                torch.save(state, best_path)
                print(f'  💾 Best model saved (WF1={self.best_weighted_f1:.4f})')

            print(f'{"─"*70}\n')

        # ── Save last ─────────────────────────────────────────────────────────
        last_path = self.checkpoint_dir / 'last_model.pt'
        torch.save(
            {
                'epoch'           : self.num_epochs,
                'model_state_dict': self.model.state_dict(),
                'history'         : self.history,
                'label_map'       : self.label_map,
            },
            last_path
        )

        total_time = time.time() - t_total
        print(f'✅ Training complete!')
        print(f'   Total time   : {total_time:.0f}s  ({total_time/60:.1f} min)')
        print(f'   Best epoch   : {self.best_epoch}')
        print(f'   Best WF1     : {self.best_weighted_f1:.4f}')
        print(f'   Paper target : 0.98')

        return self.history
