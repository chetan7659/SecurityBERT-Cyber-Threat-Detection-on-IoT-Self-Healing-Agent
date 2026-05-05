"""
SecurityBERT Model Architecture.
SecurityBERT Paper — Section III-E, III-F, III-H, Figure 5, Table III/IV.

15-layer BERT-based model with:
    - BERT Embeddings (word + position + token_type)
    - 4 Encoder Layers (self-attention + FFN)
    - BERT Pooler
    - Dropout + Linear classifier
    - Softmax output
"""

import math
from pathlib import Path
from typing  import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel


# ─────────────────────────────────────────────────────────────────────────────
# Config builder
# ─────────────────────────────────────────────────────────────────────────────

def build_config(
    vocab_size          : int   = 5000,
    hidden_size         : int   = 128,
    num_hidden_layers   : int   = 2,
    num_attention_heads : int   = 4,
    intermediate_size   : int   = 512,
    max_position_embeddings: int= 512,
    hidden_dropout_prob : float = 0.1,
    attention_probs_dropout_prob: float = 0.1,
    layer_norm_eps      : float = 1e-12,
    num_classes         : int   = 15,
    pad_token_id        : int   = 1,
) -> BertConfig:
    """
    Build BertConfig with paper-exact parameters (Table III + Table IV).

    Parameters — directly from paper Table IV:
        vocab_size           = 5,000
        hidden_size          = 128
        num_hidden_layers    = 2  (×2 = 4 actual encoder layers)
        num_attention_heads  = 4
        intermediate_size    = 512
        max_position_embeddings = 512

    Returns
    -------
    BertConfig
    """
    return BertConfig(
        vocab_size                   = vocab_size,
        hidden_size                  = hidden_size,
        num_hidden_layers            = num_hidden_layers,
        num_attention_heads          = num_attention_heads,
        intermediate_size            = intermediate_size,
        hidden_act                   = 'gelu',
        hidden_dropout_prob          = hidden_dropout_prob,
        attention_probs_dropout_prob = attention_probs_dropout_prob,
        max_position_embeddings      = max_position_embeddings,
        type_vocab_size              = 2,
        initializer_range            = 0.02,
        layer_norm_eps               = layer_norm_eps,
        pad_token_id                 = pad_token_id,
        num_labels                   = num_classes,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SecurityBERT — Pre-training model
# ─────────────────────────────────────────────────────────────────────────────

class SecurityBERT(nn.Module):
    """
    SecurityBERT — Pre-training model (Notebooks 05, 06, 07).

    Architecture (Paper Figure 5):
        INPUT (input_ids, attention_mask)
            ↓
        BertModel
            ├── BertEmbeddings  (word + position + type + LN + Dropout)
            ├── BertEncoder     (4 × BertLayer)
            └── BertPooler      (Linear(128→128) + Tanh)
            ↓
        Dropout(p=0.1)
            ↓
        Linear(128 → num_classes)
            ↓
        Logits (batch × num_classes)

    Parameters (Paper Table IV):
        hidden_size        = 128
        num_hidden_layers  = 2  (= 4 encoder layers)
        num_attention_heads= 4
        intermediate_size  = 512
        total params       ≈ 11,174,415
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.bert       = BertModel(config, add_pooling_layer=True)
        self.dropout    = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
        self,
        input_ids            : torch.Tensor,
        attention_mask       : torch.Tensor,
        token_type_ids       : torch.Tensor = None,
        output_attentions    : bool = False,
        output_hidden_states : bool = False,
    ) -> dict:
        """
        Forward pass.

        Parameters
        ----------
        input_ids            : (batch, seq_len) — BBPE token IDs
        attention_mask       : (batch, seq_len) — 1=real, 0=pad
        token_type_ids       : (batch, seq_len) — segment IDs
        output_attentions    : return attention weights
        output_hidden_states : return all hidden states

        Returns
        -------
        dict:
            logits        : (batch, num_classes)
            pooled_output : (batch, hidden_size)
            last_hidden   : (batch, seq_len, hidden_size)
            all_attentions: list of attention tensors (if requested)
            all_hidden    : list of hidden state tensors (if requested)
        """
        outputs = self.bert(
            input_ids             = input_ids,
            attention_mask        = attention_mask,
            token_type_ids        = token_type_ids,
            output_attentions     = output_attentions,
            output_hidden_states  = output_hidden_states,
        )

        pooled = self.dropout(outputs.pooler_output)
        logits = self.classifier(pooled)

        return {
            'logits'       : logits,
            'pooled_output': pooled,
            'last_hidden'  : outputs.last_hidden_state,
            'all_attentions': outputs.attentions
                              if output_attentions else None,
            'all_hidden'   : outputs.hidden_states
                              if output_hidden_states else None,
        }

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# SecurityBERTWithSoftmax — Fine-tuning model
# ─────────────────────────────────────────────────────────────────────────────

class SecurityBERTWithSoftmax(nn.Module):
    """
    SecurityBERT with explicit Softmax head — Paper Section III-H.

    "We added one linear layer followed by a Softmax activation
    function on top of the pre-trained SecurityBERT model."

    Architecture (Figure 5 + Section III-H):
        BertModel
            ↓
        BertPooler → [CLS] ∈ R¹²⁸
            ↓
        Dropout(p=0.1)
            ↓
        Linear(128 → num_classes)     ← "one linear layer"
            ↓
        Softmax(dim=-1)               ← "Softmax activation function"
            ↓
        Class probabilities ∈ R¹⁵    (sums to 1.0)
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.bert       = BertModel(config, add_pooling_layer=True)
        self.dropout    = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.softmax    = nn.Softmax(dim=-1)

    def forward(
        self,
        input_ids      : torch.Tensor,
        attention_mask : torch.Tensor,
        token_type_ids : torch.Tensor = None,
    ) -> dict:
        """
        Forward pass with Softmax output.

        Returns
        -------
        dict:
            logits        : (batch, num_classes) — raw scores
            probs         : (batch, num_classes) — softmax probs
            pooled_output : (batch, hidden_size)
        """
        outputs = self.bert(
            input_ids      = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
        )
        pooled = self.dropout(outputs.pooler_output)
        logits = self.classifier(pooled)
        probs  = self.softmax(logits)

        return {
            'logits'        : logits,
            'probs'         : probs,
            'pooled_output' : pooled,
        }

    def predict(
        self,
        input_ids     : torch.Tensor,
        attention_mask: torch.Tensor,
        device        : torch.device = None,
    ) -> tuple:
        """
        Inference — returns predicted class + confidence score.

        Returns
        -------
        predicted_classes : (batch,) int tensor
        confidences       : (batch,) float tensor
        """
        self.eval()
        if device:
            input_ids      = input_ids.to(device)
            attention_mask = attention_mask.to(device)
        with torch.no_grad():
            out   = self.forward(input_ids, attention_mask)
            probs = out['probs']
            preds = probs.argmax(dim=-1)
            confs = probs.max(dim=-1).values
        return preds.cpu(), confs.cpu()

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
