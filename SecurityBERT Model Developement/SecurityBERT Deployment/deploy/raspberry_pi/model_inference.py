"""
SecurityBERT + PPO inference engine for Raspberry Pi.
Loads both models and runs the full pipeline:
  PPFLE sequence → BBPE tokens → SecurityBERT → PPO Agent → action
"""

import time
import json
import math
import logging
from pathlib    import Path
from typing     import Dict, List, Tuple, Optional
from collections import deque

import numpy as np
import torch
import torch.nn            as nn
import torch.nn.functional as F
from torch.distributions   import Categorical
from transformers          import BertConfig, BertModel
from tokenizers            import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

logger = logging.getLogger(__name__)

# ── Shared constants — match NB11/NB12/NB13 exactly ──────────────────────────
CLASS_NAMES = [
    'Backdoor','DDoS_HTTP','DDoS_ICMP','DDoS_TCP','DDoS_UDP',
    'Fingerprinting','MITM','Normal','Password','Port_Scanning',
    'Ransomware','SQL_injection','Uploading','Vulnerability_scanner','XSS',
]
ACTION_NAMES = [
    'BLOCK_IP','RESET_CONNECTION','RESTART_SERVICE',
    'ISOLATE_DEVICE','LOG_AND_ALERT',
]
OPTIMAL_ACTION = {
    'Normal':4,'Fingerprinting':4,
    'Port_Scanning':0,'DDoS_TCP':0,'DDoS_UDP':0,'DDoS_ICMP':0,'DDoS_HTTP':0,
    'SQL_injection':1,'XSS':1,'MITM':1,
    'Uploading':2,'Backdoor':2,'Password':2,'Vulnerability_scanner':2,
    'Ransomware':3,
}
N_CLASSES   = 15;  N_ACTIONS = 5
N_NET_STATS = 10;  N_HISTORY = 5
STATE_DIM   = N_CLASSES + N_NET_STATS + N_HISTORY   # 30
HIDDEN_DIM  = 128

# Network stats signatures per attack (for state building)
_NET_SIGS = {
    'DDoS_TCP'    :[0.9,0.6,0.9,0.7,0.9,0.7,0.9,0.8,0.8,0.8],
    'DDoS_UDP'    :[0.9,0.7,0.8,0.6,0.9,0.8,0.9,0.9,0.7,0.8],
    'DDoS_ICMP'   :[0.8,0.5,0.7,0.5,0.8,0.6,0.8,0.8,0.6,0.7],
    'DDoS_HTTP'   :[0.7,0.4,0.8,0.8,0.7,0.5,0.8,0.7,0.5,0.7],
    'Ransomware'  :[0.3,0.2,0.9,0.9,0.3,0.4,0.6,0.3,0.4,0.9],
    'SQL_injection':[0.4,0.1,0.5,0.4,0.4,0.3,0.3,0.3,0.2,0.6],
    'MITM'        :[0.3,0.3,0.4,0.4,0.3,0.4,0.4,0.5,0.3,0.7],
    'Backdoor'    :[0.2,0.1,0.6,0.7,0.2,0.3,0.4,0.2,0.2,0.8],
    'Port_Scanning':[0.6,0.1,0.3,0.2,0.6,0.2,0.4,0.3,0.3,0.4],
    'Normal'      :[0.3,0.1,0.4,0.5,0.3,0.1,0.4,0.2,0.1,0.1],
}
_NET_DEFAULT = [0.3,0.1,0.4,0.5,0.3,0.1,0.4,0.2,0.1,0.1]


# ── Model classes (must match NB08 and NB12 exactly) ─────────────────────────
class SecurityBERTWithSoftmax(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.bert       = BertModel(config, add_pooling_layer=True)
        self.dropout    = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.softmax    = nn.Softmax(dim=-1)

    def forward(self, input_ids, attention_mask):
        out    = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(out.pooler_output)
        logits = self.classifier(pooled)
        return logits, self.softmax(logits)


class PPOActorCritic(nn.Module):
    """Identical to NB11/NB12 — canonical attribute names."""
    def __init__(self, state_dim=STATE_DIM, action_dim=N_ACTIONS,
                 hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
        )
        self.actor_head  = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, action_dim)
        )
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def get_action(self, state: torch.Tensor) -> Tuple[int, float]:
        with torch.no_grad():
            logits = self.actor_head(self.backbone(state))
            probs  = torch.softmax(logits, dim=-1)
        return probs.argmax(-1).item(), probs.max(-1).values.item()


class SecurityBERTInference:
    """
    Full inference pipeline:
    raw packet features → PPFLE → BBPE → SecurityBERT → PPO → action
    """

    def __init__(
        self,
        bert_ckpt_path : str,
        ppo_ckpt_path  : str,
        tokenizer_dir  : str,
        conf_threshold : float = 0.60,
        device         : str   = 'cpu',
        max_seq_len    : int   = 512,
    ):
        self.device        = torch.device(device)
        self.conf_threshold= conf_threshold
        self.max_seq_len   = max_seq_len
        self.action_history= deque([0] * N_HISTORY, maxlen=N_HISTORY)
        self._rng          = np.random.RandomState(42)

        self._load_tokenizer(tokenizer_dir)
        self._load_bert(bert_ckpt_path)
        self._load_ppo(ppo_ckpt_path)

        logger.info(
            f'SecurityBERT pipeline ready — '
            f'BERT: {sum(p.numel() for p in self.bert.parameters()):,} params  '
            f'PPO: {sum(p.numel() for p in self.ppo.parameters()):,} params'
        )

    # ── Load methods ──────────────────────────────────────────────────────────

    def _load_tokenizer(self, tokenizer_dir: str) -> None:
        td = Path(tokenizer_dir)
        self.tokenizer = ByteLevelBPETokenizer(
            vocab  = str(td / 'vocab.json'),
            merges = str(td / 'merges.txt'),
        )
        self.tokenizer.post_processor = BertProcessing(
            sep=('</s>', self.tokenizer.token_to_id('</s>')),
            cls=('<s>',  self.tokenizer.token_to_id('<s>')),
        )
        self.tokenizer.enable_truncation(max_length=self.max_seq_len)
        self.tokenizer.enable_padding(
            pad_id    = self.tokenizer.token_to_id('<pad>'),
            pad_token = '<pad>',
            length    = self.max_seq_len,
        )
        logger.info(f'Tokenizer loaded: {self.tokenizer.get_vocab_size():,} vocab')

    def _load_bert(self, path: str) -> None:
        ckpt         = torch.load(path, map_location=self.device,
                                  weights_only=False)
        config       = BertConfig(**ckpt['config'])
        self.bert    = SecurityBERTWithSoftmax(config).to(self.device)
        self.bert.load_state_dict(ckpt['model_state_dict'])
        self.bert.eval()
        self.label_map   = ckpt['label_map']
        logger.info(
            f'BERT loaded — acc={ckpt.get("val_accuracy",0)*100:.1f}%'
        )

    def _load_ppo(self, path: str) -> None:
        ckpt      = torch.load(path, map_location=self.device,
                               weights_only=False)
        self.ppo  = PPOActorCritic(
            state_dim  = ckpt['state_dim'],
            action_dim = ckpt['action_dim'],
            hidden_dim = ckpt.get('hidden_dim', HIDDEN_DIM),
        ).to(self.device)
        self.ppo.load_state_dict(ckpt['model_state_dict'])
        self.ppo.eval()
        logger.info(
            f'PPO loaded — best_reward={ckpt.get("best_reward",0):.2f}'
        )

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(
        self,
        ppfle_sequence : str,
        packet_features: Optional[Dict] = None,
    ) -> Dict:
        """
        Run full pipeline on one PPFLE-encoded sequence.

        Returns
        -------
        dict:
            predicted_class, confidence, all_probs,
            action_id, action_name,
            bert_ms, ppo_ms, total_ms
        """
        # ── Step 1: BBPE tokenize ─────────────────────────────────────────────
        enc   = self.tokenizer.encode(ppfle_sequence)
        ids   = torch.tensor([enc.ids],            dtype=torch.long).to(self.device)
        masks = torch.tensor([enc.attention_mask], dtype=torch.long).to(self.device)

        # ── Step 2: SecurityBERT ──────────────────────────────────────────────
        t0 = time.perf_counter()
        with torch.no_grad():
            _, probs = self.bert(ids, masks)
        probs_np   = probs[0].cpu().numpy()
        pred_idx   = int(probs_np.argmax())
        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(probs_np[pred_idx])
        bert_ms    = (time.perf_counter() - t0) * 1000

        # ── Step 3: Build PPO state ───────────────────────────────────────────
        if packet_features is not None:
            net_stats = self._net_from_features(packet_features)
        else:
            sig       = np.array(
                _NET_SIGS.get(pred_class, _NET_DEFAULT), np.float32
            )
            net_stats = np.clip(
                sig + self._rng.normal(0, 0.03, N_NET_STATS).astype(np.float32),
                0, 1
            )

        hist_enc = np.array(list(self.action_history),
                            dtype=np.float32) / N_ACTIONS
        state_np = np.concatenate([probs_np, net_stats, hist_enc])
        state_t  = torch.FloatTensor(state_np).unsqueeze(0).to(self.device)

        # ── Step 4: PPO decision ──────────────────────────────────────────────
        t1 = time.perf_counter()
        if confidence < self.conf_threshold:
            action_id = 4   # LOG_AND_ALERT for low-confidence
        else:
            action_id, _ = self.ppo.get_action(state_t)
        ppo_ms = (time.perf_counter() - t1) * 1000

        self.action_history.append(action_id)

        return {
            'predicted_class' : pred_class,
            'confidence'      : confidence,
            'all_probs'       : probs_np,
            'action_id'       : action_id,
            'action_name'     : ACTION_NAMES[action_id],
            'bert_ms'         : bert_ms,
            'ppo_ms'          : ppo_ms,
            'total_ms'        : bert_ms + ppo_ms,
        }

    def _net_from_features(self, features: Dict) -> np.ndarray:
        stats = np.array([
            min(features.get('tcp.connection.syn', 0) / 100.0, 1.0),
            min(features.get('tcp.connection.rst', 0) / 10.0,  1.0),
            min(features.get('tcp.len',            0) / 65535.0, 1.0),
            min(features.get('icmp.checksum',      0) / 65535.0, 1.0),
            min(features.get('tcp.ack',            0) / 1e9,   1.0),
            min(features.get('tcp.flags',          0) / 255.0, 1.0),
            min(features.get('udp.stream',         0) / 100.0, 1.0),
            min(features.get('dns.qry.name.len',   0) / 200.0, 1.0),
            min(features.get('mqtt.len',           0) / 1000.0,1.0),
            min(features.get('mbtcp.len',          0) / 100.0, 1.0),
        ], dtype=np.float32)
        return np.clip(stats, 0.0, 1.0)