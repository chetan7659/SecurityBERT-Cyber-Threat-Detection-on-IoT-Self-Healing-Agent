"""
SecurityBERT — Source Package
BERT-based cybersecurity threat detection for IoT/IIoT networks.
Paper: "Revolutionizing Cyber Threat Detection with Large Language Models"
"""

from .ppfle    import ppfle, H, s
from .model    import SecurityBERT, SecurityBERTWithSoftmax, build_config
from .dataset  import SecurityBERTDataset, build_dataloaders
from .trainer  import FocalLoss, Trainer, evaluate
from .tokenizer import PPFLETokenizer
from .utils    import (
    set_seed,
    load_config,
    get_device,
    compute_class_weights,
    save_checkpoint,
    load_checkpoint,
)

__version__ = '1.0.0'
__author__  = 'SecurityBERT Implementation'
