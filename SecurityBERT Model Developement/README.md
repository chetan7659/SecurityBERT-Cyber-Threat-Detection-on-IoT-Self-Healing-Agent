# SecurityBERT — Cyber Threat Detection for IoT/IIoT

> *"Revolutionizing Cyber Threat Detection with Large Language Models:
> A privacy-preserving BERT-based Lightweight Model for IoT/IIoT Devices"*
> — Ferrag et al., IEEE Access 2024

---

## 🎯 Overview

SecurityBERT is a novel lightweight BERT-based architecture for
detecting **14 attack types** across **5 threat categories** in IoT/IIoT
networks, achieving **98.2% accuracy** on the Edge-IIoTset dataset.

| Property | Value |
|----------|-------|
| Dataset | Edge-IIoTset (~2.2M rows) |
| Classes | 15 (Normal + 14 attacks) |
| Parameters | 11,174,415 |
| Model size | 16.7 MB |
| Inference | <0.16s on CPU |
| Accuracy | 98.2% |

---

## 🏗️ Pipeline — 8 Steps

```
Raw CSV → Clean → Features → PPFLE → BBPE → Embed → Encode → Train → Finetune
  NB01     NB01     NB02      NB03    NB04   NB05    NB06     NB07    NB08
```

| Notebook | Step | Output |
|----------|------|--------|
| `01_dataset_utilization` | Load & clean dataset | `cleaned_dataset.csv` |
| `02_feature_extraction` | Extract 61 features | `features_extracted.csv` |
| `03_ppfle_encoding` | Privacy-preserving encoding | `ppfle_encoded.csv` |
| `04_bbpe_tokenizer` | Train BBPE tokenizer | `tokenized_sequences.pt` |
| `05_securitybert_embedding` | Embedding layer | `securitybert_init.pt` |
| `06_contextual_representation` | Self-attention encoder | `contextual_repr.pt` |
| `07_training_securitybert` | Pre-training | `best_model.pt` |
| `08_finetuning_softmax` | Fine-tuning + Evaluation | `final_model.pt` |

---

## ⚖️ Class Imbalance Strategy — Option 3

```
Problem: 1,615× imbalance (Normal vs MITM/Fingerprinting)
SMOTE failed → MD5 hash interpolation is meaningless

Solution — 3 components:
  ✅ Class Weights      → weight_i = total/(n_classes × count_i)
  ✅ Stratified Sampler → all classes in every batch
  ✅ Focal Loss γ=2     → rare attacks dominate gradient
```

---

## 🔑 Key Innovation — PPFLE

```
Raw features (numbers) → BERT cannot understand
    ↓ PPFLE (Algorithm 1)
H(column_name$value) → 32-char MD5 hex token
    ↓ BBPE Tokenizer
Subword tokens → BERT processes as language
    ↓ SecurityBERT
Attack type classification → 98.2% accuracy

Without PPFLE: 51.3% (random level)
With PPFLE:    98.2% (+46.9%)
```

---

## 🚀 Quick Start

```python
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place dataset
# data/raw/DNN-EdgeIIoT-dataset.csv

# 3. Run notebooks in order
# notebooks/01_dataset_utilization.ipynb
# notebooks/02_feature_extraction.ipynb
# ...
# notebooks/08_finetuning_softmax.ipynb

# 4. Or use src modules directly
from src.ppfle   import ppfle
from src.model   import SecurityBERT, build_config
from src.trainer import Trainer, FocalLoss
from src.utils   import load_config, set_seed, get_device

config = load_config('configs/config.yaml')
set_seed(42)
device = get_device()
```

---

## 📊 Paper Results (Table VI)

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Normal | 1.00 | 1.00 | 1.00 |
| DDoS_UDP | 1.00 | 1.00 | 1.00 |
| MITM | 1.00 | 1.00 | 1.00 |
| Ransomware | 1.00 | 0.40 | 0.57 |
| Fingerprinting | 0.00 | 0.00 | 0.00 |
| **Weighted Avg** | **0.98** | **0.98** | **0.98** |
| **Accuracy** | | | **98.2%** |

---

## 📁 Project Structure

```
SecurityBERT/
├── data/raw/          ← DNN-EdgeIIoT-dataset.csv
├── data/processed/    ← Pipeline outputs
├── notebooks/         ← 8 step-by-step notebooks
├── src/               ← Reusable Python modules
│   ├── ppfle.py       ← Algorithm 1 (PPFLE)
│   ├── tokenizer.py   ← Algorithm 2 (BBPE)
│   ├── model.py       ← SecurityBERT architecture
│   ├── dataset.py     ← Dataset + DataLoader
│   ├── trainer.py     ← FocalLoss + Trainer
│   └── utils.py       ← Config, seed, checkpoints
├── tokenizer/         ← vocab.json, merges.txt
├── checkpoints/       ← best_model.pt, final_model.pt
├── outputs/           ← Figures + Reports
├── configs/config.yaml
├── requirements.txt
└── README.md
```

---

## 📄 Citation

```bibtex
@article{ferrag2024securitybert,
  title   = {Revolutionizing Cyber Threat Detection with Large Language Models},
  author  = {Ferrag, Mohamed Amine and others},
  journal = {IEEE Access},
  year    = {2024},
  doi     = {10.1109/ACCESS.2024.3363469}
}
```
