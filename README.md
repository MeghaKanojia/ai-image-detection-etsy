# 🖼️ Detecting AI-Generated Product Images on the Etsy Marketplace

A three-phase comparative study of **hybrid feature extraction** and **end-to-end fine-tuning** for binary classification of AI-generated vs. authentic product images, achieving a best validation **F1 score of 0.9356** with ConvNeXt-Base.

> **Context:** The dataset is proprietary Etsy marketplace data and cannot be redistributed. `submission.csv` is excluded from this repository.

---

## 📌 Table of Contents
- [Problem Statement](#problem-statement)
- [Results Summary](#results-summary)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Key Findings & Ablations](#key-findings--ablations)
- [Conclusion](#conclusion)

---

## Problem Statement

Online marketplaces like Etsy depend on trust. Generative models (Midjourney, DALL-E, Stable Diffusion) now allow sellers to pass off AI-generated images as real product photographs. This project addresses the binary classification task:

> Given a product listing image, predict whether it is **AI-generated (1)** or an **authentic photograph (0)**.

**Performance metric:** F1 score on a hidden test set.

**Three core challenges:**
- Small dataset (~4,800 training images) — training from scratch is not viable
- Subtle signal — AI artifacts appear as slight texture irregularities, not obvious object differences
- Multi-generator dataset — images likely come from several different generative models, each leaving distinct traces

---

## Results Summary

| Phase | Approach | Best Model / Config | Val F1 |
|---|---|---|---|
| **Phase 1** | Frozen CNN features + classical ML | ResNet50 (2048-d) + SVM (RBF) | **0.8134** |
| **Phase 2** | End-to-end fine-tuning | EfficientNetV2-S (light aug, Adam) | **0.9076** |
| **Phase 3** | Single model | ConvNeXt-Base | **0.9356** |
| **Phase 3** | Multi-model ensemble + frequency features | EfficientNetV2-S+FFT, ConvNeXt-S, ConvNeXt-B | **0.9317** |

**Key jump:** End-to-end fine-tuning (Phase 2) outperformed the best hybrid model (Phase 1) by **+9.4 F1 points**, by allowing the backbone to adapt to AI-artifact texture features — something a frozen backbone cannot do regardless of the classifier placed on top.

---

## Dataset

| Split | Images | AI-generated | Authentic |
|---|---|---|---|
| Training | 3,840 | ~48% | ~52% |
| Validation | 960 | 463 (48.2%) | 497 (51.8%) |
| Test (hidden) | 2,058 | — | — |

> **Data availability:** The dataset consists of proprietary Etsy marketplace images and **cannot be shared publicly**. To reproduce experiments, you would need access to a comparable AI-vs-real image dataset. The code structure is dataset-agnostic — images are expected in a folder organised by label (see How to Run).

---

## Methodology

### Phase 1 — Hybrid Feature Extraction

Classical ML classifiers trained on three levels of image features:

| Feature Representation | Dimensions | Best Classifier | Val F1 |
|---|---|---|---|
| Raw pixels (32×32 RGB) | 3,072-d | Random Forest (100 est.) | 0.6229 |
| Hand-crafted: YCrCb + Sobel + colour histograms | 17,152-d | Random Forest (200 est.) | 0.6525 |
| Frozen ResNet50 embeddings (ImageNet) | 2,048-d | SVM (RBF, C=1.0) | **0.8134** |

The jump from hand-crafted to deep CNN features (+0.161 F1) confirms that frozen deep embeddings are vastly more discriminative than engineered descriptors for subtle texture-based tasks.

---

### Phase 2 — End-to-End Fine-Tuning (EfficientNetV2-S)

- **Backbone:** `tf_efficientnetv2_s` from `timm` — 22M parameters, 384×384 input
- **Head:** Single-logit binary classifier with `BCEWithLogitsLoss`
- **Optimiser:** Adam, lr=1×10⁻⁴, batch size=24, 10 epochs
- **Augmentation:** Random horizontal flip only (intentionally minimal)
- **Hardware:** NVIDIA RTX 4070 Ti (12 GB VRAM)

Best checkpoint saved per epoch by validation F1 → **final Val F1: 0.9076**

---

### Phase 3 — Multi-Model Ensemble with Frequency-Domain Features

Three architecturally diverse models trained together:

| Model | Architecture | Parameters | Val F1 |
|---|---|---|---|
| Model A | EfficientNetV2-S + FFT frequency branch | ~22M + MLP | 0.8930 |
| Model B | ConvNeXt-Small | ~50M | 0.9201 |
| Model C | ConvNeXt-Base | ~89M | **0.9356** |
| Equal ensemble (3-model + 5-view TTA) | — | — | 0.9310 |
| Rank-squared weighted ensemble | — | — | 0.9317 |

**FFT Branch (Model A):** Image → grayscale → 2D FFT → log-magnitude → resize to 64×64 → 2-layer MLP (4096→256→128). FFT and CNN outputs are concatenated before the classification head, allowing joint learning of spatial and frequency-domain artifacts.

**Training configuration (Phase 3):**
- Loss: Focal Loss (γ=2.0) with positive class weighting
- Optimiser: AdamW — backbone lr 2×10⁻⁵, head/FFT branch lr 1×10⁻⁴, weight decay 1×10⁻⁴
- Scheduler: OneCycleLR (epochs 1–13) → SWALR at 1×10⁻⁵ (epoch 14 onward)
- Mixup augmentation (α=0.4, p=0.5 per batch)
- 5-view TTA: original, horizontal flip, center crop, ±7° rotation, mild colour jitter
- Hardware: Google Colab T4 (15 GB RAM)

---

## Tech Stack

| Tool | Role |
|---|---|
| **PyTorch 2.5.1 + CUDA 12.1** | Model training and inference (Phase 2 & 3) |
| **timm** | Pretrained backbones (EfficientNetV2-S, ConvNeXt) |
| **TensorFlow 2.15** | ResNet50 feature extraction (Phase 1) |
| **scikit-learn 1.3** | SVM, Random Forest, Gradient Boosting (Phase 1) |
| **OpenCV 4.8** | Image preprocessing and feature extraction |
| **torch.optim.swa_utils** | Stochastic Weight Averaging (Phase 3) |
| **Google Colab (T4 GPU)** | Phase 1 & 3 training environment |
| **NVIDIA RTX 4070 Ti** | Phase 2 training environment |

---

## Project Structure

```
ai-image-detection/
├── AdvMLProject_Code/
│   ├── AdvMLProject_Phase1_and_3.ipynb    # Phase 1 (hybrid ML) + Phase 3 (ensemble)
│   └── AdvMLProject_Phase2.ipynb          # Phase 2 (EfficientNetV2-S fine-tuning)
├── AdvMLProject_Report.pdf                # Full academic report
└── README.md

# Not included in this repository:
# - Dataset images       (proprietary Etsy data — cannot be redistributed)
# - submission.csv       (competition predictions file)
```

---

## How to Run

### Prerequisites

```bash
pip install torch torchvision timm scikit-learn opencv-python tensorflow pandas numpy
```

### Expected data folder structure

```
data/
├── train/
│   ├── 0/       ← authentic images
│   └── 1/       ← AI-generated images
└── test/
    └── images/  ← unlabelled test images
```

### Phase 1 & 3 (Google Colab recommended)

1. Open `AdvMLProject_Phase1_and_3.ipynb` in Google Colab
2. Mount your Google Drive and update `DATA_DIR` to point to your image folder
3. Run cells in order — Phase 1 and Phase 3 are clearly labelled sections

### Phase 2 (Local GPU recommended)

1. Open `AdvMLProject_Phase2.ipynb` in Jupyter
2. Update `DATA_DIR` to your image folder path
3. Confirm CUDA is available: `torch.cuda.is_available()`
4. Run all cells — the best epoch checkpoint is saved automatically

---

## Key Findings & Ablations

### Aggressive augmentation hurts on this task

| Config | Augmentation | Optimiser | Epochs | Val F1 |
|---|---|---|---|---|
| Light (ours) | HFlip only | Adam | 10 | **0.9076** |
| Strong | HFlip + RandomCrop + ColorJitter + RandomErasing | AdamW | 15 | 0.8947 |

Strong augmentation **reduced F1 by 1.3 points** even with 5 extra epochs. AI-generated artifacts manifest as spatially-localised texture inconsistencies — aggressive cropping, colour jitter, and random erasing destroy the very signal the model needs to learn.

### Test-Time Augmentation (TTA) does not help

| Strategy | Threshold | Val F1 | Δ vs. Baseline |
|---|---|---|---|
| Baseline | 0.50 | 0.9076 | — |
| TTA only | 0.50 | 0.9053 | −0.0023 |
| TTA + threshold search | 0.48 | 0.9066 | −0.0010 |
| Threshold search only | 0.53 | 0.9085 | +0.0009 |

TTA with horizontal flips **reduced performance** because flipping was already used during training. Flipping at inference adds no new information and introduces prediction jitter. Threshold tuning gained 0.0009 F1 — equivalent to one image label change out of 960, well within statistical noise.

### Ensemble does not always beat the best single model

ConvNeXt-Base alone (**0.9356**) outperformed the 3-model ensemble (**0.9317**). The weaker Model A (0.8930) pulled the ensemble average down — directly contradicting the common intuition that more models always help. Ensemble benefits require diversity in *quality*, not just architecture.

---

## Conclusion

End-to-end fine-tuning of a compact modern backbone is a strong default for forensic image classification with subtle texture signals. Our best result was **Val F1 = 0.9356** (ConvNeXt-Base, single model). Inference-time tricks and ensemble strategies helped marginally or not at all — the limiting factor was model capacity and architectural choice, not inference strategy.

Future direction: CLIP-based universal detectors have shown strong generalisation across generative models and may transfer well to marketplace settings.

---

## Authors

- **Megha Kanojia** 
- **Shubin Li** 

