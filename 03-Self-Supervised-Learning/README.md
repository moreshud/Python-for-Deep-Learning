# 03: Self-Supervised Learning

Learning visual representations without any human labels — using the data itself as supervision. Covers reconstruction-based, contrastive, and feature prediction methods.

## Structure

```
03-Self-Supervised-Learning/
├── 01-Reconstruction/          ← Learn by reconstructing masked input
│   └── 01-MAE.ipynb
├── 02-ContrastiveLearning/     ← Learn by attracting similar, repelling different
│   ├── 01-SimCLR.ipynb
│   └── 02-MoCo.ipynb
├── 03-FeaturePrediction/       ← Learn by predicting teacher's output
│   ├── 01-BYOL.ipynb
│   └── 02-DINO.ipynb
└── 04-Text-to-Image/           ← (coming soon)
```

## Notebooks

### 01 — Reconstruction
| Notebook | Paper | Key Concepts |
|----------|-------|-------------|
| `01-MAE.ipynb` | MAE (He et al., 2021) | 75% masking, ViT encoder on visible patches only, lightweight decoder, MSE pixel loss |

### 02 — Contrastive Learning
| Notebook | Paper | Key Concepts |
|----------|-------|-------------|
| `01-SimCLR.ipynb` | SimCLR (Chen et al., 2020) | NT-Xent loss, 2-view augmentation, MLP projector, large batch |
| `02-MoCo.ipynb` | MoCo (He et al., 2020) | Momentum encoder, FIFO queue, InfoNCE loss |

### 03 — Feature Prediction
| Notebook | Paper | Key Concepts |
|----------|-------|-------------|
| `01-BYOL.ipynb` | BYOL (Grill et al., 2020) | No negative pairs, online/target network, EMA, predictor MLP |
| `02-DINO.ipynb` | DINO (Caron et al., 2021) | Self-distillation, multi-crop, centering trick, temperature sharpening |

## Method Comparison

| Method | Category | Negatives needed | Key trick | Loss |
|--------|----------|-----------------|-----------|------|
| **SimCLR** | Contrastive | Yes (large batch) | 2-view augmentation | NT-Xent |
| **MoCo** | Contrastive | Yes (queue K=4096) | Momentum encoder + queue | InfoNCE |
| **BYOL** | Feature pred | No | Asymmetric predictor + EMA | Cosine similarity |
| **DINO** | Feature pred | No | Centering + temperature | Cross-entropy |
| **MAE** | Reconstruction | No | 75% masking, encoder on visible only | MSE |

## How Each Avoids Collapse

| Method | Collapse Prevention |
|--------|-------------------|
| SimCLR / MoCo | Explicit negative pairs — push different views apart |
| BYOL | Asymmetric architecture (predictor only on online, not target) |
| DINO | Centering (subtract running mean from teacher logits) |
| MAE | Reconstruction target is deterministic (masked pixels) |

## Evaluation Protocol

All methods use **linear evaluation**: freeze the pre-trained encoder, train a single linear layer on labeled data. This tests whether the representation is linearly separable without fine-tuning.

## References

| Paper | Link |
|-------|------|
| SimCLR — A Simple Framework for Contrastive Learning | [arxiv](https://arxiv.org/abs/2002.05709) |
| MoCo v2 — Improved Baselines with Momentum Contrastive Learning | [arxiv](https://arxiv.org/abs/2003.04297) |
| BYOL — Bootstrap Your Own Latent | [arxiv](https://arxiv.org/abs/2006.07733) |
| DINO — Self-Distillation with No Labels | [arxiv](https://arxiv.org/abs/2104.14294) |
| MAE — Masked Autoencoders Are Scalable Vision Learners | [arxiv](https://arxiv.org/abs/2111.06377) |
