# 01-02: Vision Transformers

Applying the Transformer architecture to computer vision — from the original ViT that treats image patches as tokens, to Swin Transformer that introduces hierarchical windows for efficiency.

## Notebooks

| Notebook | Topic | Key Concepts |
|----------|-------|-------------|
| `01-ViT.ipynb` | Vision Transformer (2020) | Patch embedding, positional encoding, class token, multi-head self-attention |
| `02-Swin-Transformer.ipynb` | Swin Transformer (2021) | Shifted window attention, hierarchical feature maps, linear complexity |

## How Vision Transformers Work

```
Image (224×224)
  → Split into 16×16 patches → 196 tokens
  → Linear projection → patch embeddings
  → + [CLS] token + positional encoding
  → Transformer Encoder (L layers)
        Multi-Head Self-Attention
        Feed-Forward Network
        LayerNorm + residual
  → [CLS] token → Linear classifier → class
```

## ViT vs Swin Transformer

| | ViT | Swin Transformer |
|---|---|---|
| **Attention** | Global (all patches) | Local window (7×7) + shifted |
| **Complexity** | O(N²) — N patches | O(N) — linear via windows |
| **Feature map** | Single scale | Hierarchical (like ResNet) |
| **Best for** | Large datasets (JFT-300M) | General vision tasks |
| **FPN compatible** | No (flat) | Yes (multi-scale outputs) |

## Key Innovations

**ViT (Dosovitskiy et al., 2020)**
- First to show pure Transformer (no convolution) can match CNNs on image classification
- Requires large-scale pre-training (ImageNet-21k or JFT-300M)
- 16×16 patch size → 196 tokens for 224×224 image

**Swin Transformer (Liu et al., 2021)**
- Shifted Window: odd layers use non-overlapping windows, even layers shift by half-window — enabling cross-window connections
- Hierarchical: 4 stages with patch merging (4×, 8×, 16×, 32× downsampling)
- Outperforms ViT on downstream tasks (detection, segmentation) with fewer FLOPs

## References

| Paper | Link |
|-------|------|
| ViT — An Image is Worth 16×16 Words | [arxiv](https://arxiv.org/abs/2010.11929) |
| Swin Transformer — Hierarchical Vision Transformer using Shifted Windows | [arxiv](https://arxiv.org/abs/2103.14030) |
| DeiT — Training data-efficient image transformers | [arxiv](https://arxiv.org/abs/2012.12877) |
