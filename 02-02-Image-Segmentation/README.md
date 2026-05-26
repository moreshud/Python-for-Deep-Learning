# 02-02: Image Segmentation

Going beyond bounding boxes — assigning a class label to every pixel in the image. Covers semantic segmentation (U-Net), feature pyramids (FPN), and instance segmentation (Mask R-CNN).

## Notebooks

| Notebook | Topic | Key Concepts |
|----------|-------|-------------|
| `00-FPN.ipynb` | Feature Pyramid Network (2017) | Top-down pathway, lateral connections, multi-scale features |
| `01-UNET.ipynb` | U-Net (2015) | Encoder-decoder, skip connections, medical image segmentation |
| `02-Mask-R-CNN.ipynb` | Mask R-CNN (2017) | Instance segmentation, RoI Align, mask head |

## Types of Segmentation

```
Semantic Segmentation   — each pixel gets a class label (no instance distinction)
                          "all people are labeled 'person'"

Instance Segmentation   — each object instance gets its own mask
                          "person 1, person 2, ..."

Panoptic Segmentation   — combines both: stuff (sky, road) + things (people, cars)
```

## Architecture Overview

**U-Net**
```
Encoder (contracting path): Conv → Pool → Conv → Pool ...
                                    ↓ skip connections ↓
Decoder (expanding path):  ... Upsample → Conv → Upsample → Conv
Output: per-pixel class map (H × W × n_classes)
```

**FPN**
```
Bottom-up  (ResNet): C2 → C3 → C4 → C5   (increasingly semantic, less spatial)
Top-down:            P5 → P4 → P3 → P2   (add upsampled features back)
Lateral:             1×1 conv + element-wise add at each level
→ Every level has both rich semantics AND full spatial resolution
```

**Mask R-CNN = Faster R-CNN + FPN + Mask Head**
```
Faster R-CNN backbone+RPN → region proposals
RoI Align (not RoI Pool) → fixed-size features per instance
Parallel heads:
  cls_head  → class label
  box_head  → refined bounding box
  mask_head → 28×28 binary mask per instance
```

## Key Differences

| | U-Net | FPN | Mask R-CNN |
|---|---|---|---|
| **Task** | Semantic seg | Feature backbone | Instance seg |
| **Output** | Dense pixel map | Feature pyramid | Per-instance masks |
| **Skip connections** | Direct concat | Lateral add | — |
| **Proposals** | None | None | RPN |
| **Speed** | Fast | Fast (backbone only) | Slower (two-stage) |

## References

| Paper | Link |
|-------|------|
| U-Net — Convolutional Networks for Biomedical Segmentation | [arxiv](https://arxiv.org/abs/1505.04597) |
| FPN — Feature Pyramid Networks for Object Detection | [arxiv](https://arxiv.org/abs/1612.03144) |
| Mask R-CNN | [arxiv](https://arxiv.org/abs/1703.06870) |
| FCN — Fully Convolutional Networks for Semantic Segmentation | [arxiv](https://arxiv.org/abs/1411.4038) |
