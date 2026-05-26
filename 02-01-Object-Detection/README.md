# 02-01: Object Detection

From classical region-based methods to real-time single-shot detectors — covering every major architectural innovation in object detection from 2014 to 2023.

## Notebooks

### Foundations
| Notebook | Topic |
|----------|-------|
| `00-IoU-and-mAP.ipynb` | IoU, GIoU, DIoU, CIoU, NMS, Soft-NMS, mAP evaluation |

### Two-Stage Detectors (`single-stage/`)
> Propose regions first, then classify each region

| Notebook | Topic | Key Concepts |
|----------|-------|-------------|
| `single-stage/01-R-CNN.ipynb` | R-CNN (2014) | Selective Search, VGG16 features, SVM, bbox regression |
| `single-stage/02-Fast-R-CNN.ipynb` | Fast R-CNN (2015) | RoI Pooling, shared feature map, multi-task loss |
| `single-stage/03-Faster-R-CNN.ipynb` | Faster R-CNN (2016) | RPN, anchor boxes, end-to-end training |

### Single-Stage Detectors (`two-stage/`)
> Predict boxes and classes in a single forward pass

| Notebook | Topic | Key Concepts |
|----------|-------|-------------|
| `two-stage/01-YOLO.ipynb` | YOLO v1/v2 (2016/2017) | Grid prediction, anchor boxes, YOLO loss |
| `two-stage/02-YOLOv3.ipynb` | YOLOv3 (2018) | Darknet-53, multi-scale detection, Darknet config |
| `two-stage/03-YOLOv4.ipynb` | YOLOv4 (2020) | CSPDarknet, PANet, BoF/BoS tricks |
| `two-stage/04-YOLOv7.ipynb` | YOLOv7 (2022) | ELAN, auxiliary head, re-parameterization |

## Detection Evolution

```
R-CNN (2014)         47 sec/img   66.0% VOC07
Fast R-CNN (2015)     2.3 sec/img  70.0% VOC07   ← RoI Pooling
Faster R-CNN (2016)   0.2 sec/img  73.2% VOC07   ← RPN
YOLO v1 (2016)       45 fps       63.4% VOC07    ← single-shot
YOLO v3 (2018)       35 fps       ~57 AP50 COCO
YOLOv7 (2022)        161 fps      55.9 AP COCO
```

## IoU Variants

| Metric | What it adds | Used in |
|--------|-------------|---------|
| IoU | Basic overlap ratio | NMS, mAP evaluation |
| GIoU | Enclosing box penalty (non-zero gradient) | Box regression loss |
| DIoU | + Center distance penalty | Box loss, DIoU-NMS |
| CIoU | + Aspect ratio consistency | YOLOv5/7/8 default loss |

## References

| Paper | Link |
|-------|------|
| R-CNN — Rich Feature Hierarchies | [arxiv](https://arxiv.org/abs/1311.2524) |
| Fast R-CNN | [arxiv](https://arxiv.org/abs/1504.08083) |
| Faster R-CNN | [arxiv](https://arxiv.org/abs/1506.01497) |
| YOLO v1 — You Only Look Once | [arxiv](https://arxiv.org/abs/1506.02640) |
| YOLOv3 | [arxiv](https://arxiv.org/abs/1804.02767) |
| YOLOv4 | [arxiv](https://arxiv.org/abs/2004.10934) |
| YOLOv7 | [arxiv](https://arxiv.org/abs/2207.02696) |
| GIoU | [arxiv](https://arxiv.org/abs/1902.09630) |
| DIoU / CIoU | [arxiv](https://arxiv.org/abs/1911.08287) |
