# 01-01: Classic CNN Architectures

Evolution of convolutional neural networks from LeNet to modern residual networks — covering the key architectural innovations that shaped modern deep learning.

## Notebooks

| Notebook | Topic | Key Concepts |
|----------|-------|-------------|
| `00-ConvolutionalNetwork.ipynb` | CNN Fundamentals | Convolution, pooling, receptive field, feature maps |
| `01-AlexNet.ipynb` | AlexNet (2012) | ReLU, dropout, data augmentation, GPU training |
| `02-GoogLeNet.ipynb` | GoogLeNet / Inception (2014) | Inception module, 1×1 conv, auxiliary classifiers |
| `03-ResNet-SqueezeAndExcitation.ipynb` | ResNet + SE-Net (2015/2018) | Residual connections, skip connections, channel attention |

## Architecture Timeline

```
LeNet-5 (1998)       — First successful CNN, handwritten digit recognition
    ↓
AlexNet (2012)       — Deep CNN on GPU, won ImageNet by 10% margin
    ↓
VGGNet (2014)        — Very deep networks with 3×3 convolutions only
    ↓
GoogLeNet (2014)     — Inception module, parallel multi-scale filters
    ↓
ResNet (2015)        — Residual connections, enabled 100+ layer networks
    ↓
SE-Net (2018)        — Squeeze-and-Excitation: channel-wise attention
```

## Key Innovations

| Year | Model | Innovation |
|------|-------|-----------|
| 1998 | LeNet | Convolution + pooling pipeline |
| 2012 | AlexNet | ReLU, dropout, GPU, data augmentation |
| 2014 | VGGNet | Depth via small 3×3 filters only |
| 2014 | GoogLeNet | Inception module (1×1, 3×3, 5×5 in parallel) |
| 2015 | ResNet | Skip connections → train 152-layer networks |
| 2018 | SE-Net | Channel attention: learn which features to emphasize |

## References

| Paper | Link |
|-------|------|
| AlexNet — ImageNet Classification with Deep CNNs | [arxiv](https://arxiv.org/abs/1404.5997) |
| GoogLeNet — Going Deeper with Convolutions | [arxiv](https://arxiv.org/abs/1409.4842) |
| ResNet — Deep Residual Learning for Image Recognition | [arxiv](https://arxiv.org/abs/1512.03385) |
| SE-Net — Squeeze-and-Excitation Networks | [arxiv](https://arxiv.org/abs/1709.01507) |
