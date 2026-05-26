# 04-1: Generative Adversarial Networks

Training two networks in competition — a generator that creates fake data and a discriminator that tries to tell real from fake — until the generator produces indistinguishable samples.

## Notebooks

| Notebook | Topic | Key Concepts |
|----------|-------|-------------|
| `01-GANs.ipynb` | Vanilla GAN (2014) | Minimax game, BCE loss, mode collapse, training instability |
| `02-DCGAN.ipynb` | DCGAN (2015) | Strided conv, BatchNorm, LeakyReLU, architectural guidelines |
| `03-WGAN.ipynb` | WGAN (2017) | Wasserstein distance, weight clipping / gradient penalty, training stability |

## The GAN Framework

```
Noise z ~ N(0,1)
    ↓
Generator G(z) → fake image
    ↓
Discriminator D → real or fake?
    ↑
Real images from dataset
```

**Minimax objective:**
```
min_G max_D  E[log D(x)] + E[log(1 - D(G(z)))]
```

The discriminator maximizes its ability to distinguish real/fake.
The generator minimizes the discriminator's success (maximize D(G(z))).

## GAN Family Evolution

| Model | Year | Key Improvement | Problem Fixed |
|-------|------|-----------------|---------------|
| **GAN** | 2014 | Original framework | — (baseline) |
| **DCGAN** | 2015 | CNN architecture guidelines | Training instability with raw MLPs |
| **WGAN** | 2017 | Wasserstein loss + Lipschitz constraint | Mode collapse, vanishing gradients |
| **StyleGAN** | 2019 | Style-based generator, AdaIN | Control over image attributes |
| **BigGAN** | 2019 | Large-scale class-conditional | High-fidelity diverse samples |

## Common GAN Problems

| Problem | Symptom | Fix |
|---------|---------|-----|
| **Mode collapse** | Generator always outputs same image | WGAN, minibatch discrimination |
| **Vanishing gradient** | Discriminator too good → G gets no signal | WGAN, label smoothing |
| **Training instability** | Loss oscillates, never converges | DCGAN guidelines, WGAN-GP |
| **Checkerboard artifacts** | Grid pattern in generated images | Upsample + Conv instead of TransposeConv |

## DCGAN Design Rules

1. Replace pooling with **strided convolutions** (discriminator) and fractional-strided convolutions (generator)
2. Use **BatchNorm** in both G and D (except output of G and input of D)
3. Remove fully connected hidden layers
4. Use **ReLU** in G (except output: Tanh), **LeakyReLU** in D

## References

| Paper | Link |
|-------|------|
| GAN — Generative Adversarial Nets | [arxiv](https://arxiv.org/abs/1406.2661) |
| DCGAN — Unsupervised Representation Learning with DCGANs | [arxiv](https://arxiv.org/abs/1511.06434) |
| WGAN — Wasserstein GAN | [arxiv](https://arxiv.org/abs/1701.07875) |
| WGAN-GP — Improved Training of Wasserstein GANs | [arxiv](https://arxiv.org/abs/1704.00028) |
