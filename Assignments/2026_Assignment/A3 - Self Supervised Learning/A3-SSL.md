# A3: Self-Supervised Learning — Teaching Guide

---

## Opening: The Labeling Problem

**What to say:**

> "ImageNet has 1.2 million labeled images. It took years and thousands of human annotators. Your model trained on it — it has seen more labeled data than any human ever will. Now imagine you're a doctor trying to detect a rare disease. You have 200 MRI scans. That's it. Supervised learning collapses."

**The core question of this lab:**

> "What if the signal for learning came from the data itself — not from humans?"

---

## The SSL Intuition (before any code)

**Pause and ask students:**

> "If I show you two photos of the same dog — one bright, one dark, one cropped differently — do you still know it's the same dog?"

Of course. Because you understand the *concept* of a dog, not just pixel patterns. SSL tries to teach models this same invariance.

**The key idea:**

> "If two views come from the same image, their representations should be similar. If they come from different images, they should be different."

This is the entire foundation of contrastive SSL.

---

## Part 1: SimCLR (Cells 3–8)

### The story

**What to say:**

> "SimCLR (Chen et al., 2020) is the 'hello world' of modern SSL. It's simple enough to understand completely, powerful enough to match supervised learning on many benchmarks."

### Architecture walkthrough

```
Image x
 ├── Augment (random crop, flip, color jitter...) → x_i
 └── Augment (independently) → x_j

x_i → Encoder (ResNet) → h_i → Projector (MLP) → z_i ─┐
x_j → Encoder (ResNet) → h_j → Projector (MLP) → z_j ─┴─→ NT-Xent Loss
```

**Three things to explain:**

**1. Why two augmentations of the same image?**

> "That's the self-supervision signal. We're not saying 'this is a cat.' We're saying 'this crop and this crop came from the same image — they should look similar in feature space.' The label is the image identity, and we get it for free."

**2. Why a projector head?**

> "Counterintuitive result from the paper: the projector MLP actually *hurts* the representations for downstream tasks. They found that `h` (before the projector) works better for linear evaluation. The projector absorbs augmentation-specific information so the encoder doesn't have to."

**3. NT-Xent Loss (the loss function)**

> "For a batch of N images, we get 2N views. For each view, we want its pair to be closer than ALL other 2N-2 views. Think of it as a softmax over a similarity matrix."

$$\ell_{i,j} = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k \neq i} \exp(\text{sim}(z_i, z_k) / \tau)}$$

**Temperature τ (tau):**

> "τ controls how sharp the distribution is. Low τ → sharper → model is penalized more for similar negatives. High τ → softer → easier task. SimCLR uses τ=0.5 — found by ablation."

**SimCLR's dirty secret — batch size:**

SimCLR needs **large batches** (4096+) in the original paper because more negatives = better contrastive signal. With small batches, there aren't enough negatives to make the task hard.

**Pause and ask students:**

> "If your batch size is 2 (1 image → 2 views), how many negatives does each view have? Is that a useful learning signal?"
>
> Answer: Zero other images in the batch, no negatives at all. The loss would always be trivially zero.

---

### Linear Evaluation (Cell 6)

**What to say:**

> "The gold standard test for SSL: freeze the encoder completely. Train only a linear layer on top using labels. If accuracy is high, the SSL encoder learned meaningful features — not just memorized augmentation artifacts."

This is important: we're testing whether SSL features are **linearly separable** — as good as or better than supervised features.

---

## Bridge: From SimCLR to DINO

**The problem with SimCLR:**

1. **Needs large batches** — memory intensive
2. **Doesn't work well with ViT** — ViT with contrastive learning collapses without careful tricks
3. **All negatives are treated equally** — a view of a dog is equally "pushed away" from a view of another dog as from a car

**The evolution:**

```
SimCLR (2020)     Contrastive, needs big batches, ResNet-focused
    ↓
MoCo (2020)       Memory bank for negatives — smaller batches
    ↓
BYOL (2020)       No negatives at all — but needs asymmetric predictor to prevent collapse
    ↓
DINO (2021)       No negatives, no predictor, ViT-native — just a centering trick
```

**What to say:**

> "Each step removes a requirement. SimCLR needs negatives and big batches. BYOL removes negatives. DINO removes the need for architectural asymmetry. By DINO, the method is almost embarrassingly simple — and it produces the best features."

---

## Part 2: DINO (Cells 10–18)

### The story

**What to say:**

> "DINO asks: what if we train a student network to match the output of a teacher network — where the teacher is a slowly-updated copy of the student? No labels. No negatives. Just self-distillation."

### Architecture — Teacher/Student

```
global crop 1 → [Student ViT] → softmax(z / τ_s) ─┐
global crop 2 → [Student ViT] → softmax(z / τ_s) ─┤
local crop 1  → [Student ViT] → softmax(z / τ_s) ─┤→ cross-entropy loss
local crop 2  → [Student ViT] → ...               ─┘
                                                    ↑
global crop 1 → [Teacher ViT] → softmax((z - c) / τ_t)  (no grad)
global crop 2 → [Teacher ViT] → softmax((z - c) / τ_t)

Teacher weights = EMA of Student weights  (no gradient, no backprop)
```

**Three things to explain:**

**1. EMA teacher (Exponential Moving Average)**

> "The teacher is not trained by backprop. It's a running average of the student weights. Think of it as a 'smoothed-out, slower-updating' version of the student. This gives stable targets — otherwise the student would be chasing a target that's changing just as fast."

```python
teacher = momentum * teacher + (1 - momentum) * student
# momentum = 0.996 → teacher updates very slowly
```

**2. Multi-crop strategy**

> "Teacher sees only global crops (large, full context). Student sees global AND local crops (small patches). The student has to predict what the teacher sees from a tiny local view — this forces it to understand global context from local information. Harder task → better features."

**3. Centering — the key trick preventing collapse**

**This is the most important concept to explain carefully:**

> "Why doesn't DINO collapse? If teacher and student both output the same constant vector for every image, the loss would be zero. That's mode collapse."
>
> "DINO prevents this with centering: subtract a running mean from the teacher output before softmax. If one dimension dominates (the network trying to collapse), the running mean catches it and subtracts it out."

$$z_{\text{teacher,corrected}} = z_{\text{teacher}} - c$$

$$c \leftarrow m \cdot c + (1 - m) \cdot \text{mean\_batch}(z_{\text{teacher}})$$

> "Low teacher temperature (τ_t = 0.04) makes the teacher output sharp and confident. High student temperature (τ_s = 0.1) makes the student softer. The student has to match a sharp teacher — this sharpness is what forces learning."

---

### The "Wow" Moment: Attention Maps (Cell 17)

**Build up to this — don't just show the output.**

**What to say:**

> "DINO was trained with zero segmentation labels. Zero bounding box labels. We just told it: 'these crops came from the same image.' Now watch what its attention does."

> "In a Vision Transformer, the [CLS] token attends to every patch. After DINO training, the [CLS] token has learned to attend to the *semantically relevant* patches — the foreground object — and ignore the background."

> "Nobody told it what the foreground was. Nobody drew a single mask. It emerged purely from the SSL objective."

**This is why DINO is famous.** Visualization of this result in the original paper shocked the community — people didn't expect unsupervised segmentation to emerge from a classification-style pretext task.

**Pause and ask students:**

> "What does this tell us about what the model is actually learning? Is it memorizing pixel patterns, or learning something more abstract?"

---

## SimCLR vs DINO — Side by Side

| | SimCLR | DINO |
|---|---|---|
| **Negatives** | Yes (all other images in batch) | No |
| **Backbone** | ResNet (works poorly with ViT) | ViT-native |
| **Batch size** | Needs large (4096+) | Works with small batches |
| **Collapse prevention** | Negatives push representations apart | Centering + EMA teacher |
| **Key result** | Good linear eval | Unsupervised segmentation emerges |
| **Temperature** | Single τ on loss | Separate τ for teacher/student |

---

## t-SNE Comparison (Cells 20–21)

**What to say:**

> "t-SNE projects high-dimensional features into 2D. If SSL worked, images of the same class should cluster together — even though the model never saw class labels during training."

**What to look for:**
- Well-separated clusters → good representations
- Mixed clusters → model learned augmentation invariance but not semantic meaning
- SimCLR with ResNet vs DINO with ViT — expect DINO to show cleaner class separation

---

## Exercise Walkthrough Tips

**Ex1 (DINO centering ablation):** This is the most valuable exercise. Without centering, the teacher distribution collapses to one or two dominant dimensions. The attention maps become meaningless — uniform attention over all patches. Students should see the loss stop decreasing (trivial solution) and the attention maps go flat.

**The center norm question:** It should stabilize, not grow unboundedly. If it grows, the centering update is chasing a moving distribution.

**Ex2 (Temperature sensitivity):** Very low student temperature (τ_s → 0) makes gradients unstable — sharp targets + sharp predictions → numerical issues. Very high teacher temperature (τ_t → 1) makes the teacher signal too soft — the student has nothing meaningful to match. Sweet spot is asymmetry: sharp teacher, softer student.

---

## The Big Picture — SSL Evolution

```
2020  SimCLR     "Contrastive + augmentations. Needs big batches + negatives."
        ↓
2020  BYOL       "No negatives — EMA teacher + predictor MLP. Mysterious why it works."
        ↓
2021  DINO       "No negatives + no predictor. Centering explains the mystery.
                  ViT attention maps → emergent segmentation. Zero labels."
        ↓
2022  MAE        "Mask 75% of patches, reconstruct them. Even simpler pretext task."
        ↓
2023  DINOv2     "DINO at scale (142M params, curated 142M images).
                  Best off-the-shelf vision features ever measured."
```

**Closing question for students:**

> "SSL removes the need for labels during *pretraining*. But linear evaluation still uses labels. Is there a world where we never use labels at all? What would that look like?"
>
> Follow-up: DINOv2 features are so good that k-NN classification (no training at all, just nearest neighbors in feature space) matches fine-tuned supervised models on many benchmarks. Labels may matter less than we thought.
