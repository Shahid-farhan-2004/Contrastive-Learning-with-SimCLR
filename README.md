# SimCLR Self-Supervised Learning on CIFAR-10 using PyTorch

## Overview

This project implements **SimCLR (Simple Framework for Contrastive Learning of Visual Representations)** using **PyTorch**. SimCLR is a **self-supervised learning** algorithm that learns meaningful image representations without using class labels.

The model generates two different augmented views of the same image, encodes them using a shared neural network, projects them into an embedding space, and learns to bring similar views closer while pushing different images farther apart using the **NT-Xent (Normalized Temperature-scaled Cross Entropy) Loss**.

---

## Features

- Self-supervised learning (no labels required)
- SimCLR data augmentation pipeline
- ResNet-18 encoder
- Projection head (MLP)
- NT-Xent Contrastive Loss
- CIFAR-10 dataset
- PyTorch implementation

---

## Technologies Used

- Python
- PyTorch
- Torchvision
- CIFAR-10 Dataset

---

## Requirements

Install the required libraries:

```bash
pip install torch torchvision
```

---

## Dataset

The project uses the **CIFAR-10** dataset.

It is downloaded automatically using:

```python
base_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True
)
```

Unlike supervised learning, **only the images are used**. Class labels are ignored.

---

## Data Augmentation

Each image is transformed twice to create two correlated views.

```python
simclr_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([
        transforms.ColorJitter(0.4,0.4,0.4,0.1)
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5,0.5,0.5),
        (0.5,0.5,0.5)
    )
])
```

### Augmentations Used

- Random Resized Crop
- Random Horizontal Flip
- Color Jitter
- Random Grayscale
- Normalization

These augmentations create two different views of the same image while preserving its semantic meaning.

---

## SimCLR Dataset

A custom dataset returns two augmented versions of the same image.

```python
view1, view2 = dataset[index]
```

Both views originate from the same image but contain different random augmentations.

---

## Model Architecture

```
Input Image
      │
      ▼
Data Augmentation
      │
 ┌────┴────┐
 ▼         ▼
View 1   View 2
 │          │
 ▼          ▼
Shared ResNet18 Encoder
 │          │
 ▼          ▼
512-D Feature Vector
 │          │
 ▼          ▼
Projection Head (MLP)
 │          │
 ▼          ▼
128-D Embedding
      │
      ▼
 NT-Xent Loss
```

The encoder and projector share weights for both image views.

---

## Encoder

The encoder is **ResNet-18** with its classification layer removed.

```python
base = models.resnet18(pretrained=False)

self.encoder = nn.Sequential(
    *list(base.children())[:-1]
)
```

The encoder outputs a **512-dimensional feature vector**.

---

## Projection Head

The projection head maps encoder features into a lower-dimensional embedding space.

```python
Linear(512 → 256)
ReLU
Linear(256 → 128)
```

The final embedding is normalized using:

```python
F.normalize(x, dim=1)
```

This produces unit-length feature vectors for cosine similarity.

---

## NT-Xent Loss

The model uses **Normalized Temperature-scaled Cross Entropy (NT-Xent) Loss**.

### Steps

1. Generate embeddings for two augmented views.
2. Concatenate embeddings.
3. Compute pairwise cosine similarity.
4. Remove self-similarity.
5. Identify positive pairs.
6. Treat all other samples as negatives.
7. Compute contrastive loss.

---

## Loss Workflow

```
View 1 Embeddings
        │
        ▼
      Concatenate
        ▲
        │
View 2 Embeddings
        │
        ▼
Cosine Similarity Matrix
        │
        ▼
Mask Self Similarities
        │
        ▼
Positive Pair Selection
        │
        ▼
NT-Xent Loss
```

---

## Training

Run the script:

```bash
python simclr.py
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 0.01 |
| Batch Size | 128 |
| Epochs | 5 |
| Encoder | ResNet-18 |
| Projection Dimension | 128 |

Example output:

```text
loss is 4.82
loss is 4.11
loss is 3.76
loss is 3.41
loss is 3.18
```

---

## Project Structure

```text
.
├── data/
│   └── CIFAR10/
├── simclr.py
└── README.md
```

---

## Future Improvements

- Train for more epochs (100+)
- Use GPU (CUDA)
- Use pretrained ResNet backbone
- Add learning rate scheduler
- Save trained encoder
- Perform linear evaluation
- Train on larger datasets (STL-10, ImageNet)
- Visualize embeddings using t-SNE

---

## Learning Outcomes

This project demonstrates:

- Self-Supervised Learning
- SimCLR Framework
- Contrastive Learning
- ResNet Feature Extraction
- Projection Head Design
- Data Augmentation
- Cosine Similarity
- NT-Xent Loss
- Representation Learning
- PyTorch Deep Learning

---

## References

- SimCLR: *A Simple Framework for Contrastive Learning of Visual Representations* (Chen et al., 2020)
- PyTorch Documentation
- Torchvision Documentation

---

## Author

**Shahid Farhan KP**

**B.Tech Computer Science and Engineering**

**Cochin University of Science and Technology (CUSAT)**
