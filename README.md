# Multimodal ViT — COCO Image-Text Contrastive Learning

A minimal, readable implementation of CLIP-style multimodal learning using COCO.

## What it does

Trains a **VisionTransformer** and a **TextTransformer** jointly with an **InfoNCE contrastive loss** so that matching image–caption pairs are close in embedding space and non-matching pairs are far apart.

After training the model supports **zero-shot classification**: encode an image and a set of text prompts, pick the prompt with the highest cosine similarity.

## Project structure

```
cv_system/
├── config.yaml              ← all hyperparameters
├── dataset/
│   └── get_dataset.py       ← download COCO subset (no account needed)
├── data/                    ← created by get_dataset.py
│   ├── Images/              ← downloaded JPEG images
│   ├── categories.json      ← 80 COCO class names
│   ├── train/annotations.json
│   └── val/annotations.json
├── models/
│   ├── vit_encoder.py       ← VisionTransformer
│   ├── text_encoder.py      ← TextTransformer + Tokenizer
│   └── model.py             ← MultimodalModel (InfoNCE loss + zero_shot)
├── exp/                     ← created by train.py
│   ├── train.log            ← one line per epoch
│   ├── val.log              ← one line per epoch
│   ├── best.pt              ← best checkpoint (overwritten)
│   ├── last.pt              ← latest checkpoint (overwritten)
│   └── vocab.json
├── train.py
├── demo.ipynb
└── requirements.txt
```

## Quick start

```bash
pip install -r requirements.txt

# 1. Download 1000 COCO images (~240 MB annotations one-time + ~100 MB images)
python dataset/get_dataset.py

# 2. Train
python train.py

# 3. Explore results
jupyter notebook demo.ipynb
```

## Inference

Run zero-shot classification on any image against a list of text prompts:

```bash
python inference.py --image path/to/image.jpg --prompts "a photo of a dog" "a photo of a cat" "a photo of a car"
```

Output:
```
Results:
  0.7832  a photo of a dog
  0.1421  a photo of a cat
  0.0747  a photo of a car

Predicted: a photo of a dog
```

## Dataset

COCO val2017 subset downloaded directly from cocodataset.org — no account needed.

| File | Size | Downloaded once |
|---|---|---|
| `annotations_trainval2017.zip` | ~240 MB | ✓ zip deleted after extraction |
| Individual images | ~100 KB each | on-demand, skips existing |

Re-running `get_dataset.py` skips images already on disk.

## Model

| Component | Class | Details |
|---|---|---|
| Image encoder | `VisionTransformer` | Patch embed → CLS token → Transformer blocks → L2-proj |
| Text encoder | `TextTransformer` | Token embed → Transformer blocks → mean-pool → L2-proj |
| Loss | InfoNCE | Symmetric cross-entropy on N×N cosine-similarity matrix |

Both encoders project to the same `proj_dim`-dimensional space and L2-normalise their outputs, so dot product = cosine similarity.

## Loss function

```
logits  = (img_emb @ txt_emb.T) / temperature     # (B, B)
targets = [0, 1, 2, ..., B-1]                      # diagonal = positives

loss = (cross_entropy(logits, targets) + cross_entropy(logits.T, targets)) / 2
```

Temperature is a **learnable parameter** initialised from config, clamped to `[0.01, 1.0]`.

## Experiment output

Every `train.py` run appends to `exp/train.log` and `exp/val.log` (no new folders created).  
Checkpoints `exp/best.pt` and `exp/last.pt` are overwritten each run.

Log format:
```
2024-03-01 12:00:00  epoch=  1  loss=4.3821  temp=0.0700
```

## Configuration

Edit `config.yaml` to change any hyperparameter:

| Key | Default | Description |
|---|---|---|
| `image_size` | 64 | Input resolution (increase for quality, slower) |
| `embed_dim` | 128 | Transformer hidden dimension |
| `proj_dim` | 128 | Shared embedding space dimension |
| `temperature` | 0.07 | InfoNCE temperature (learnable) |
| `epochs` | 30 | Training epochs |
| `num_images` | 1000 | Images to download |
