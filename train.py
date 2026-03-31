"""
train.py — Train the multimodal ViT with InfoNCE contrastive loss on COCO.

Usage:
    python train.py
"""

import json, yaml, torch, logging
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.text_encoder import Tokenizer
from models.model        import MultimodalModel


# ── Compute mean and std from training images ─────────────────────────────────

def compute_mean_std(image_paths, image_size):
    """Compute per-channel mean and std over all training images."""
    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    mean = torch.zeros(3)
    std  = torch.zeros(3)
    n    = 0
    print("Computing dataset mean and std ...")
    for path in image_paths:
        try:
            img = tfm(Image.open(path).convert("RGB"))  # (3, H, W)
        except Exception:
            continue
        mean += img.mean(dim=[1, 2])
        std  += img.std(dim=[1, 2])
        n    += 1
    mean /= n
    std  /= n
    print(f"  mean={mean.tolist()}  std={std.tolist()}")
    return mean.tolist(), std.tolist()


# ── Dataset ───────────────────────────────────────────────────────────────────

class COCODataset(Dataset):
    def __init__(self, split, cfg, tokenizer, mean, std):
        self.root      = Path(cfg["data_dir"])
        self.tokenizer = tokenizer
        with open(self.root / split / "annotations.json") as f:
            self.samples = json.load(f)

        if split == "train":
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(cfg["image_size"], scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((cfg["image_size"], cfg["image_size"])),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s   = self.samples[idx]
        img = Image.open(self.root / s["file_path"]).convert("RGB")
        return self.transform(img), s["caption"]

    @staticmethod
    def collate(batch, tokenizer):
        imgs, captions = zip(*batch)
        return torch.stack(imgs), tokenizer.encode(list(captions))


# ── Logger ────────────────────────────────────────────────────────────────────

def make_logger(name, log_file):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh  = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


# ── Plot loss curves ──────────────────────────────────────────────────────────

def plot_losses(train_losses, val_losses, save_path):
    epochs = range(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, label="Train loss")
    ax.plot(epochs, val_losses,   label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("InfoNCE Loss")
    ax.set_title("Train vs Val Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Loss curve saved → {save_path}")


# ── Training ──────────────────────────────────────────────────────────────────

def train():
    cfg = yaml.safe_load(open("config.yaml"))
    exp = Path("exp")
    exp.mkdir(exist_ok=True)

    train_log = make_logger("train", exp / "train.log")
    val_log   = make_logger("val",   exp / "val.log")

    # Tokenizer
    ann = json.load(open(Path(cfg["data_dir"]) / "train" / "annotations.json"))
    tok = Tokenizer(cfg["vocab_size"], cfg["max_seq_len"])
    tok.build([s["caption"] for s in ann])
    tok.save(exp / "vocab.json")
    cfg["vocab_size"] = len(tok)

    # Compute mean/std from actual training images
    train_paths = [Path(cfg["data_dir"]) / s["file_path"] for s in ann]
    mean, std   = compute_mean_std(train_paths, cfg["image_size"])
    # Save so inference.py can reuse them
    json.dump({"mean": mean, "std": std}, open(exp / "norm_stats.json", "w"))

    # Dataloaders
    def make_loader(split, shuffle):
        ds = COCODataset(split, cfg, tok, mean, std)
        return DataLoader(
            ds, batch_size=cfg["batch_size"], shuffle=shuffle, drop_last=shuffle,
            collate_fn=lambda b: COCODataset.collate(b, tok),
        )

    train_dl = make_loader("train", shuffle=True)
    val_dl   = make_loader("val",   shuffle=False)

    # Model + optimiser
    model = MultimodalModel(cfg)
    opt   = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        betas=(cfg["momentum"], 0.999),
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])

    best_val    = float("inf")
    train_losses = []
    val_losses   = []

    for epoch in range(1, cfg["epochs"] + 1):

        # ── Train ──
        model.train()
        train_loss = 0.0
        for imgs, tok_ids in train_dl:
            opt.zero_grad()
            loss, _, _ = model(imgs, tok_ids)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            train_loss += loss.item()
        sched.step()
        train_loss /= len(train_dl)

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        correct  = 0
        total    = 0
        with torch.no_grad():
            for imgs, tok_ids in val_dl:
                loss, img_emb, txt_emb = model(imgs, tok_ids)
                val_loss += loss.item()
                sims    = img_emb @ txt_emb.T
                preds   = sims.argmax(dim=-1)
                correct += (preds == torch.arange(len(imgs))).sum().item()
                total   += len(imgs)
        val_loss      /= len(val_dl)
        retrieval_acc  = correct / max(total, 1)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # ── Log ──
        print(f"Epoch {epoch:3d}/{cfg['epochs']}  train={train_loss:.4f}  val={val_loss:.4f}  acc={retrieval_acc:.4f}")
        train_log.info(f"epoch={epoch:3d}  loss={train_loss:.4f}  temp={model.temperature.item():.4f}")
        val_log.info(f"epoch={epoch:3d}  loss={val_loss:.4f}  retrieval_acc={retrieval_acc:.4f}")

        # ── Checkpoints ──
        state = {"model": model.state_dict(), "cfg": cfg, "epoch": epoch}
        torch.save(state, exp / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(state, exp / "best.pt")

    # ── Save loss plot ──
    plot_losses(train_losses, val_losses, exp / "loss_curve.png")

    print(f"Done. Best val loss: {best_val:.4f}")
    print(f"Logs       → exp/train.log, exp/val.log")
    print(f"Checkpoint → exp/best.pt, exp/last.pt")


if __name__ == "__main__":
    train()
