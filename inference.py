"""
inference.py — Zero-shot classification on a single image.

Usage:
    python inference.py --image path/to/image.jpg --prompts "a dog" "a cat" "a car"
"""

import torch, json, argparse
from PIL import Image
from torchvision import transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.text_encoder import Tokenizer
from models.model        import MultimodalModel


def load_model():
    ckpt  = torch.load("exp/best.pt", map_location="cpu")
    cfg   = ckpt["cfg"]
    tok   = Tokenizer(cfg["vocab_size"], cfg["max_seq_len"])
    tok.load("exp/vocab.json")
    cfg["vocab_size"] = len(tok)
    model = MultimodalModel(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    # Load the mean/std computed from actual training data
    stats = json.load(open("exp/norm_stats.json"))
    return model, tok, cfg, stats["mean"], stats["std"]


def preprocess(image_path, image_size, mean, std):
    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return tfm(Image.open(image_path).convert("RGB")).unsqueeze(0)


def show_result(image_path, prompts, probs):
    img  = Image.open(image_path).convert("RGB")
    best = probs.argmax().item()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.imshow(img)
    ax1.axis("off")
    ax1.set_title(f"Predicted: {prompts[best]}", fontsize=11, fontweight="bold")

    colors = ["crimson" if i == best else "steelblue" for i in range(len(prompts))]
    ax2.barh(prompts, probs.tolist(), color=colors)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Probability")
    ax2.set_title("Zero-Shot Scores")
    ax2.grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig("inference_result.png", dpi=150, bbox_inches="tight")
    print("Result image saved → inference_result.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",   required=True,  help="Path to input image")
    parser.add_argument("--prompts", nargs="+", required=True, help="Text prompts")
    args = parser.parse_args()

    model, tok, cfg, mean, std = load_model()
    img = preprocess(args.image, cfg["image_size"], mean, std)

    with torch.no_grad():
        probs = model.zero_shot(img, tok.encode(args.prompts))

    print("\nResults:")
    for prompt, prob in sorted(zip(args.prompts, probs.tolist()), key=lambda x: -x[1]):
        print(f"  {prob:.4f}  {prompt}")
    print(f"\nPredicted: {args.prompts[probs.argmax()]}")

    show_result(args.image, args.prompts, probs)


if __name__ == "__main__":
    main()
