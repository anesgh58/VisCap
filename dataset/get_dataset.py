"""
dataset/get_dataset.py
----------------------
Download a COCO subset directly from cocodataset.org.
No account or API key needed.

What it downloads:
  - annotations_trainval2017.zip (~240 MB, one-time)
    → extracts only 2 JSON files → zip is deleted immediately
  - N individual val2017 images on demand (~100 KB each, skips existing)

Each sample stores: image path, caption, bounding boxes, class labels.

Usage:
    python dataset/get_dataset.py           # 1000 images (default)
    python dataset/get_dataset.py --n 500
"""

import sys, json, random, zipfile, urllib.request, argparse
from pathlib import Path

ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMAGES_BASE     = "http://images.cocodataset.org/val2017"
DATA_DIR        = Path("./data")
ANN_DIR         = DATA_DIR / "annotations"
IMAGES_DIR      = DATA_DIR / "Images"


def download(url, dest, label=""):
    print(f"Downloading {label or url} ...")
    def progress(count, block, total):
        sys.stdout.write(f"\r  {count*block/1e6:.1f} / {total/1e6:.1f} MB")
        sys.stdout.flush()
    urllib.request.urlretrieve(url, dest, reporthook=progress)
    print()


def get_annotations():
    """Download annotation zip once, extract the 2 needed JSONs, delete zip."""
    ANN_DIR.mkdir(parents=True, exist_ok=True)
    inst_path = ANN_DIR / "instances_val2017.json"
    caps_path = ANN_DIR / "captions_val2017.json"

    if not inst_path.exists() or not caps_path.exists():
        zip_path = DATA_DIR / "annotations.zip"
        download(ANNOTATIONS_URL, zip_path, "COCO annotations (~240 MB, one-time)")
        print("Extracting ...")
        with zipfile.ZipFile(zip_path) as z:
            z.extract("annotations/instances_val2017.json", DATA_DIR)
            z.extract("annotations/captions_val2017.json",  DATA_DIR)
        zip_path.unlink()
        print("Annotation zip deleted — only the 2 JSONs kept.")

    return json.load(open(inst_path)), json.load(open(caps_path))


def build_samples(instances, captions, n):
    """Select N images that each have a caption and at least one bounding box."""
    cat_map   = {c["id"]: i for i, c in enumerate(instances["categories"])}
    cat_names = [c["name"] for c in instances["categories"]]
    img_info  = {img["id"]: img for img in instances["images"]}

    # image_id → boxes (sorted largest-first)
    img_boxes = {}
    for ann in instances["annotations"]:
        iid = ann["image_id"]
        img_boxes.setdefault(iid, []).append({
            "bbox":  ann["bbox"],               # [x, y, w, h] absolute pixels
            "label": cat_map[ann["category_id"]],
        })

    # image_id → first caption
    img_caps = {}
    for ann in captions["annotations"]:
        iid = ann["image_id"]
        if iid not in img_caps:
            img_caps[iid] = ann["caption"]

    valid = [iid for iid in img_info if iid in img_caps and iid in img_boxes]
    random.seed(42)
    random.shuffle(valid)

    samples = []
    for iid in valid[:n]:
        info = img_info[iid]
        W, H = info["width"], info["height"]

        # Normalize boxes to [cx, cy, w, h] in [0, 1], sorted by area desc
        norm_boxes, labels = [], []
        for item in sorted(img_boxes[iid], key=lambda x: x["bbox"][2] * x["bbox"][3], reverse=True):
            x, y, w, h = item["bbox"]
            norm_boxes.append([(x + w/2)/W, (y + h/2)/H, w/W, h/H])
            labels.append(item["label"])

        samples.append({
            "id":        iid,
            "file_name": info["file_name"],
            "file_path": f"Images/{info['file_name']}",
            "caption":   img_caps[iid],
            "boxes":     norm_boxes,
            "labels":    labels,
        })

    return samples, cat_names


def download_images(samples):
    """Download each image individually; skip if already on disk."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(samples)} images ...")
    for i, s in enumerate(samples, 1):
        dest = IMAGES_DIR / s["file_name"]
        if not dest.exists():
            try:
                urllib.request.urlretrieve(f"{IMAGES_BASE}/{s['file_name']}", dest)
            except Exception as e:
                print(f"  [warn] {s['file_name']}: {e}")
        if i % 100 == 0:
            print(f"  {i}/{len(samples)}")


def write_splits(samples, cat_names):
    """Write 80/20 train/val splits as annotations.json."""
    random.seed(42)
    random.shuffle(samples)
    cut    = int(len(samples) * 0.8)
    splits = {"train": samples[:cut], "val": samples[cut:]}

    for split, data in splits.items():
        split_dir = DATA_DIR / split
        split_dir.mkdir(exist_ok=True)
        with open(split_dir / "annotations.json", "w") as f:
            json.dump(data, f, indent=2)
        print(f"  {split}: {len(data)} images")

    with open(DATA_DIR / "categories.json", "w") as f:
        json.dump(cat_names, f)
    print("  categories.json saved")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000, help="Number of images (default: 1000)")
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)

    if (DATA_DIR / "train" / "annotations.json").exists():
        print("Dataset already exists. Delete data/ to re-download.")
        return

    instances, captions = get_annotations()
    samples, cat_names  = build_samples(instances, captions, args.n)
    download_images(samples)
    write_splits(samples, cat_names)
    print("Done. Run: python train.py")


if __name__ == "__main__":
    main()
