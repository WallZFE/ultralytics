#!/usr/bin/env python3
"""Create one reproducible 1920x1080 crop per training image with clipped YOLO labels."""

import argparse
import hashlib
import random
import re
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1] / "data/model_use"
CROP_WIDTH, CROP_HEIGHT = 1920, 1080
MIN_IOU = 0.2
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def crop_origin(name: str, width: int, height: int, seed: int) -> tuple[int, int]:
    """Choose one random crop, stable across retries and independent of file order."""
    digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    return rng.randint(0, width - CROP_WIDTH), rng.randint(0, height - CROP_HEIGHT)


def crop_labels(label_path: Path, width: int, height: int, left: int, top: int):
    """Return normalized crop labels and partially visible box decisions."""
    kept, edges = [], []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Expected YOLO class/xywh at {label_path}:{line_number}")
        class_id = int(parts[0])
        x, y, w, h = map(float, parts[1:])
        x1, y1 = (x - w / 2) * width, (y - h / 2) * height
        x2, y2 = (x + w / 2) * width, (y + h / 2) * height
        ix1, iy1 = max(x1, left), max(y1, top)
        ix2, iy2 = min(x2, left + CROP_WIDTH), min(y2, top + CROP_HEIGHT)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        # The intersection is contained in the original box, so IoU = visible / original area.
        iou = (ix2 - ix1) * (iy2 - iy1) / ((x2 - x1) * (y2 - y1))
        keep = iou > MIN_IOU
        if keep:
            kept.append(
                f"{class_id} {((ix1 + ix2) / 2 - left) / CROP_WIDTH:.6f} "
                f"{((iy1 + iy2) / 2 - top) / CROP_HEIGHT:.6f} "
                f"{(ix2 - ix1) / CROP_WIDTH:.6f} {(iy2 - iy1) / CROP_HEIGHT:.6f}"
            )
        if iou < 1.0 - 1e-9:
            edges.append((class_id, (x1, y1, x2, y2), (ix1, iy1, ix2, iy2), iou, keep))
    return kept, edges


def save_preview(image: Image.Image, left: int, top: int, edge: tuple, destination: Path):
    """Show the original box, crop border, and visible intersection around one edge object."""
    class_id, box, intersection, iou, keep = edge
    color = "#17b44a" if keep else "#e53935"
    x1, y1, x2, y2 = box
    margin = max(100, int(max(x2 - x1, y2 - y1) * 2))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    view = (int(cx - margin), int(cy - margin), int(cx + margin), int(cy + margin))
    zoom = image.crop(view).resize((800, 800))
    draw = ImageDraw.Draw(zoom)

    def project(rect):
        return tuple((v - (view[0] if i % 2 == 0 else view[1])) * 800 / (2 * margin) for i, v in enumerate(rect))

    draw.rectangle(project((left, top, left + CROP_WIDTH, top + CROP_HEIGHT)), outline="#ffca28", width=5)
    draw.rectangle(project(box), outline="#2196f3", width=5)
    draw.rectangle(project(intersection), outline=color, width=5)
    canvas = Image.new("RGB", (800, 850), "white")
    canvas.paste(zoom, (0, 50))
    ImageDraw.Draw(canvas).text(
        (12, 15), f"class {class_id} | IoU={iou:.3f} | {'KEEP' if keep else 'DROP'} | blue=original, yellow=crop, color=visible", fill="black"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)


def update_dataset_yaml(dataset_root: Path):
    """Add crop images to training only after all crops have been written."""
    yaml_path = dataset_root / "dataset.yaml"
    source = yaml_path.read_text(encoding="utf-8")
    if re.search(r"(?m)^train: \[images/train, images/train_crops\][ \t]*$", source):
        return
    updated, count = re.subn(
        r"(?m)^train: images/train[ \t]*$",
        "train: [images/train, images/train_crops]",
        source,
    )
    if count != 1:
        raise ValueError(f"Expected exactly one 'train: images/train' in {yaml_path}")
    yaml_path.write_text(updated, encoding="utf-8")


def write_crop(image_path: Path, root: Path, seed: int):
    """Write one cropped image and its clipped labels."""
    label_path = root / "labels/train" / f"{image_path.stem}.txt"
    if not label_path.is_file():
        raise FileNotFoundError(f"Missing label: {label_path}")
    with Image.open(image_path) as image:
        width, height = image.size
        if width < CROP_WIDTH or height < CROP_HEIGHT:
            raise ValueError(f"Image smaller than crop: {image_path} ({width}x{height})")
        left, top = crop_origin(image_path.name, width, height, seed)
        labels, _ = crop_labels(label_path, width, height, left, top)
        out_image = root / "images/train_crops" / image_path.name
        out_label = root / "labels/train_crops" / label_path.name
        out_image.parent.mkdir(parents=True, exist_ok=True)
        out_label.parent.mkdir(parents=True, exist_ok=True)
        crop = image.crop((left, top, left + CROP_WIDTH, top + CROP_HEIGHT))
        save_args = {"quality": 95, "subsampling": 0} if out_image.suffix.lower() in {".jpg", ".jpeg"} else {}
        crop.save(out_image, **save_args)
        out_label.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preview", action="store_true", help="Show one kept and one dropped edge case; write no crops.")
    parser.add_argument("--workers", type=int, default=8, help="Parallel image decoding and saving workers.")
    args = parser.parse_args()
    root = args.data_root.expanduser().resolve()
    image_dir, label_dir = root / "images/train", root / "labels/train"
    images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise FileNotFoundError(f"No training images in {image_dir}")
    if len({p.stem for p in images}) != len(images):
        raise ValueError("Training images must have unique stems for YOLO labels")

    examples = {}
    if args.preview:
        for image_path in images:
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                raise FileNotFoundError(f"Missing label: {label_path}")
            with Image.open(image_path) as image:
                width, height = image.size
                if width < CROP_WIDTH or height < CROP_HEIGHT:
                    raise ValueError(f"Image smaller than crop: {image_path} ({width}x{height})")
                left, top = crop_origin(image_path.name, width, height, args.seed)
                _, edges = crop_labels(label_path, width, height, left, top)
                for edge in edges:
                    key = "keep" if edge[-1] else "drop"
                    if key not in examples:
                        destination = root / "crop_preview" / f"{key}_{image_path.stem}.jpg"
                        save_preview(image, left, top, edge, destination)
                        examples[key] = (destination, edge[3], image_path.name)
                if len(examples) == 2:
                    break
        if len(examples) != 2:
            raise RuntimeError(f"Could not find both kept and dropped partial boxes: {examples}")
        for key, (path, iou, name) in examples.items():
            print(f"{key}: {name}, IoU={iou:.3f}, preview={path}")
    else:
        if args.workers < 1:
            raise ValueError("--workers must be at least 1")
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for count, _ in enumerate(pool.map(partial(write_crop, root=root, seed=args.seed), images), 1):
                if count % 1000 == 0:
                    print(f"Cropped {count}/{len(images)} images", flush=True)
        update_dataset_yaml(root)
        print(f"Wrote {len(images)} crops and updated {root / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
