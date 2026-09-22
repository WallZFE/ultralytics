#!/usr/bin/env python3
"""Train the official four-scale YOLO26s-P2 detector on the prepared model_use dataset."""

import argparse
from pathlib import Path

from ultralytics import YOLO


ULTRALYTICS_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ULTRALYTICS_ROOT.parent
DEFAULT_DATA = PROJECT_ROOT / "data/model_use/dataset.yaml"
DEFAULT_MODEL = PROJECT_ROOT / "ultralytics/ultralytics/cfg/models/26/yolo26s.yaml"
DEFAULT_WEIGHTS = "yolo26s.pt"


def parse_args():
    """Parse detection training arguments."""
    parser = argparse.ArgumentParser(description="Train the official four-scale YOLO26s-P2 model with Ultralytics.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Dataset YAML path.")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Four-head P2/P3/P4/P5 model YAML (the scale-specific path may be virtual).",
    )
    parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS, help="Initial YOLO26s weights.")
    parser.add_argument("--epochs", type=int, default=250, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=960, help="Training image size.")
    parser.add_argument("--batch", type=int, default=32, help="Total batch size across all GPUs.")
    parser.add_argument("--device", type=str, default="0,1,2,3", help="CUDA devices, for example 0,1,2,3.")
    parser.add_argument("--workers", type=int, default=32, help="Data loader workers per process.")
    parser.add_argument("--project", type=Path, default=ULTRALYTICS_ROOT / "runs/detect", help="Output directory.")
    parser.add_argument("--name", type=str, default="yolo26s", help="Training run name.")
    parser.add_argument("--cache", action="store_true", help="Cache dataset images in RAM.")
    return parser.parse_args()


def main():
    """Run four-scale YOLO26s-P2 detection training."""
    args = parse_args()
    data = args.data.expanduser().resolve()
    if not data.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data}")

    model = YOLO(str(args.model.expanduser().resolve()), task="detect")

    model.train(
        data=str(data),
        pretrained=args.weights or False,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(args.project.expanduser().resolve()),
        name=args.name,
        cache="ram" if args.cache else False,
    )


if __name__ == "__main__":
    main()
