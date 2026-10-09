#!/usr/bin/env python3
"""Benchmark a trained detection model on the training dataset's validation split."""

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = ROOT / "runs/detect/yolo26s_new_b32/weights/best.pt"
DEFAULT_DATA = ROOT.parent / "data/model_use/dataset.yaml"


def parse_args():
    """Parse detection benchmark arguments."""
    parser = argparse.ArgumentParser(description="Benchmark detection accuracy and speed on the validation set.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="Trained .pt model path.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Training dataset YAML path.")
    parser.add_argument("--imgsz", type=int, default=960, help="Validation image size.")
    parser.add_argument("--batch", type=int, default=32, help="Validation batch size.")
    parser.add_argument("--device", default="0", help="Device, for example 0, 0,1, or cpu.")
    parser.add_argument("--workers", type=int, default=8, help="Number of data loader workers.")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/benchmark", help="Output directory.")
    parser.add_argument("--name", default="model_use", help="Benchmark run name.")
    parser.add_argument("--plots", action="store_true", help="Save PR curves and confusion matrices.")
    return parser.parse_args()


def main():
    """Evaluate accuracy and inference speed on the validation split."""
    args = parse_args()
    weights = args.weights.expanduser().resolve()
    data = args.data.expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Model weights not found: {weights}")
    if not data.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data}")

    metrics = YOLO(weights, task="detect").val(
        data=str(data),
        split="val",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(args.project.expanduser().resolve()),
        name=args.name,
        plots=args.plots,
    )

    speed = {key: float(value) for key, value in metrics.speed.items()}
    total_ms = sum(speed.get(key, 0.0) for key in ("preprocess", "inference", "postprocess"))
    summary = {
        "weights": str(weights),
        "data": str(data),
        "split": "val",
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "metrics": {key: float(value) for key, value in metrics.results_dict.items()},
        "speed_ms_per_image": speed,
        "inference_fps": 1000.0 / speed["inference"] if speed.get("inference", 0.0) > 0 else None,
        "end_to_end_fps": 1000.0 / total_ms if total_ms > 0 else None,
    }

    output = Path(metrics.save_dir) / "benchmark.json"
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("\nBenchmark summary")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()
