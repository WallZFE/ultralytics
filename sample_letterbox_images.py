#!/usr/bin/env python3
"""从 YOLO train 数据集中按类别覆盖采样图片，并进行 letterbox 处理。

默认从 ``data/model_use/images/train`` 选择 256 张图片，利用同名的
``data/model_use/labels/train`` 标注尽量均衡覆盖所有类别，将图片等比例缩放并
填充为 640x384，保存到 ``data/model_use/images/train_letterbox_640x384``。
同时写出 manifest.json，记录每张图的类别、缩放比例和填充量。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class Sample:
    image_path: Path
    class_ids: frozenset[int]


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=project_root / "data/model_use/images/train",
        help="输入图片目录",
    )
    parser.add_argument(
        "--label-dir",
        type=Path,
        default=project_root / "data/model_use/labels/train",
        help="YOLO 标签目录，用于按类别采样",
    )
    parser.add_argument(
        "--dataset-yaml",
        type=Path,
        default=project_root / "data/model_use/dataset.yaml",
        help="数据集 YAML，仅用于读取类别名称（读取失败时仍可处理）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "data/model_use/images/train_letterbox_640x384",
        help="处理后图片的保存目录",
    )
    parser.add_argument("--count", type=int, default=256, help="抽取图片数量")
    parser.add_argument("--width", type=int, default=960, help="输出宽度")
    parser.add_argument("--height", type=int, default=544, help="输出高度")
    parser.add_argument(
        "--fill", type=int, nargs=3, default=(114, 114, 114), metavar=("B", "G", "R"),
        help="填充颜色（B G R）",
    )
    parser.add_argument("--seed", type=int, default=42, help="确定性采样的随机种子")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖输出目录中的同名图片和 manifest.json",
    )
    return parser.parse_args()


def read_class_ids(label_path: Path) -> frozenset[int]:
    class_ids: set[int] = set()
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"标签读取失败：{label_path}: {exc}") from exc

    for line_number, line in enumerate(lines, start=1):
        fields = line.split()
        if not fields:
            continue
        try:
            class_id = int(fields[0])
        except ValueError as exc:
            raise ValueError(
                f"标签类别 ID 不是整数：{label_path}:{line_number}"
            ) from exc
        if class_id < 0:
            raise ValueError(f"标签类别 ID 小于 0：{label_path}:{line_number}")
        class_ids.add(class_id)
    return frozenset(class_ids)


def load_samples(image_dir: Path, label_dir: Path) -> list[Sample]:
    if not image_dir.is_dir():
        raise ValueError(f"图片目录不存在：{image_dir}")
    if not label_dir.is_dir():
        raise ValueError(f"标签目录不存在：{label_dir}")

    image_paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise ValueError(f"图片目录为空：{image_dir}")

    samples: list[Sample] = []
    missing_labels: list[Path] = []
    for image_path in image_paths:
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            missing_labels.append(label_path)
            continue
        samples.append(Sample(image_path, read_class_ids(label_path)))

    if missing_labels:
        preview = ", ".join(str(path) for path in missing_labels[:3])
        raise ValueError(f"有 {len(missing_labels)} 张图片缺少同名标签，例如：{preview}")
    return samples


def balanced_sample(samples: list[Sample], count: int, seed: int) -> list[Sample]:
    """类别覆盖优先采样；类别不足均衡时，仍保证结果不重复。"""
    if count <= 0:
        raise ValueError("--count 必须大于 0")
    if count > len(samples):
        raise ValueError(f"要求 {count} 张，但输入目录只有 {len(samples)} 张")

    by_class: dict[int, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        for class_id in sample.class_ids:
            by_class[class_id].append(index)

    rng = random.Random(seed)
    tie_breaker = list(range(len(samples)))
    rng.shuffle(tie_breaker)
    rank = {sample_index: order for order, sample_index in enumerate(tie_breaker)}
    selected: set[int] = set()
    selected_class_counts: Counter[int] = Counter()

    # 每轮优先照顾当前入选图片数最少的类别。候选图片若还能兼顾其他弱覆盖
    # 类别则优先；同分时使用 seed 控制的固定随机顺序。
    while len(selected) < count and by_class:
        available_classes = [
            class_id
            for class_id, indices in by_class.items()
            if any(index not in selected for index in indices)
        ]
        if not available_classes:
            break
        target_class = min(
            available_classes,
            key=lambda class_id: (
                selected_class_counts[class_id],
                len(by_class[class_id]),
                class_id,
            ),
        )
        candidates = [index for index in by_class[target_class] if index not in selected]

        def candidate_key(index: int) -> tuple[float, int]:
            bonus = sum(
                1.0 / (selected_class_counts[class_id] + 1)
                for class_id in samples[index].class_ids
            )
            return (-bonus, rank[index])

        chosen = min(candidates, key=candidate_key)
        selected.add(chosen)
        selected_class_counts.update(samples[chosen].class_ids)

    # 空标签图片不参与上面的类别均衡；如有余量，从所有未选图片中补足。
    if len(selected) < count:
        remaining = [index for index in tie_breaker if index not in selected]
        selected.update(remaining[: count - len(selected)])

    # 保留确定性的随机输出顺序，避免文件名排序造成相邻图片高度相似。
    selected_indices = list(selected)
    rng.shuffle(selected_indices)
    return [samples[index] for index in selected_indices]


def read_image(path: Path) -> np.ndarray:
    # imdecode 对包含非 ASCII 字符的路径也可靠。
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
    except OSError as exc:
        raise ValueError(f"图片读取失败：{path}: {exc}") from exc
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法解码图片：{path}")
    return image


def letterbox(
    image: np.ndarray, width: int, height: int, fill: tuple[int, int, int]
) -> tuple[np.ndarray, float, tuple[int, int, int, int]]:
    source_height, source_width = image.shape[:2]
    scale = min(width / source_width, height / source_height)
    resized_width = min(width, max(1, round(source_width * scale)))
    resized_height = min(height, max(1, round(source_height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(
        image, (resized_width, resized_height), interpolation=interpolation
    )

    pad_x = width - resized_width
    pad_y = height - resized_height
    left = pad_x // 2
    right = pad_x - left
    top = pad_y // 2
    bottom = pad_y - top
    output = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=fill
    )
    return output, scale, (left, top, right, bottom)


def write_jpeg(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise ValueError(f"图片编码失败：{path}")
    try:
        encoded.tofile(path)
    except OSError as exc:
        raise ValueError(f"图片保存失败：{path}: {exc}") from exc


def read_class_names(dataset_yaml: Path) -> dict[int, str]:
    """轻量读取 dataset.yaml 中的 ``id: name``，避免引入 PyYAML。"""
    names: dict[int, str] = {}
    try:
        lines = dataset_yaml.read_text(encoding="utf-8").splitlines()
    except OSError:
        return names
    in_names = False
    for line in lines:
        stripped = line.strip()
        if stripped == "names:":
            in_names = True
            continue
        if not in_names:
            continue
        if line and not line[0].isspace():
            break
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        try:
            names[int(key)] = value.strip().strip("'\"")
        except ValueError:
            continue
    return names


def main() -> int:
    args = parse_args()
    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width 和 --height 必须大于 0")
    if any(channel < 0 or channel > 255 for channel in args.fill):
        raise ValueError("--fill 的三个通道值必须在 0 到 255 之间")

    samples = load_samples(args.image_dir, args.label_dir)
    selected = balanced_sample(samples, args.count, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    output_paths = [args.output_dir / f"{sample.image_path.stem}.jpg" for sample in selected]
    existing = [path for path in [manifest_path, *output_paths] if path.exists()]
    if existing and not args.overwrite:
        raise ValueError(
            f"输出目录中已有 {len(existing)} 个同名结果；如需覆盖请添加 --overwrite"
        )

    records: list[dict[str, object]] = []
    class_image_counts: Counter[int] = Counter()
    for number, (sample, output_path) in enumerate(
        zip(selected, output_paths), start=1
    ):
        image = read_image(sample.image_path)
        source_height, source_width = image.shape[:2]
        output, scale, padding = letterbox(
            image, args.width, args.height, tuple(args.fill)
        )
        write_jpeg(output_path, output)
        class_image_counts.update(sample.class_ids)
        records.append(
            {
                "source": str(sample.image_path.resolve()),
                "output": str(output_path.resolve()),
                "class_ids": sorted(sample.class_ids),
                "source_size": [source_width, source_height],
                "output_size": [args.width, args.height],
                "scale": scale,
                "padding_ltrb": list(padding),
            }
        )
        print(f"\r处理进度：{number}/{len(selected)}", end="", file=sys.stderr)
    print(file=sys.stderr)

    class_names = read_class_names(args.dataset_yaml)
    summary = {
        str(class_id): {
            "name": class_names.get(class_id, str(class_id)),
            "image_count": class_image_counts[class_id],
        }
        for class_id in sorted({class_id for sample in samples for class_id in sample.class_ids})
    }
    manifest = {
        "sample_count": len(selected),
        "seed": args.seed,
        "fill_bgr": list(args.fill),
        "class_summary": summary,
        "images": records,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"已保存 {len(selected)} 张图片到：{args.output_dir.resolve()}")
    print(f"输出尺寸：{args.width}x{args.height}")
    print("各类别入选图片数：")
    for class_id, info in summary.items():
        print(f"  {class_id:>2} {info['name']}: {info['image_count']}")
    print(f"清单：{manifest_path.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(1)
