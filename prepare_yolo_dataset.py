#!/usr/bin/env python3
"""将 LabelMe 原始数据和 benchmark 整理成 YOLO 检测数据集。

默认输入：
  - data/origin_data：递归查找 LabelMe JSON，作为 train
  - data/benchmark：同目录存放图片和同名 LabelMe JSON，作为 val

默认输出：
  data/model_use/
  ├── images/train/*.jpg
  ├── images/val/*.jpg
  ├── labels/train/*.txt
  ├── labels/val/*.txt
  ├── dataset.yaml
  └── label_counts.json

train 图片和标签会直接放在对应 split 目录下；检测到重名时会报错，
避免不同文件互相覆盖。
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


# 输入标注使用的完整 17 类定义。
SOURCE_CLASS_NAMES = [
    "person",
    "car",
    "light-trunk",
    "heavy-trunk",
    "bus",
    "motorcycle",
    "water-barriers",
    "traffic-cones",
    "warning-posts",
    "crash-barriers",
    "road-construction-signs",
    "minimum-speed-limit-sign",
    "speed-limit-sign",
    "vehicle-type",
    "weight-limit-sign",
    "triangular-warning-signs",
    "width-limit-sign",
]
# 原始标注中的类别别名，统一映射到上面的正式类别。
LABEL_ALIASES = {
    "uncertain-trunk": "heavy-trunk",
}
# 暂时不参与训练/验证的类别；输入时跳过，输出 names 中也不包含。
IGNORED_LABELS = {
    "width-limit-sign",
}
CLASS_NAMES = [name for name in SOURCE_CLASS_NAMES if name not in IGNORED_LABELS]
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class DatasetError(ValueError):
    """输入数据不符合预期。"""


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--origin-dir",
        type=Path,
        default=project_root / "data" / "origin_data",
        help="LabelMe 原始数据目录（递归扫描，默认：data/origin_data）",
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=project_root / "data" / "benchmark",
        help="验证集目录（默认：data/benchmark）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "data" / "model_use",
        help="YOLO 数据集输出目录（默认：data/model_use）",
    )
    parser.add_argument(
        "--limit-train",
        type=int,
        default=0,
        metavar="N",
        help="只处理前 N 个 train 标注，0 表示全部（用于抽样验证）",
    )
    parser.add_argument(
        "--limit-val",
        type=int,
        default=0,
        metavar="N",
        help="只处理前 N 张 val 图片，0 表示全部（用于抽样验证）",
    )
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="跳过有问题的样本并继续；默认遇到错误立即停止",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="运行前删除输出目录中的 images、labels、dataset.yaml 和统计文件",
    )
    return parser.parse_args()


def finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DatasetError(f"{field} 不是数字：{value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise DatasetError(f"{field} 不是有限数：{value!r}")
    return number


def find_image(directory: Path, stem: str) -> Path | None:
    matches = sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.stem == stem
        and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if len(matches) > 1:
        raise DatasetError(f"找到多张同名图片：{matches}")
    return matches[0] if matches else None


def labelme_to_yolo(json_path: Path) -> tuple[list[str], Counter[int]]:
    try:
        annotation = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetError(f"JSON 读取失败：{exc}") from exc

    width = finite_number(annotation.get("imageWidth"), "imageWidth")
    height = finite_number(annotation.get("imageHeight"), "imageHeight")
    if width <= 0 or height <= 0:
        raise DatasetError("imageWidth 和 imageHeight 必须大于 0")

    shapes = annotation.get("shapes", [])
    if not isinstance(shapes, list):
        raise DatasetError("shapes 必须是列表")

    lines: list[str] = []
    counts: Counter[int] = Counter()
    for index, shape in enumerate(shapes):
        if not isinstance(shape, dict):
            raise DatasetError(f"shapes[{index}] 不是对象")
        original_label = shape.get("label")
        label = LABEL_ALIASES.get(original_label, original_label)
        if label in IGNORED_LABELS:
            continue
        if label not in SOURCE_CLASS_NAMES:
            raise DatasetError(f"shapes[{index}] 含未知类别：{original_label!r}")
        if shape.get("shape_type") != "rectangle":
            raise DatasetError(
                f"shapes[{index}] 的 shape_type 不是 rectangle："
                f"{shape.get('shape_type')!r}"
            )
        points = shape.get("points")
        if (
            not isinstance(points, list)
            or len(points) != 2
            or any(not isinstance(point, list) or len(point) < 2 for point in points)
        ):
            raise DatasetError(f"shapes[{index}] 的 rectangle points 格式错误")

        x1 = finite_number(points[0][0], f"shapes[{index}].x1")
        y1 = finite_number(points[0][1], f"shapes[{index}].y1")
        x2 = finite_number(points[1][0], f"shapes[{index}].x2")
        y2 = finite_number(points[1][1], f"shapes[{index}].y2")

        # 两点顺序不作假设；越界框裁剪到图片范围。
        left = min(max(min(x1, x2), 0.0), width)
        right = min(max(max(x1, x2), 0.0), width)
        top = min(max(min(y1, y2), 0.0), height)
        bottom = min(max(max(y1, y2), 0.0), height)
        box_width = right - left
        box_height = bottom - top
        if box_width <= 0 or box_height <= 0:
            raise DatasetError(f"shapes[{index}] 裁剪后的框面积为 0")

        class_id = CLASS_TO_ID[label]
        center_x = (left + right) / (2.0 * width)
        center_y = (top + bottom) / (2.0 * height)
        lines.append(
            f"{class_id} {center_x:.8f} {center_y:.8f} "
            f"{box_width / width:.8f} {box_height / height:.8f}"
        )
        counts[class_id] += 1
    return lines, counts


def limited(paths: Iterable[Path], limit: int) -> list[Path]:
    selected = sorted(paths)
    return selected[:limit] if limit > 0 else selected


def show_progress(split: str, current: int, total: int) -> None:
    """在 stderr 上显示一个不依赖第三方库的进度条。"""
    width = 30
    ratio = current / total if total else 1.0
    completed = min(width, int(ratio * width))
    bar = "█" * completed + "-" * (width - completed)
    ending = "\n" if current >= total else "\r"
    print(
        f"{split:>5} [{bar}] {current}/{total} {ratio:6.1%}",
        end=ending,
        file=sys.stderr,
        flush=True,
    )


def write_label(path: Path, lines: list[str] | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(lines, list):
        text = "\n".join(lines) + ("\n" if lines else "")
    else:
        text = lines
    path.write_text(text, encoding="utf-8")


def copy_image(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def prepare_train(
    origin_dir: Path, output_dir: Path, limit: int, skip_invalid: bool
) -> tuple[int, Counter[int], int]:
    json_paths = limited(origin_dir.rglob("*.json"), limit)
    if not json_paths:
        raise DatasetError(f"train 目录中没有 JSON：{origin_dir}")

    image_count = 0
    skipped = 0
    counts: Counter[int] = Counter()
    used_stems: dict[str, Path] = {}
    show_progress("train", 0, len(json_paths))
    for progress, json_path in enumerate(json_paths, start=1):
        try:
            image_path = find_image(json_path.parent, json_path.stem)
            if image_path is None:
                raise DatasetError("找不到同目录、同名的图片")
            normalized_stem = image_path.stem.casefold()
            if normalized_stem in used_stems:
                raise DatasetError(
                    f"扁平输出发生重名：{image_path.name!r}；"
                    f"另一个文件为 {used_stems[normalized_stem]}"
                )
            lines, sample_counts = labelme_to_yolo(json_path)
            destination_image = (
                output_dir / "images" / "train" / image_path.name
            ).with_suffix(image_path.suffix.lower())
            destination_label = output_dir / "labels" / "train" / f"{image_path.stem}.txt"
            copy_image(image_path, destination_image)
            write_label(destination_label, lines)
        except (DatasetError, OSError) as exc:
            if not skip_invalid:
                print(file=sys.stderr)
                raise DatasetError(f"train {json_path}: {exc}") from exc
            print(f"警告：跳过 train {json_path}：{exc}", file=sys.stderr)
            skipped += 1
        else:
            used_stems[normalized_stem] = image_path
            counts.update(sample_counts)
            image_count += 1
        show_progress("train", progress, len(json_paths))
    return image_count, counts, skipped


def prepare_val(
    benchmark_dir: Path, output_dir: Path, limit: int, skip_invalid: bool
) -> tuple[int, Counter[int], int]:
    image_paths = limited(
        (
            path
            for path in benchmark_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ),
        limit,
    )
    if not image_paths:
        raise DatasetError(f"val 目录中没有图片：{benchmark_dir}")

    image_count = 0
    skipped = 0
    counts: Counter[int] = Counter()
    show_progress("val", 0, len(image_paths))
    for progress, image_path in enumerate(image_paths, start=1):
        relative = image_path.relative_to(benchmark_dir)
        json_path = image_path.with_suffix(".json")
        try:
            if not json_path.is_file():
                raise DatasetError("找不到同名的 LabelMe JSON 标签")
            lines, sample_counts = labelme_to_yolo(json_path)

            destination_image = output_dir / "images" / "val" / relative
            destination_label = (output_dir / "labels" / "val" / relative).with_suffix(".txt")
            copy_image(image_path, destination_image)
            write_label(destination_label, lines)
        except (DatasetError, OSError) as exc:
            if not skip_invalid:
                print(file=sys.stderr)
                raise DatasetError(f"val {image_path}: {exc}") from exc
            print(f"警告：跳过 val {image_path}：{exc}", file=sys.stderr)
            skipped += 1
        else:
            counts.update(sample_counts)
            image_count += 1
        show_progress("val", progress, len(image_paths))
    return image_count, counts, skipped


def clean_outputs(output_dir: Path) -> None:
    for directory_name in ("images", "labels"):
        target = output_dir / directory_name
        if target.exists():
            shutil.rmtree(target)
    for file_name in ("dataset.yaml", "label_counts.json"):
        target = output_dir / file_name
        if target.exists():
            target.unlink()


def write_metadata(
    output_dir: Path,
    train_images: int,
    val_images: int,
    train_counts: Counter[int],
    val_counts: Counter[int],
) -> None:
    yaml_lines = [
        f"path: {output_dir.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "",
        "names:",
    ]
    yaml_lines.extend(f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES))
    (output_dir / "dataset.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    statistics = {
        "train": {
            "images": train_images,
            "boxes": sum(train_counts.values()),
            "labels": {
                name: train_counts[index] for index, name in enumerate(CLASS_NAMES)
            },
        },
        "val": {
            "images": val_images,
            "boxes": sum(val_counts.values()),
            "labels": {
                name: val_counts[index] for index, name in enumerate(CLASS_NAMES)
            },
        },
    }
    (output_dir / "label_counts.json").write_text(
        json.dumps(statistics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def print_statistics(split: str, image_count: int, counts: Counter[int]) -> None:
    print(f"{split}: {image_count} 张图片，{sum(counts.values())} 个框")
    for class_id, name in enumerate(CLASS_NAMES):
        print(f"  {class_id:2d} {name}: {counts[class_id]}")


def main() -> int:
    args = parse_args()
    if args.limit_train < 0 or args.limit_val < 0:
        print("错误：limit 不能为负数", file=sys.stderr)
        return 2

    origin_dir = args.origin_dir.expanduser().resolve()
    benchmark_dir = args.benchmark_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not origin_dir.is_dir():
        print(f"错误：origin 目录不存在：{origin_dir}", file=sys.stderr)
        return 2
    if not benchmark_dir.is_dir():
        print(f"错误：benchmark 目录不存在：{benchmark_dir}", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.clean:
        clean_outputs(output_dir)

    try:
        train_images, train_counts, skipped_train = prepare_train(
            origin_dir, output_dir, args.limit_train, args.skip_invalid
        )
        val_images, val_counts, skipped_val = prepare_val(
            benchmark_dir, output_dir, args.limit_val, args.skip_invalid
        )
        write_metadata(output_dir, train_images, val_images, train_counts, val_counts)
    except DatasetError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1

    print(f"输出目录：{output_dir}")
    print_statistics("train", train_images, train_counts)
    print_statistics("val", val_images, val_counts)
    if skipped_train or skipped_val:
        print(f"跳过样本：train={skipped_train}, val={skipped_val}")
    print(f"数据集配置：{output_dir / 'dataset.yaml'}")
    print(f"类别统计：{output_dir / 'label_counts.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
