# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Run folder inference with a raw one-to-one YOLO26 ONNX model."""

import argparse
import ast
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "runs/detect/yolo26s-p2-4/weights/best_onetoone.onnx"
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Image file or folder. Folders are searched recursively.")
    parser.add_argument("--model", type=Path, default=MODEL, help="Path to the raw one-to-one ONNX model.")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/onnx_predict", help="Output image folder.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections per image.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="ONNX Runtime device.")
    return parser.parse_args()


def letterbox(image: np.ndarray, shape: tuple[int, int]):
    """Resize and center-pad an image while preserving its aspect ratio."""
    height, width = image.shape[:2]
    new_height, new_width = shape
    gain = min(new_width / width, new_height / height)
    resized_width, resized_height = round(width * gain), round(height * gain)
    pad_width, pad_height = new_width - resized_width, new_height - resized_height
    left, top = round(pad_width / 2 - 0.1), round(pad_height / 2 - 0.1)
    right, bottom = round(pad_width / 2 + 0.1), round(pad_height / 2 + 0.1)
    if (width, height) != (resized_width, resized_height):
        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return image, gain, (left, top)


def decode(outputs: list[np.ndarray], strides: list[int], conf: float, max_det: int):
    """Decode four raw regression maps and four raw class maps from the one-to-one head."""
    levels = len(outputs) // 2
    boxes, scores = [], []
    for box_map, score_map, stride in zip(outputs[:levels], outputs[levels:], strides):
        _, _, height, width = box_map.shape
        distances = box_map[0].reshape(4, -1).T
        logits = score_map[0].reshape(score_map.shape[1], -1).T
        grid_y, grid_x = np.mgrid[:height, :width]
        anchors = np.stack((grid_x.ravel() + 0.5, grid_y.ravel() + 0.5), axis=1)
        boxes.append(
            np.concatenate((anchors - distances[:, :2], anchors + distances[:, 2:]), axis=1) * stride
        )
        scores.append(1.0 / (1.0 + np.exp(-np.clip(logits, -80, 80))))

    boxes, scores = np.concatenate(boxes), np.concatenate(scores)
    # The one-to-one head is NMS-free. Match Ultralytics' end-to-end TopK over box/class pairs.
    flat_scores = scores.ravel()
    count = min(max_det, flat_scores.size)
    indices = np.argpartition(flat_scores, -count)[-count:]
    indices = indices[np.argsort(flat_scores[indices])[::-1]]
    keep = flat_scores[indices] >= conf
    indices = indices[keep]
    box_indices, class_ids = np.divmod(indices, scores.shape[1])
    return boxes[box_indices], flat_scores[indices], class_ids


def draw(image: np.ndarray, boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray, names: dict):
    """Draw detection boxes and labels in-place."""
    line_width = max(round(min(image.shape[:2]) * 0.002), 2)
    font_scale = max(line_width * 0.25, 0.5)
    for box, score, class_id in zip(boxes, scores, class_ids):
        class_id = int(class_id)
        color = tuple(int(x) for x in np.random.default_rng(class_id).integers(64, 256, 3))
        x1, y1, x2, y2 = box.round().astype(int)
        label = f"{names.get(class_id, class_id)} {score:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), color, line_width, cv2.LINE_AA)
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, line_width)
        outside = y1 - text_height - 6 >= 0
        text_y = y1 - 4 if outside else y1 + text_height + 4
        background_y = y1 - text_height - 6 if outside else y1
        cv2.rectangle(image, (x1, background_y), (x1 + text_width + 4, text_y + 2), color, -1, cv2.LINE_AA)
        cv2.putText(
            image,
            label,
            (x1 + 2, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            line_width,
            cv2.LINE_AA,
        )


def get_images(source: Path):
    """Return supported image paths and their common output root."""
    if source.is_file():
        if source.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image extension: {source.suffix}")
        return [source], source.parent
    if not source.is_dir():
        raise FileNotFoundError(f"Source does not exist: {source}")
    images = sorted(path for path in source.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise FileNotFoundError(f"No images found in: {source}")
    return images, source


def main():
    """Run ONNX inference on every source image and save annotated images."""
    args = parse_args()
    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"ONNX model does not exist: {model_path}")

    available = ort.get_available_providers()
    if args.device == "cuda" and "CUDAExecutionProvider" not in available:
        raise RuntimeError(f"CUDAExecutionProvider is unavailable; available providers: {available}")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if args.device == "cuda" else ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(model_path), providers=providers)
    model_input = session.get_inputs()[0]
    if len(model_input.shape) != 4 or not all(isinstance(x, int) for x in model_input.shape[2:]):
        raise ValueError(f"Expected a fixed BCHW ONNX input, got {model_input.shape}")
    input_height, input_width = model_input.shape[2:]

    metadata = session.get_modelmeta().custom_metadata_map
    names = ast.literal_eval(metadata.get("names", "{}"))
    output_shapes = [output.shape for output in session.get_outputs()]
    if len(output_shapes) != 8 or any(shape[1] != 4 for shape in output_shapes[:4]):
        raise ValueError(f"Expected 8 raw YOLO outputs (4 box + 4 class), got {output_shapes}")
    strides = [input_width // shape[3] for shape in output_shapes[:4]]

    images, source_root = get_images(args.source.expanduser().resolve())
    output_root = args.output.expanduser().resolve()
    detections = 0
    print(f"Model input: {input_width}x{input_height}; providers: {session.get_providers()}")
    for index, image_path in enumerate(images, 1):
        original = cv2.imread(str(image_path))
        if original is None:
            print(f"[{index}/{len(images)}] skip unreadable image: {image_path}")
            continue
        tensor, gain, (pad_x, pad_y) = letterbox(original, (input_height, input_width))
        tensor = np.ascontiguousarray(tensor[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        outputs = session.run(None, {model_input.name: tensor})
        boxes, scores, class_ids = decode(outputs, strides, args.conf, args.max_det)
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / gain
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / gain
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, original.shape[1] - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, original.shape[0] - 1)
        draw(original, boxes, scores, class_ids, names)

        destination = output_root / image_path.relative_to(source_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination), original):
            raise OSError(f"Failed to write: {destination}")
        detections += len(boxes)
        print(f"[{index}/{len(images)}] {image_path.name}: {len(boxes)} detections -> {destination}")
    print(f"Done: {len(images)} images, {detections} detections, output={output_root}")


if __name__ == "__main__":
    main()
