# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Export fixed-shape official YOLO26s-P2 raw detection heads for the GK020 NPU."""

import argparse
from pathlib import Path

from ultralytics import YOLO
from ultralytics.nn.modules import Attention


ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "runs/detect/yolo26s/weights/best.pt"
INPUT_SIZE = (544, 960)  # height, width
OPSET = 13  # Supported by GK020 and used by the verified reference deployment.


def parse_args():
    """Parse export options while keeping the board input shape fixed."""
    parser = argparse.ArgumentParser(description="Export official YOLO26s-P2 raw heads for GK020.")
    parser.add_argument("--weights", type=Path, default=WEIGHTS, help="YOLO26s-P2 checkpoint path.")
    parser.add_argument(
        "--head",
        choices=("one-to-many", "one-to-one"),
        default="one-to-one",
        help="Detection head to export: one-to-many needs NMS; one-to-one needs TopK and no NMS.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output ONNX path. Defaults to <weights>_one2many.onnx or <weights>_one2one.onnx.",
    )
    return parser.parse_args()


def validate_gk020_onnx(path: Path, num_levels: int, num_classes: int, reg_max: int):
    """Reject graph patterns that violate the documented GK020 ONNX limits."""
    import onnx

    graph = onnx.shape_inference.infer_shapes(onnx.load(path)).graph
    ranks = {}
    for value in [*graph.input, *graph.value_info, *graph.output]:
        shape = value.type.tensor_type.shape
        if shape.dim:
            ranks[value.name] = len(shape.dim)

    unsupported = {node.op_type for node in graph.node} & {"TopK", "NonMaxSuppression"}
    if unsupported:
        raise RuntimeError(f"GK020 export contains unsupported operators: {sorted(unsupported)}")
    bad_matmul = [node.name for node in graph.node if node.op_type == "MatMul" and ranks.get(node.input[0]) != 3]
    if bad_matmul:
        raise RuntimeError(f"GK020 only supports 3D MatMul; invalid nodes: {bad_matmul}")
    bad_split = []
    for node in graph.node:
        if node.op_type != "Split":
            continue
        axis = next((attr.i for attr in node.attribute if attr.name == "axis"), 0)
        if ranks.get(node.input[0]) == 4 and axis != 1:
            bad_split.append(node.name)
    if bad_split:
        raise RuntimeError(f"GK020 only supports 4D Split on axis=1; invalid nodes: {bad_split}")

    expected_outputs = num_levels * 2  # all box maps followed by all class maps
    if len(graph.output) != expected_outputs:
        raise RuntimeError(f"Expected {expected_outputs} raw outputs, got {len(graph.output)}")
    output_channels = [output.type.tensor_type.shape.dim[1].dim_value for output in graph.output]
    expected_channels = [4 * reg_max] * num_levels + [num_classes] * num_levels
    if output_channels != expected_channels:
        raise RuntimeError(f"Unexpected output channels: expected {expected_channels}, got {output_channels}")


def main():
    """Export regression maps followed by classification maps for each feature level."""
    args = parse_args()
    weights = args.weights.expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Model weights not found: {weights}")

    model = YOLO(weights, task="detect")
    head = model.model.model[-1]
    if getattr(head, "nl", None) not in {3, 4} or getattr(head, "reg_max", None) != 1:
        raise ValueError(
            f"Expected three- or four-scale DFL-free YOLO26 head, got nl={getattr(head, 'nl', None)}, "
            f"reg_max={getattr(head, 'reg_max', None)}"
        )
    if args.head == "one-to-one" and getattr(head, "one2one_cv2", None) is None:
        raise ValueError("This checkpoint has no one-to-one detection head; export a complete YOLO26 checkpoint.")
    head.end2end = args.head == "one-to-one"  # Make fuse() retain the selected detection branch.
    head.export_onnx = True
    head.export_onnx_head = args.head
    for module in model.model.modules():
        if isinstance(module, Attention):
            module.export_gk020 = True

    export_args = dict(
        format="onnx",
        imgsz=INPUT_SIZE,
        batch=1,
        opset=OPSET,
        dynamic=False,
        simplify=True,
        device="cpu",
    )
    if args.head == "one-to-one":
        # The exporter uses an explicit nms=False to preserve the end-to-end branch during fuse(). Raw output mode
        # below still bypasses the built-in TopK, so TopK remains a board-side post-processing operation.
        export_args["nms"] = False
    output = model.export(**export_args)
    output = Path(output)
    validate_gk020_onnx(output, head.nl, head.nc, head.reg_max)

    suffix = args.head.replace("-", "")
    destination = (args.output or weights.with_name(f"{weights.stem}_{suffix}.onnx")).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if output != destination:
        output.replace(destination)
    print(f"GK020 {args.head} ONNX validated: {destination}")


if __name__ == "__main__":
    main()
