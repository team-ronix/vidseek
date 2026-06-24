import argparse
import json
import sys
from pathlib import Path
import cv2
import numpy as np
from visual.hog.hog_detector import HOGDetector

colors = [
    (214, 39, 40), (255, 127, 14), ( 44, 160, 44), ( 31, 119, 180),
    (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
    ( 23, 190, 207), (188, 189, 34), (174, 199, 232), (255, 187, 120),
    (152, 223, 138), (255, 152, 150), (197, 176, 213), (196, 156, 148),
    (247, 182, 210), (199, 199, 199), (219, 219, 141), (158, 218, 229),
]

def _color(cls: str, class_list: list) -> tuple:
    idx = class_list.index(cls) if cls in class_list else 0
    return colors[idx % len(colors)]


def draw_detections(image: np.ndarray, boxes, scores, labels, class_list: list) -> np.ndarray:
    out = image.copy()
    for box, score, lbl in zip(boxes, scores, labels):
        x0, y0, x1, y1 = (int(v) for v in box)
        color = _color(lbl, class_list)
        cv2.rectangle(out, (x0, y0), (x1, y1), color, 2)
        text  = f"{lbl}: {score:.2f}"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x0, y0 - th - baseline - 4), (x0 + tw + 4, y0), color, -1)
        cv2.putText(out, text, (x0 + 2, y0 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _load_config(config_path: str | None) -> dict:
    if not config_path:
        return {}

    config_file = Path(config_path)
    if not config_file.exists():
        sys.exit(f"Error: config file not found: {config_file}")

    with config_file.open("r", encoding="utf-8") as f:
        config = json.load(f)

    if not isinstance(config, dict):
        sys.exit(f"Error: config file must contain a JSON object: {config_file}")

    return config

def parse_args():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None,
                     help="Path to a JSON file with default inference arguments")
    pre_args, remaining = pre.parse_known_args()
    config = _load_config(pre_args.config)

    p = argparse.ArgumentParser(
        description="Run HOG detector inference on a single image",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[pre],
    )
    p.add_argument("--model-dir", default=config.get("model_dir"),
                   required=config.get("model_dir") in (None, ""),
                   help="Directory of the saved model (created by pipeline.py)")
    p.add_argument("--image", default=config.get("image"),
                   required=config.get("image") in (None, ""),
                   help="Path to the input image")
    p.add_argument("--threshold", type=float, default=config.get("threshold", 0.3),
                   help="Minimum detection score to keep")
    p.add_argument("--nms-thresh", type=float, default=config.get("nms_thresh", 0.3),
                   help="NMS IoU threshold")
    p.add_argument("--pyramid-lambda", type=int, default=config.get("pyramid_lambda"),
                   help="Feature pyramid scale steps (None = detector default)")
    p.add_argument("--no-context", action="store_true",
                   default=bool(config.get("no_context", False)),
                   help="Disable contextual rescoring")
    p.add_argument("--output", default=config.get("output"),
                   help="Save annotated image to this path (optional)")
    p.add_argument("--json", action="store_true",
                   default=bool(config.get("json", False)),
                   help="Print detections as a JSON array and exit (no image display)")
    return p.parse_args(remaining)


def main():
    args = parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        sys.exit(f"Error: image not found: {image_path}")

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        sys.exit(f"Error: model directory not found: {model_dir}")

    detector = HOGDetector(
        classes=[],
        hog_descriptor_params=dict(
            cell_size=8, n_orient_cs=18, n_orient_ci=9,
            alpha=0.2, n_energy=4, n_octaves=4, llambda=4, min_size=48,
        ),
    )
    detector.load(str(model_dir))
    print(f"Model loaded from {model_dir}")
    print(f" Classes: {detector.classes}")
    print(f" Components: {detector.n_components} per class")
    print(f" Rescorer: {'fitted' if detector.contextual_rescorer.fitted else 'not fitted'}")

    image = cv2.imread(str(image_path))
    if image is None:
        sys.exit(f"Error: could not read image: {image_path}")
    print(f"\nImage: {image_path} ({image.shape[1]}x{image.shape[0]} px)")

    use_context = not args.no_context
    boxes, scores, labels = detector.detect(
        image,
        threshold = args.threshold,
        overlap_threshold = args.nms_thresh,
        pyramid_lambda = args.pyramid_lambda,
        use_context = use_context,
    )

    if args.json:
        results = [
            {"box": list(int(v) for v in box), "score": round(float(s), 4), "label": lbl}
            for box, s, lbl in zip(boxes, scores, labels)
        ]
        print(json.dumps(results, indent=2))
        return results

    print(f"\nDetections: {len(boxes)}")
    if boxes:
        print(f"  {'Label':<15} {'Score':>6}   {'x0':>5} {'y0':>5} {'x1':>5} {'y1':>5}")
        print(f"  {'-'*52}")
        for box, score, lbl in zip(boxes, scores, labels):
            x0, y0, x1, y1 = (int(v) for v in box)
            print(f"  {lbl:<15} {score:>6.3f}   {x0:>5} {y0:>5} {x1:>5} {y1:>5}")
    else:
        print("  (no detections above threshold)")

    if args.output:
        annotated = draw_detections(image, boxes, scores, labels, detector.classes)
        cv2.imwrite(args.output, annotated)
        print(f"\nAnnotated image saved -> {args.output}")

    return boxes, scores, labels


if __name__ == "__main__":
    main()
