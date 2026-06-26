import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
import numpy as np
from visual.hog.datastructures.voc_dataset import VOCDataset
from visual.hog.hog_detector import HOGDetector
from visual.hog.eval import evaluate_hog_detection_ap


VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable","dog", "horse", "motorbike", "person",
    "pottedplant","sheep", "sofa", "train", "tvmonitor",
]



def print_results(ap_per_class, mean_ap, classes):
    active = [c for c in classes if c in ap_per_class]
    print(f"\n{'Class':<15} {'AP':>8}")
    print("-" * 26)
    for cls in active:
        ap = ap_per_class[cls]
        tag = f"{ap}" if not np.isnan(ap) else "   N/A"
        print(f"  {cls}  {tag:}")
    print("-" * 26)
    print(f"  {'mAP':<13}  {mean_ap:>8.4f}\n")


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


def _config_value(config: dict, key: str, default=None):
    value = config.get(key, default)
    return default if value in (None, "") else value


def parse_args():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None,help="Path to a JSON file with default training arguments")
    pre_args, remaining = pre.parse_known_args()
    config = _load_config(pre_args.config)
    p = argparse.ArgumentParser(
        description="Train HOG detector on VOC and evaluate on test split",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[pre],
    )
    p.add_argument("--train-root", default=_config_value(config, "train_root"),
                   required=_config_value(config, "train_root") in (None, ""),
                   help="Root of the VOC train/val split (contains JPEGImages, Annotations, ImageSets)")
    p.add_argument("--test-root", default=_config_value(config, "test_root"),
                   required=_config_value(config, "test_root") in (None, ""),
                   help="Root of the VOC test split")
    p.add_argument("--train-split", default=_config_value(config, "train_split"),
                   help="Path to trainval.txt (auto-detected if omitted)")
    p.add_argument("--test-split", default=_config_value(config, "test_split"),
                   help="Path to test.txt (auto-detected if omitted)")
    p.add_argument("--model-dir", default=_config_value(config, "model_dir", "./model"),
                   help="Directory to save/load the trained model")
    p.add_argument("--checkpoint-dir", default=_config_value(config, "checkpoint_dir", "./checkpoint"),
                   help="Directory for training checkpoints")
    p.add_argument("--skip-step1", action="store_true",
                   default=bool(_config_value(config, "skip_step1", False)),
                   help="Resume training from an existing checkpoint (skip Step 1)")
    p.add_argument("--eval-only", action="store_true",
                   default=bool(_config_value(config, "eval_only", False)),
                   help="Skip training; load model from --model-dir and evaluate")
    p.add_argument("--cell-size", type=int, default=_config_value(config, "cell_size", 8))
    p.add_argument("--n-orient-cs", type=int, default=_config_value(config, "n_orient_cs", 18))
    p.add_argument("--n-orient-ci", type=int, default=_config_value(config, "n_orient_ci", 9))
    p.add_argument("--alpha", type=float, default=_config_value(config, "alpha", 0.2))
    p.add_argument("--n-energy", type=int, default=_config_value(config, "n_energy", 4))
    p.add_argument("--n-octaves", type=int, default=_config_value(config, "n_octaves", 4))
    p.add_argument("--llambda", type=int, default=_config_value(config, "llambda", 4))
    p.add_argument("--min-size", type=int, default=_config_value(config, "min_size", 48))
    p.add_argument("--n-components", type=int, default=_config_value(config, "n_components", 3))
    p.add_argument("--c-svm", type=float, default=_config_value(config, "c_svm", 0.001))
    p.add_argument("--max-itr-svm", type=int, default=_config_value(config, "max_itr_svm", 50_000))
    p.add_argument("--epochs", type=int, default=_config_value(config, "training_epochs", 10), dest="training_epochs")
    p.add_argument("--hard-neg-threshold", type=float, default=_config_value(config, "hard_neg_threshold", -1.0))
    p.add_argument("--pyramid-step", type=int, default=_config_value(config, "pyramid_step", 2))
    p.add_argument("--max-hard-per-image", type=int, default=_config_value(config, "max_hard_per_image", 20))
    p.add_argument("--bg-multiplier", type=float, default=_config_value(config, "bg_multiplier", 3.0))
    p.add_argument("--other-cls-ratio", type=float, default=_config_value(config, "other_classes_total_ratio", 1.0),
                   dest="other_classes_total_ratio")
    p.add_argument("--bbr-alpha", type=float, default=_config_value(config, "bbr_alpha", 100.0))
    p.add_argument("--min-iou-latent", type=float, default=_config_value(config, "min_iou_between_gt_and_latent", 0.5),
                   dest="min_iou_between_gt_and_latent")
    p.add_argument("--rescorer-iou-thresh", type=float, default=_config_value(config, "rescorer_iou_thresh", 0.5))
    p.add_argument("--rescorer-c", type=float, default=_config_value(config, "rescorer_c", 0.1))
    p.add_argument("--rescorer-det-threshold", type=float, default=_config_value(config, "rescorer_det_threshold", 0.05))
    p.add_argument("--rescorer-neg-size", type=int, default=_config_value(config, "rescorer_neg_size", 30_000))
    p.add_argument("--rescorer-checkpoint-every", type=int, default=_config_value(config, "rescorer_checkpoint_every", 50))
    p.add_argument("--max-train-images", type=int, default=_config_value(config, "max_train_images"))
    p.add_argument("--max-rescore-images", type=int, default=_config_value(config, "max_rescore_images"))
    p.add_argument("--neg-patches", type=int, default=_config_value(config, "neg_patches_per_image", 10),
                   dest="neg_patches_per_image")
    p.add_argument("--min-box-area", type=int, default=_config_value(config, "min_box_area", 0))
    p.add_argument("--eval-pyramid-lambda", type=int, default=_config_value(config, "eval_pyramid_lambda", 4))
    p.add_argument("--eval-score-thresh", type=float, default=_config_value(config, "eval_score_thresh", 0.05))
    p.add_argument("--eval-iou-match", type=float, default=_config_value(config, "eval_iou_match", 0.5))
    p.add_argument("--eval-nms-iou", type=float, default=_config_value(config, "eval_nms_iou", 0.3))
    p.add_argument("--eval-max-images", type=int, default=_config_value(config, "eval_max_images"))
    p.add_argument("--no-context", action="store_true",
                   default=bool(_config_value(config, "no_context", False)),
                   help="Disable contextual rescoring during evaluation")
    p.add_argument("--eval-checkpoint", default=_config_value(config, "eval_checkpoint"),
                   help="Pickle file for resumable evaluation progress")
    return p.parse_args(remaining)



def find_split_file(root, split_name):
    path = os.path.join(root, "ImageSets", "Main", f"{split_name}.txt")
    return path if os.path.exists(path) else None


def main():
    args = parse_args()
    CLASS_TO_IDX = {c: i for i, c in enumerate(VOC_CLASSES)}
    train_split = args.train_split or find_split_file(args.train_root, "trainval")
    test_split = args.test_split or find_split_file(args.test_root, "test")
    print("Building datasets ")
    train_ds = VOCDataset(args.train_root, train_split, class_to_idx=CLASS_TO_IDX)
    test_ds = VOCDataset(args.test_root, test_split, class_to_idx=CLASS_TO_IDX)
    print(f" Train: {len(train_ds)} images | Test: {len(test_ds)} images")
    hog_params = dict(
        cell_size = args.cell_size,
        n_orient_cs = args.n_orient_cs,
        n_orient_ci = args.n_orient_ci,
        alpha = args.alpha,
        n_energy = args.n_energy,
        n_octaves = args.n_octaves,
        llambda = args.llambda,
        min_size = args.min_size,
    )
    detector = HOGDetector(
        classes = VOC_CLASSES,
        hog_descriptor_params = hog_params,
        n_components = args.n_components,
        c_svm = args.c_svm,
        max_itr_svm = args.max_itr_svm,
        training_epochs = args.training_epochs,
        hard_neg_threshold = args.hard_neg_threshold,
        pyramid_step = args.pyramid_step,
        max_hard_per_image = args.max_hard_per_image,
        bg_multiplier = args.bg_multiplier,
        other_classes_total_ratio = args.other_classes_total_ratio,
        bbr_alpha = args.bbr_alpha,
        min_iou_between_gt_and_latent = args.min_iou_between_gt_and_latent,
        contextual_rescorer_iou_thresh = args.rescorer_iou_thresh,
        contextual_rescorer_C = args.rescorer_c,
        contextual_rescorer_detection_threshold = args.rescorer_det_threshold,
        contextual_rescorer_neg_size = args.rescorer_neg_size,
    )
    if args.eval_only:
        print(f"\nLoading model from {args.model_dir}")
        detector.load(args.model_dir)
    else:
        print("\nStarting training")
        t0 = time.time()
        detector.train(
            train_ds = train_ds,
            min_box_area = args.min_box_area,
            max_train_images = args.max_train_images,
            max_rescore_images = args.max_rescore_images,
            neg_patches_per_image = args.neg_patches_per_image,
            checkpoint_path = args.checkpoint_dir,
            skip_step1 = args.skip_step1,
            rescorer_checkpoint_every = args.rescorer_checkpoint_every,
        )
        print(f"\nTraining finished in {(time.time() - t0) / 60:.1f} min")
        detector.save(args.model_dir)
        print(f"Model saved -> {args.model_dir}")
    use_context = not args.no_context
    print(f"\nEvaluating on test split (use_context={use_context})")
    ap_per_class, mean_ap = evaluate_hog_detection_ap(
        detector = detector,
        dataset = test_ds,
        max_imgs = args.eval_max_images,
        iou_match_thresh = args.eval_iou_match,
        nms_iou_thresh = args.eval_nms_iou,
        score_thresh = args.eval_score_thresh,
        split_name = "Test",
        pyramid_lambda = args.eval_pyramid_lambda,
        use_context = use_context,
        checkpoint_path = args.eval_checkpoint,
        checkpoint_every = 50,
    )
    print(f"\nTest mAP@0.5 (VOC 11-pt): {mean_ap:.4f}")
    print_results(ap_per_class, mean_ap, VOC_CLASSES)


if __name__ == "__main__":
    main()
