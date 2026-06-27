from pathlib import Path
import joblib
import numpy as np
from visual.hog.features.hog_descriptor import HOGDescriptor
from visual.hog.datastructures.component import Component
from visual.hog.inference.contextual_rescoring import ContextualRescorer


def save(detector, path) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    components_cfg: dict = {cls: [] for cls in detector.classes}
    total_components = 0
    for cls in detector.classes:
        safe = cls.replace(" ", "_")
        for comp in detector.cls_comps.get(cls, []):
            comp_id = int(comp.id)
            svm_file = f"svm_{safe}_c{comp_id}.pkl"
            calibration_file = f"cal_{safe}_c{comp_id}.pkl"
            joblib.dump(comp.svm, path / svm_file)
            joblib.dump(comp.cal, path / calibration_file)
            components_cfg[cls].append({
                "component_id": comp_id,
                "cell_w": int(comp.cell_w),
                "cell_h": int(comp.cell_h),
                "svm_file": svm_file,
                "calibration_file": calibration_file,
            })
            total_components += 1
    rescorer = getattr(detector, "contextual_rescorer", None)
    ctx_cfg = {
        "fitted": False,
        "iou_thresh": 0.5,
        "C": 1.0,
        "detection_threshold": 0.5,
        "svm_files": {},
        "neg_size": 30000,
    }
    if rescorer != None:
        ctx_cfg["fitted"] = bool(rescorer.fitted)
        ctx_cfg["iou_thresh"] = float(rescorer.iou_thresh)
        ctx_cfg["C"] = float(rescorer.C)
        ctx_cfg["detection_threshold"] = float(rescorer.detection_threshold)
        ctx_cfg["neg_size"] = int(rescorer.neg_size)
        if rescorer.fitted == True:
            for cls, svc in rescorer.rescorers.items():
                safe = cls.replace(" ", "_")
                ctx_file = f"ctx_{safe}.pkl"
                joblib.dump(svc, path / ctx_file)
                ctx_cfg["svm_files"][cls] = ctx_file
            print(f"  Contextual rescoring saved ({len(rescorer.rescorers)} class SVMs)")
    config = {
        "classes": list(detector.classes),
        "hog_params": dict(detector.hog_params),
        "n_components": int(detector.n_components),
        "c_svm": float(detector.c_svm),
        "max_itr_svm": int(detector.max_itr_svm),
        "training_epochs": int(detector.training_epochs),
        "area_percentile": float(detector.area_percentile),
        "hard_neg_threshold": float(detector.hard_neg_threshold),
        "pyramid_step": int(detector.pyramid_step),
        "max_hard_per_image": int(detector.max_hard_per_image),
        "bg_multiplier": float(detector.bg_multiplier),
        "other_classes_total_ratio": float(detector.other_classes_total_ratio),
        "bbr_alpha": float(detector.bbr_alpha),
        "min_iou_between_gt_and_latent": float(detector.min_iou_between_gt_and_latent),
        "default_window_size": tuple(detector.default_window_size),
        "trained_flag": bool(detector.trained_flag),
        "components": components_cfg,
        "regressors": {
            cls: {
                str(comp.id): comp.bbox_reg.get_state()
                for comp in detector.cls_comps[cls]
            }
            for cls in detector.classes
        },
        "contextual_rescoring": ctx_cfg,
    }
    np.save(path / "config.npy", config, allow_pickle=True)
    print(f"Model saved to '{path}/' ({len(detector.classes)} classes, {total_components} components)")


def load(detector, path):
    path = Path(path)
    if not path.exists() or not path.is_dir():
        print(f"Invalid path: {path}")
        return
    cfg = np.load(path / "config.npy", allow_pickle=True).item()
    detector.classes = list(cfg["classes"])
    detector.hog_params = dict(cfg["hog_params"])
    detector.n_components = int(cfg.get("n_components", 1))
    detector.c_svm = float(cfg.get("c_svm", 0.01))
    detector.max_itr_svm = int(cfg.get("max_itr_svm", 30000))
    detector.training_epochs = int(cfg.get("training_epochs", 1))
    detector.area_percentile = float(cfg.get("area_percentile", 80))
    detector.hard_neg_threshold = float(cfg.get("hard_neg_threshold", -1))
    detector.pyramid_step = int(cfg.get("pyramid_step", 2))
    detector.max_hard_per_image = int(cfg.get("max_hard_per_image", 20))
    detector.bg_multiplier = float(cfg.get("bg_multiplier", 2.0))
    detector.other_classes_total_ratio = float(cfg.get("other_classes_total_ratio", 1.0))
    detector.bbr_alpha = float(cfg.get("bbr_alpha", 1000.0))
    detector.min_iou_between_gt_and_latent = float(cfg.get("min_iou_between_gt_and_latent", 0.5))
    detector.default_window_size = tuple(cfg.get("default_window_size", (64, 64)))
    detector.hog_descriptor = HOGDescriptor(**detector.hog_params)
    detector.cls_comps = {cls: [] for cls in detector.classes}
    detector.svms = {cls: [] for cls in detector.classes}
    components_cfg = cfg.get("components", {})
    loaded_components = 0
    for cls in detector.classes:
        for comp_meta in components_cfg.get(cls, []):
            comp_id = int(comp_meta["component_id"])
            svm_path = path / comp_meta["svm_file"]
            calibration_path = path / comp_meta["calibration_file"]

            if not svm_path.exists():
                continue

            comp = Component(
                component_id=comp_id,
                class_name=cls,
                cell_w=int(comp_meta["cell_w"]),
                cell_h=int(comp_meta["cell_h"]),
                cell_size=detector.hog_descriptor.cell_size,
                c_svm=detector.c_svm,
                max_itr_svm=detector.max_itr_svm,
                alpha=detector.bbr_alpha,
            )
            comp.svm = joblib.load(svm_path)
            if calibration_path.exists():
                comp.cal = joblib.load(calibration_path)
            else:
                comp.cal = None

            detector.cls_comps[cls].append(comp)
            loaded_components += 1

        detector.cls_comps[cls].sort(key=lambda c: c.id)
        detector.svms[cls] = [comp.svm for comp in detector.cls_comps[cls]]

    detector.trained_flag = bool(cfg.get("trained_flag", loaded_components > 0))
    reg_states = cfg.get("regressors", {})
    for cls in detector.classes:
        id_to_comp = {comp.id: comp for comp in detector.cls_comps[cls]}
        for comp_id_str, state in reg_states.get(cls, {}).items():
            comp_id = int(comp_id_str)
            if comp_id in id_to_comp:
                id_to_comp[comp_id].bbox_reg.set_state(state)

    ctx_cfg = cfg.get("contextual_rescoring", {})
    iou_thresh = float(ctx_cfg.get("iou_thresh", 0.5))
    C = float(ctx_cfg.get("C", 1.0))
    detection_threshold = float(ctx_cfg.get("detection_threshold", 0.5))
    fitted = bool(ctx_cfg.get("fitted", False))
    svm_files = dict(ctx_cfg.get("svm_files", {}))
    neg_size = int(ctx_cfg.get("neg_size", 30000))
    rescorer = ContextualRescorer(
        classes=detector.classes,
        iou_thresh=iou_thresh,
        C=C,
        detection_threshold=detection_threshold,
        neg_size=neg_size,
    )

    if fitted == True and svm_files:
        loaded_ctx = 0
        missing_ctx = []
        for cls, filename in svm_files.items():
            ctx_path = path / filename
            if ctx_path.exists():
                rescorer.rescorers[cls] = joblib.load(ctx_path)
                loaded_ctx += 1
            else:
                missing_ctx.append(cls)

        if len(missing_ctx) > 0:
            print(f"  Warning: contextual rescoring SVM files missing for: {missing_ctx}.  Rescorer marked as unfitted.")
            rescorer.fitted = False
        else:
            rescorer.fitted = True
            print(f"  Contextual rescoring loaded ({loaded_ctx} class SVMs)")
    else:
        rescorer.fitted = False
        if fitted == True and not svm_files:
            print("  Warning: config marks contextual rescoring as fitted but no SVM files were recorded.  Rescorer left unfitted.")
    detector.contextual_rescorer = rescorer
    print(f"Model loaded from '{path}/' ({len(detector.classes)} classes, {loaded_components} components)")
