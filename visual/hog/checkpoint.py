import numpy as np
import joblib
from pathlib import Path


def save_checkpoint(detector, checkpoint_path, pos_patches, cur_epoch=0):
    path = Path(checkpoint_path)
    detector.save(path)
    cfg = np.load(path / "config.npy", allow_pickle=True).item()
    cfg['cur_epoch'] = cur_epoch
    np.save(path / "config.npy", cfg)

    for cls, comps in detector.cls_comps.items():
        for comp in comps:
            comp.save_training_data(path)

    joblib.dump(pos_patches, path / "pos_patches.pkl")
    print(f"Checkpoint saved to {path}")


def load_checkpoint(detector, checkpoint_path):
    path = Path(checkpoint_path)
    detector.load(path)
    cfg = np.load(path / "config.npy", allow_pickle=True).item()
    cur_epoch = cfg.get('cur_epoch', 0)

    for cls, comps in detector.cls_comps.items():
        for comp in comps:
            comp.load_training_data(path)

    pos_patches = joblib.load(path / "pos_patches.pkl")
    print(f"Checkpoint loaded from {path}")
    return cur_epoch, pos_patches