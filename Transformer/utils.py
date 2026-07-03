import math
import os

import torch
from scipy.stats import spearmanr


def make_cosine_with_warmup(total_steps, warmup_steps):
    # Learning rate schedule:
    #   - linearly warm up from 0 to peak_lr over warmup_steps
    #   - then decay following a cosine curve down to ~0
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return lr_lambda


def save_checkpoint(path, model, optimizer, scheduler, epoch, step, best_rho):
    # save everything needed to resume training from this exact point
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "epoch":                epoch,
        "global_step":          step,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_rho":         best_rho,
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, total_epochs):
    # try to resume from a saved checkpoint; return defaults if nothing found
    if not os.path.exists(path):
        return 0, 0, -1.0
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if int(ckpt.get("epoch", 0)) >= total_epochs:
            print("  Stale checkpoint -- starting fresh.")
            return 0, 0, -1.0
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        epoch    = int(ckpt.get("epoch", 0))
        step     = int(ckpt.get("global_step", 0))
        best_rho = float(ckpt.get("best_val_rho", -1.0))
        print(f"  Resumed: epoch={epoch}  step={step}  best_rho={best_rho:.4f}")
        return epoch, step, best_rho
    except Exception as e:
        print(f"  Checkpoint load failed ({e}) -- starting fresh.")
        return 0, 0, -1.0


@torch.no_grad()
def spearman_on_sts(model, loader, device):
    # evaluate how well predicted cosine similarities correlate with human scores
    # Spearman correlation measures rank agreement, not exact values
    model.eval()
    preds, targets = [], []
    for ids_a, ids_b, mask_a, mask_b, scores in loader:
        emb_a = model.encode(ids_a.to(device), mask_a.to(device), normalize=True)
        emb_b = model.encode(ids_b.to(device), mask_b.to(device), normalize=True)
        # cosine similarity of normalized vectors = dot product
        preds.append((emb_a * emb_b).sum(-1).cpu())
        targets.append(scores)
    preds   = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    if preds.std() == 0 or targets.std() == 0:
        return 0.0
    rho, _ = spearmanr(preds, targets)
    return float(rho) if rho == rho else 0.0  # rho == rho is False for NaN
