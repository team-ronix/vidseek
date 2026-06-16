import os
import math
import pickle
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from tqdm import tqdm

from data import get_pair_dataloaders, get_dataloaders
from models.model.transformer import Transformer
from losses.mnr_loss import MultipleNegativesRankingLoss

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG = {
    # data
    "train_path":    "data/stsb_train.csv",
    "val_path":      "data/stsb_validation.csv",
    "test_path":     "data/stsb_test.csv",
    "pos_threshold": 0.65,
    "max_len":       128,
    "batch_size":    64,
    "sample_size":   None,   # cap training pairs; None = use all
    "min_freq":      2,
    "num_workers":   0,      # 2-4 on Linux/Kaggle; 0 on Windows
    "pin_memory":    True,   # auto-disabled when no CUDA

    # model
    "d_model":   256,
    "n_layers":  4,
    "n_heads":   4,          # d_k = 64 (256 // 4)
    "d_ff":      512,
    "dropout":   0.1,
    "pooling":   "mean",

    # MNR loss
    # temperature=0.15 prevents representation collapse on small batches.
    # The common default 0.05 is designed for batch >= 512 and pre-trained models;
    # with batch=64 and a randomly initialised model it creates gradients too large
    # for the early stages of training, causing collapse after epoch 1.
    "temperature": 0.15,

    # optimiser
    "peak_lr":      3e-4,
    "weight_decay": 0.01,
    "warmup_steps": 100,

    # training
    "epochs":    20,
    "clip_grad": 1.0,

    # checkpointing
    "checkpoint_dir":    "checkpoints",
    "checkpoint_latest": "checkpoints/checkpoint_latest.pt",
    "save_path":         "best_model.pt",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# LR schedule: linear warmup -> cosine decay to zero
# ---------------------------------------------------------------------------

def make_cosine_with_warmup(total_steps, warmup_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / max(1, warmup_steps)
        progress = float(current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return lr_lambda


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(path, model, optimizer, scheduler, epoch, step, best_rho, history):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "epoch":                epoch,
        "global_step":          step,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_rho":         best_rho,
        "train_loss_history":   history,
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, total_epochs):
    """Return (start_epoch, global_step, best_val_rho, history).

    Returns (0, 0, -1.0, []) when: no file, corrupt file, or saved run
    already completed all epochs (stale checkpoint from a previous run).
    """
    if not os.path.exists(path):
        return 0, 0, -1.0, []
    try:
        ckpt = torch.load(path, map_location="cpu")
        saved_epoch = int(ckpt.get("epoch", 0))
        if saved_epoch >= total_epochs:
            # Checkpoint from a completed run -- ignore it so we train fresh.
            print(f"  Stale checkpoint (epoch {saved_epoch} of {total_epochs} already done) -- starting fresh.")
            return 0, 0, -1.0, []
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        step     = int(ckpt.get("global_step", 0))
        best_rho = float(ckpt.get("best_val_rho", -1.0))
        history  = ckpt.get("train_loss_history", [])
        print(f"  Loaded checkpoint: epoch={saved_epoch}  step={step}  best_rho={best_rho:.4f}")
        return saved_epoch, step, best_rho, history
    except Exception as exc:
        print(f"  WARNING: checkpoint load failed ({exc}) -- starting from scratch")
        return 0, 0, -1.0, []


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def spearman_on_sts(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    for ids_a, ids_b, mask_a, mask_b, scores in tqdm(loader, desc="Eval ", leave=False):
        emb_a = model.encode(ids_a.to(device), mask_a.to(device), normalize=True)
        emb_b = model.encode(ids_b.to(device), mask_b.to(device), normalize=True)
        all_preds.append((emb_a * emb_b).sum(-1).cpu())
        all_targets.append(scores)
    preds   = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()
    if preds.std() == 0 or targets.std() == 0:
        return 0.0
    rho, _ = spearmanr(preds, targets)
    return float(rho) if rho == rho else 0.0


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, scheduler, criterion, device, start_step=0):
    model.train()
    total_loss = 0.0
    step = start_step
    for ids_a, ids_b, mask_a, mask_b in tqdm(loader, desc="Train", leave=False):
        optimizer.zero_grad()
        emb_a = model.encode(ids_a.to(device), mask_a.to(device), normalize=True)
        emb_b = model.encode(ids_b.to(device), mask_b.to(device), normalize=True)
        loss  = criterion(emb_a, emb_b)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CONFIG["clip_grad"])
        optimizer.step()
        scheduler.step()
        step      += 1
        total_loss += loss.item()
    return total_loss / max(len(loader), 1), step


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

    print(f"Device      : {DEVICE}")
    print(f"Loss        : MNR  temperature={CONFIG['temperature']}")
    print(f"d_model={CONFIG['d_model']}  n_layers={CONFIG['n_layers']}  "
          f"n_heads={CONFIG['n_heads']}  d_ff={CONFIG['d_ff']}")

    print("\nLoading data ...")
    pin = CONFIG["pin_memory"] and torch.cuda.is_available()

    pair_loaders, vocab = get_pair_dataloaders(
        train_path=CONFIG["train_path"],
        val_path=CONFIG["val_path"],
        test_path=CONFIG["test_path"],
        batch_size=CONFIG["batch_size"],
        max_len=CONFIG["max_len"],
        pos_threshold=CONFIG["pos_threshold"],
        min_freq=CONFIG["min_freq"],
        sample_size=CONFIG["sample_size"],
        num_workers=CONFIG["num_workers"],
        pin_memory=pin,
    )
    sts_loaders, _ = get_dataloaders(
        train_path=CONFIG["train_path"],
        val_path=CONFIG["val_path"],
        test_path=CONFIG["test_path"],
        batch_size=CONFIG["batch_size"],
        max_len=CONFIG["max_len"],
        vocab=vocab,
        num_workers=CONFIG["num_workers"],
        pin_memory=pin,
    )

    steps_per_epoch = len(pair_loaders["train"])
    total_steps     = CONFIG["epochs"] * steps_per_epoch

    with open("vocab.pkl", "wb") as _vf:
        pickle.dump(vocab, _vf)

    print(f"Vocab size    : {len(vocab):,}  (saved to vocab.pkl)")
    print(f"Train pairs   : {len(pair_loaders['train'].dataset):,}")  # type: ignore[arg-type]
    print(f"Val STS pairs : {len(sts_loaders['val'].dataset):,}")  # type: ignore[arg-type]
    print(f"Steps/epoch   : {steps_per_epoch}   total: {total_steps}")

    model = Transformer(
        vocab_size=len(vocab),
        d_model=CONFIG["d_model"],
        n_layers=CONFIG["n_layers"],
        n_heads=CONFIG["n_heads"],
        d_ff=CONFIG["d_ff"],
        max_len=CONFIG["max_len"],
        dropout=CONFIG["dropout"],
        pooling=CONFIG["pooling"],
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters    : {n_params:,}\n")

    criterion = MultipleNegativesRankingLoss(CONFIG["temperature"])

    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "norm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params,    "weight_decay": CONFIG["weight_decay"]},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=CONFIG["peak_lr"],
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=make_cosine_with_warmup(total_steps, CONFIG["warmup_steps"]),
    )

    ckpt_path = CONFIG["checkpoint_latest"]
    start_epoch, global_step, best_val_rho, loss_history = load_checkpoint(
        ckpt_path, model, optimizer, scheduler, CONFIG["epochs"]
    )

    if start_epoch > 0:
        print(f"Resuming from epoch {start_epoch + 1}/{CONFIG['epochs']}")
    else:
        print("Starting training from scratch")

    for epoch in range(start_epoch + 1, CONFIG["epochs"] + 1):
        train_loss, global_step = train_epoch(
            model, pair_loaders["train"], optimizer, scheduler,
            criterion, DEVICE, start_step=global_step,
        )
        val_rho    = spearman_on_sts(model, sts_loaders["val"], DEVICE)
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch:02d}/{CONFIG['epochs']}  "
            f"lr={current_lr:.2e}  "
            f"| MNR loss={train_loss:.4f}  "
            f"| Val Spearman={val_rho:.4f}"
        )

        loss_history.append({"epoch": epoch, "loss": train_loss, "val_rho": val_rho})

        save_checkpoint(
            ckpt_path, model, optimizer, scheduler,
            epoch, global_step, best_val_rho, loss_history,
        )

        if val_rho > best_val_rho:
            best_val_rho = val_rho
            torch.save(model.state_dict(), CONFIG["save_path"])
            print(f"  * Best model saved  (val Spearman = {best_val_rho:.4f})")

    print(f"\nTraining done.  Best Val Spearman = {best_val_rho:.4f}")

    if "test" in sts_loaders:
        model.load_state_dict(torch.load(CONFIG["save_path"], map_location=DEVICE))
        test_rho = spearman_on_sts(model, sts_loaders["test"], DEVICE)
        print(f"Test Spearman = {test_rho:.4f}")


if __name__ == "__main__":
    main()
