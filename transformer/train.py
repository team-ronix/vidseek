import os
from typing import cast
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from tqdm import tqdm

from data import get_dataloaders
from models.model.sts_transformer import STSTransformer

CONFIG = {
    "train_path": "data/train.csv",
    "val_path":   "data/val.csv",
    "test_path":  "data/test.csv",
    "max_len":    128,
    "batch_size": 32,

    "d_model":   256,
    "n_layers":  4,
    "n_heads":   8,
    "d_ff":      512,
    "dropout":   0.1,

    "epochs":    8,
    "lr":        1e-4,
    "clip_grad": 1.0,

    "save_path": "best_model.pt",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def pearson_correlation(preds, targets):
    preds = preds.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()

    finite_mask = torch.isfinite(torch.tensor(preds)) & torch.isfinite(torch.tensor(targets))
    finite_mask = finite_mask.numpy()
    preds = preds[finite_mask]
    targets = targets[finite_mask]

    if len(preds) < 2:
        return 0.0
    if preds.std() == 0 or targets.std() == 0:
        return 0.0

    corr = cast(float, pearsonr(preds, targets)[0])
    if corr != corr:
        return 0.0
    return corr
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for batch in tqdm(loader, desc="Train", leave=False):
        if len(batch) == 5:
            ids_a, ids_b, mask_a, mask_b, scores = batch
        elif len(batch) == 3:
            ids_a, ids_b, scores = batch
            mask_a = (ids_a != 0).long()
            mask_b = (ids_b != 0).long()
        else:
            raise ValueError(f"Unexpected batch format with {len(batch)} elements")

        ids_a   = ids_a.to(DEVICE)
        ids_b   = ids_b.to(DEVICE)
        mask_a  = mask_a.to(DEVICE)
        mask_b  = mask_b.to(DEVICE)
        scores  = scores.to(DEVICE)

        optimizer.zero_grad()
        preds = model(ids_a, ids_b, mask_a, mask_b)
        loss  = criterion(preds, scores)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), CONFIG["clip_grad"])
        optimizer.step()

        total_loss += loss.item()
        all_preds.append(preds.detach())
        all_targets.append(scores.detach())

    avg_loss = total_loss / len(loader)
    corr = pearson_correlation(
        torch.cat(all_preds), torch.cat(all_targets)
    )
    return avg_loss, corr
@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for batch in tqdm(loader, desc="Eval ", leave=False):
        if len(batch) == 5:
            ids_a, ids_b, mask_a, mask_b, scores = batch
        elif len(batch) == 3:
            ids_a, ids_b, scores = batch
            mask_a = (ids_a != 0).long()
            mask_b = (ids_b != 0).long()
        else:
            raise ValueError(f"Unexpected batch format with {len(batch)} elements")

        ids_a   = ids_a.to(DEVICE)
        ids_b   = ids_b.to(DEVICE)
        mask_a  = mask_a.to(DEVICE)
        mask_b  = mask_b.to(DEVICE)
        scores  = scores.to(DEVICE)

        preds = model(ids_a, ids_b, mask_a, mask_b)
        loss  = criterion(preds, scores)

        total_loss += loss.item()
        all_preds.append(preds)
        all_targets.append(scores)

    avg_loss = total_loss / len(loader)
    corr = pearson_correlation(
        torch.cat(all_preds), torch.cat(all_targets)
    )
    return avg_loss, corr
def main():
    print(f"Device: {DEVICE}")

    print("Loading data...")
    loaders, vocab = get_dataloaders(
        train_path = CONFIG["train_path"],
        val_path   = CONFIG["val_path"],
        test_path  = CONFIG["test_path"],
        batch_size = CONFIG["batch_size"],
        max_len    = CONFIG["max_len"],
    )
    print(f"Vocab size: {len(vocab):,}")
    print(f"Train batches: {len(loaders['train'])} | Val batches: {len(loaders['val'])}")

    model = STSTransformer(
        vocab_size = len(vocab),
        d_model    = CONFIG["d_model"],
        n_layers   = CONFIG["n_layers"],
        n_heads    = CONFIG["n_heads"],
        d_ff       = CONFIG["d_ff"],
        max_len    = CONFIG["max_len"],
        dropout    = CONFIG["dropout"],
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
   
    best_val_corr = -1.0

    for epoch in range(1, CONFIG["epochs"] + 1):
        train_loss, train_corr = train_epoch(model, loaders["train"], optimizer, criterion)
        val_loss,   val_corr   = evaluate(model, loaders["val"],   criterion)


        print(
            f"Epoch {epoch:02d}/{CONFIG['epochs']}  "
            f"| Train Loss: {train_loss:.4f}  Pearson: {train_corr:.4f}  "
            f"| Val Loss: {val_loss:.4f}  Pearson: {val_corr:.4f}"
        )

        if val_corr > best_val_corr:
            best_val_corr = val_corr
            torch.save(model.state_dict(), CONFIG["save_path"])
            print(f"  Best model saved  (val Pearson = {best_val_corr:.4f})")

    print(f"\nTraining done. Best Val Pearson = {best_val_corr:.4f}")

    if "test" in loaders:
        model.load_state_dict(torch.load(CONFIG["save_path"], map_location=DEVICE))
        test_loss, test_corr = evaluate(model, loaders["test"], criterion)
        print(f"Test Loss: {test_loss:.4f}  |  Test Pearson: {test_corr:.4f}")


if __name__ == "__main__":
    main()
