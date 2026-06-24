# fine-tunes the MNR baseline on hard-negative triplets with a distillation term
# to avoid forgetting what the baseline already learned

import json
import os
import pickle

import torch
import torch.nn.functional as F

from data import Vocabulary, get_allnli_triplet_loader, get_sts_loaders
from evaluate_search import evaluate
from losses.triplet_loss import TripletLoss
from models.model.transformer import Transformer
from trainer import Trainer
from utils import spearman_on_sts

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    # paths
    "hard_neg_path":     "data/AllNLI/train_hard_negatives.csv",
    "val_path":          "data/sts-222/stsb_validation.csv",
    "test_path":         "data/sts-222/stsb_test.csv",
    "vocab_path":        "vocab.pkl",
    "pretrained_path":   os.path.join(BASE_DIR, "best_model.pt"),
    "output_path":       os.path.join(BASE_DIR, "best_triplet_hard_model.pt"),
    "eval_results_path": os.path.join(BASE_DIR, "triplet_hard_eval_results.json"),
    "checkpoint_path":   os.path.join(BASE_DIR, "checkpoints", "triplet_hard_ckpt.pt"),
    # data
    "max_len":     128,
    "num_workers": 0,
    "pin_memory":  True,
    "batch_size":  32,
    "eval_batch_size": 64,
    # model arch — must match best_model.pt exactly
    "d_model":  256,
    "n_layers": 4,
    "n_heads":  4,
    "d_ff":     512,
    "dropout":  0.1,
    "pooling":  "mean",
    # fine-tuning (conservative to avoid catastrophic forgetting)
    "epochs":         3,
    "peak_lr":        5e-6,
    "weight_decay":   0.01,
    "clip_grad":      1.0,
    "warmup_ratio":   0.1,
    "triplet_margin": 0.4,
    "pos_threshold":  0.3,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    cfg = CONFIG
    pin = cfg["pin_memory"] and torch.cuda.is_available()

    print(f"Device: {DEVICE}")

    print(f"\nLoading vocabulary from {cfg['vocab_path']} ...")
    with open(cfg["vocab_path"], "rb") as f:
        vocab: Vocabulary = pickle.load(f)
    print(f"Vocab size: {len(vocab):,}")

    print(f"\nLoading hard-negative triplets from {cfg['hard_neg_path']} ...")
    if not os.path.exists(cfg["hard_neg_path"]):
        raise FileNotFoundError(cfg["hard_neg_path"])

    train_loader = get_allnli_triplet_loader(
        path=cfg["hard_neg_path"], vocab=vocab,
        batch_size=cfg["batch_size"], max_len=cfg["max_len"],
        num_workers=cfg["num_workers"], pin_memory=pin,
    )
    print(f"Triplets: {len(train_loader.dataset):,}  |  Steps/epoch: {len(train_loader):,}")  # type: ignore

    sts_loaders = get_sts_loaders(
        paths={"val": cfg["val_path"]},
        vocab=vocab, batch_size=cfg["eval_batch_size"],
        max_len=cfg["max_len"], num_workers=cfg["num_workers"], pin_memory=pin,
    )
    val_loader = sts_loaders["val"]

    model = Transformer(
        vocab_size=len(vocab),
        d_model=cfg["d_model"], n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"], d_ff=cfg["d_ff"],
        max_len=cfg["max_len"], dropout=cfg["dropout"],
        pooling=cfg["pooling"],
    ).to(DEVICE)

    print(f"\nLoading pretrained weights from {cfg['pretrained_path']} ...")
    if not os.path.exists(cfg["pretrained_path"]):
        raise FileNotFoundError(cfg["pretrained_path"])
    model.load_state_dict(torch.load(cfg["pretrained_path"], map_location=DEVICE))

    # frozen copy of the baseline — used to compute distillation penalty
    model_ref = Transformer(
        vocab_size=len(vocab),
        d_model=cfg["d_model"], n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"], d_ff=cfg["d_ff"],
        max_len=cfg["max_len"], dropout=cfg["dropout"],
        pooling=cfg["pooling"],
    ).to(DEVICE)
    model_ref.load_state_dict(torch.load(cfg["pretrained_path"], map_location=DEVICE))
    model_ref.eval()
    for p in model_ref.parameters():
        p.requires_grad_(False)

    # save baseline weights first so we always have a usable output even if training doesn't improve
    baseline_rho = spearman_on_sts(model, val_loader, DEVICE)
    print(f"Baseline val Spearman = {baseline_rho:.4f}")
    torch.save(model.state_dict(), cfg["output_path"])

    criterion = TripletLoss(margin=cfg["triplet_margin"])

    # custom loss = triplet + small distillation term to stay close to baseline
    def distill_compute_loss(model, batch, criterion, device):
        ids_a, ids_p, ids_n, mask_a, mask_p, mask_n = [x.to(device) for x in batch]
        emb_a = model.encode(ids_a, mask_a, normalize=True)
        emb_p = model.encode(ids_p, mask_p, normalize=True)
        emb_n = model.encode(ids_n, mask_n, normalize=True)
        with torch.no_grad():
            ref_a = model_ref.encode(ids_a, mask_a, normalize=True)
        distill = (1.0 - F.cosine_similarity(emb_a, ref_a, dim=-1)).mean()
        return criterion(emb_a, emb_p, emb_n) + 0.1 * distill

    total_steps = cfg["epochs"] * len(train_loader)
    warmup_steps = int(total_steps * cfg["warmup_ratio"])

    trainer_cfg = {
        "epochs":          cfg["epochs"],
        "peak_lr":         cfg["peak_lr"],
        "warmup_steps":    warmup_steps,
        "weight_decay":    cfg["weight_decay"],
        "clip_grad":       cfg["clip_grad"],
        "checkpoint_path": cfg["checkpoint_path"],
        "best_model_path": cfg["output_path"],
        "loader_type":     "triplet",
    }

    print(f"\n{'='*60}")
    print(f"Triplet Fine-Tuning  |  epochs={cfg['epochs']}  margin={cfg['triplet_margin']}")
    print(f"lr={cfg['peak_lr']:.0e}  warmup={warmup_steps}/{total_steps} steps")
    print("=" * 60)

    trainer = Trainer(model, criterion, train_loader, val_loader, DEVICE,
                      trainer_cfg, compute_loss_fn=distill_compute_loss)
    best_rho = trainer.fit()

    delta = best_rho - baseline_rho
    print(f"\nDone.  Best val Spearman = {best_rho:.4f}  ({'+' if delta >= 0 else ''}{delta:.4f} vs baseline)")

    print("\nLoading best checkpoint for final evaluation ...")
    model.load_state_dict(torch.load(cfg["output_path"], map_location=DEVICE))

    results = {}
    for split, path in [("val", cfg["val_path"]), ("test", cfg["test_path"])]:
        if not os.path.exists(path):
            print(f"  [SKIP] {path} not found")
            continue
        print(f"\n  {split.upper()} -- {path}")
        metrics = evaluate(
            model=model, csv_path=path, vocab=vocab,
            max_len=cfg["max_len"], device=DEVICE,
            pos_threshold=cfg["pos_threshold"],
            batch_size=cfg["eval_batch_size"],
        )
        results[split] = metrics
        print(f"  Recall@1={metrics['recall@1']:.4f}  Recall@5={metrics['recall@5']:.4f}  "
              f"Recall@10={metrics['recall@10']:.4f}  MRR={metrics['mrr']:.4f}  "
              f"Spearman={metrics['spearman']:.4f}")

    with open(cfg["eval_results_path"], "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved -> {cfg['eval_results_path']}")


if __name__ == "__main__":
    main()
