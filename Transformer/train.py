"""3-phase training: MNR pretraining -> eval -> Triplet fine-tuning"""

import json
import os
import pickle

import torch

from data import build_vocab_from_allnli, get_allnli_pair_loader, get_allnli_triplet_loader, get_sts_loaders
from evaluate_search import evaluate
from losses.mnr_loss import MultipleNegativesRankingLoss
from losses.triplet_loss import TripletLoss
from models.model.transformer import Transformer
from trainer import Trainer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = {
    # data paths
    "allnli_path": "data/AllNLI/AllNLI.csv",
    "val_path":    "data/sts-222/stsb_validation.csv",
    "test_path":   "data/sts-222/stsb_test.csv",
    "vocab_path":  "vocab.pkl",
    "max_len":     128,
    "min_freq":    2,
    "num_workers": 0,
    "pin_memory":  True,
    "batch_size":  128,

    # model
    "d_model":  384,
    "n_layers": 6,
    "n_heads":  6,
    "d_ff":     1536,
    "dropout":  0.1,
    "pooling":  "mean",

    # phase 1 - MNR
    "mnr_epochs":       20,
    "mnr_peak_lr":      3e-4,
    "mnr_warmup_steps": 100,
    "mnr_weight_decay": 0.01,
    "mnr_clip_grad":    1.0,
    "mnr_temperature":  0.05,

    # phase 2 - eval
    "pos_threshold":     0.3,
    "eval_results_path": "eval_results.json",

    # phase 3 - triplet fine-tuning
    "triplet_epochs":    5,
    "triplet_lr":        5e-5,
    "triplet_margin":    0.5,
    "triplet_clip_grad": 1.0,

    # checkpoints
    "best_model_path":  "best_model.pt",
    "final_model_path": "final_triplet_model.pt",
    "checkpoint_path":  "checkpoints/checkpoint_latest.pt",
}


# phase 1: train with MNR loss on AllNLI pairs
def phase1_mnr(model, vocab, val_loader, cfg, device):
    print("\n" + "=" * 60)
    print("Phase 1 -- MNR Training on AllNLI")
    print("=" * 60)

    pin = cfg["pin_memory"] and torch.cuda.is_available()
    train_loader = get_allnli_pair_loader(
        path=cfg["allnli_path"], vocab=vocab,
        batch_size=cfg["batch_size"], max_len=cfg["max_len"],
        num_workers=cfg["num_workers"], pin_memory=pin,
    )
    print(f"Training pairs: {len(train_loader.dataset):,}")  # type: ignore

    trainer_cfg = {
        "epochs":          cfg["mnr_epochs"],
        "peak_lr":         cfg["mnr_peak_lr"],
        "warmup_steps":    cfg["mnr_warmup_steps"],
        "weight_decay":    cfg["mnr_weight_decay"],
        "clip_grad":       cfg["mnr_clip_grad"],
        "checkpoint_path": cfg["checkpoint_path"],
        "best_model_path": cfg["best_model_path"],
        "loader_type":     "pair",
    }
    criterion = MultipleNegativesRankingLoss(cfg["mnr_temperature"])
    trainer = Trainer(model, criterion, train_loader, val_loader, device, trainer_cfg)
    best_rho = trainer.fit()
    print(f"\nPhase 1 done.  Best val Spearman = {best_rho:.4f}")


# phase 2: evaluate best MNR checkpoint on sts-222
def phase2_evaluate(model, vocab, cfg, device):
    print("\n" + "=" * 60)
    print("Phase 2 -- Evaluation on sts-222")
    print("=" * 60)

    model.load_state_dict(torch.load(cfg["best_model_path"], map_location=device))

    results = {}
    for split, path in [("val", cfg["val_path"]), ("test", cfg["test_path"])]:
        if not os.path.exists(path):
            print(f"  [SKIP] {path} not found")
            continue
        print(f"\n  {split.upper()} -- {path}")
        metrics = evaluate(
            model=model, csv_path=path, vocab=vocab,
            max_len=cfg["max_len"], device=device,
            pos_threshold=cfg["pos_threshold"],
        )
        results[split] = metrics
        print(f"  Recall@1={metrics['recall@1']:.4f}  Recall@5={metrics['recall@5']:.4f}  "
              f"Recall@10={metrics['recall@10']:.4f}  MRR={metrics['mrr']:.4f}  "
              f"Spearman={metrics['spearman']:.4f}")

    with open(cfg["eval_results_path"], "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {cfg['eval_results_path']}")


# phase 3: fine-tune on AllNLI triplets
def phase3_triplet(model, vocab, val_loader, cfg, device):
    print("\n" + "=" * 60)
    print("Phase 3 -- Triplet Fine-tuning on AllNLI")
    print("=" * 60)

    pin = cfg["pin_memory"] and torch.cuda.is_available()
    train_loader = get_allnli_triplet_loader(
        path=cfg["allnli_path"], vocab=vocab,
        batch_size=cfg["batch_size"], max_len=cfg["max_len"],
        num_workers=cfg["num_workers"], pin_memory=pin,
    )
    print(f"Training triplets: {len(train_loader.dataset):,}")  # type: ignore

    trainer_cfg = {
        "epochs":          cfg["triplet_epochs"],
        "peak_lr":         cfg["triplet_lr"],
        "warmup_steps":    100,
        "weight_decay":    cfg["mnr_weight_decay"],
        "clip_grad":       cfg["triplet_clip_grad"],
        "checkpoint_path": "checkpoints/triplet_finetune_ckpt.pt",
        "best_model_path": cfg["final_model_path"],
        "loader_type":     "triplet",
    }
    criterion = TripletLoss(margin=cfg["triplet_margin"])
    trainer = Trainer(model, criterion, train_loader, val_loader, device, trainer_cfg)
    trainer.fit()
    print(f"\nPhase 3 done.  Saved {cfg['final_model_path']}")


def main():
    os.makedirs(os.path.dirname(CONFIG["checkpoint_path"]) or ".", exist_ok=True)
    pin = CONFIG["pin_memory"] and torch.cuda.is_available()

    print(f"Device: {DEVICE}")

    print("\nBuilding vocabulary from AllNLI ...")
    vocab = build_vocab_from_allnli(CONFIG["allnli_path"], min_freq=CONFIG["min_freq"])
    print(f"Vocab size: {len(vocab):,}")
    with open(CONFIG["vocab_path"], "wb") as f:
        pickle.dump(vocab, f)

    sts_loaders = get_sts_loaders(
        paths={"val": CONFIG["val_path"], "test": CONFIG["test_path"]},
        vocab=vocab, batch_size=CONFIG["batch_size"], max_len=CONFIG["max_len"],
        num_workers=CONFIG["num_workers"], pin_memory=pin,
    )

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
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")

    phase1_mnr(model, vocab, sts_loaders["val"], CONFIG, DEVICE)
    phase2_evaluate(model, vocab, CONFIG, DEVICE)
    phase3_triplet(model, vocab, sts_loaders["val"], CONFIG, DEVICE)


if __name__ == "__main__":
    main()
