# entry point — configure dataset paths, model hyperparams, and training schedule
# calls train.py to run the actual training loop

import json
import os
import pickle

import torch

from configs.default import MODEL_CONFIG, DATA_CONFIG, TRAIN_DEFAULTS
from data import build_vocab, get_pair_loader, get_sts_loaders
from evaluate_search import evaluate
from losses.mnr_loss import MultipleNegativesRankingLoss
from models.model.transformer import Transformer
from train import Trainer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = {
    # model architecture — from configs/default.py
    **MODEL_CONFIG,

    # data paths and loading — from configs/default.py
    **DATA_CONFIG,

    # MNR training
    "mnr_epochs":       20,
    "mnr_peak_lr":      3e-4,
    "mnr_warmup_steps": TRAIN_DEFAULTS["warmup_steps"],
    "mnr_weight_decay": TRAIN_DEFAULTS["weight_decay"],
    "mnr_clip_grad":    TRAIN_DEFAULTS["clip_grad"],
    "mnr_temperature":  0.05,

    # evaluation
    "pos_threshold":     TRAIN_DEFAULTS["pos_threshold"],
    "eval_results_path": "eval_results.json",

    # checkpoints and saved model
    "best_model_path": "results/allnli_specter/best_model.pt",
    "checkpoint_path": "results/allnli_specter/checkpoints/checkpoint_latest.pt",
}


def train_mnr(model, vocab, val_loader, cfg, device):
    # train with MNR loss on (anchor, positive) pairs
    print("\n" + "=" * 60)
    print("MNR Training")
    print("=" * 60)

    pin = cfg["pin_memory"] and torch.cuda.is_available()
    train_loader = get_pair_loader(
        path=cfg["train_path"], vocab=vocab,
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
    }
    criterion = MultipleNegativesRankingLoss(cfg["mnr_temperature"])
    trainer   = Trainer(model, criterion, train_loader, val_loader, device, trainer_cfg)
    best_rho  = trainer.fit()
    print(f"\nTraining done.  Best val Spearman = {best_rho:.4f}")


def run_evaluation(model, vocab, cfg, device):
    # load the best checkpoint and evaluate on val + test splits
    print("\n" + "=" * 60)
    print("Evaluation on STS Benchmark")
    print("=" * 60)

    model.load_state_dict(torch.load(cfg["best_model_path"], map_location=device, weights_only=True))

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


def main():
    os.makedirs(os.path.dirname(CONFIG["checkpoint_path"]), exist_ok=True)
    os.makedirs(os.path.dirname(CONFIG["best_model_path"]),  exist_ok=True)
    pin = CONFIG["pin_memory"] and torch.cuda.is_available()

    print(f"Device: {DEVICE}")

    # build vocabulary from all sentences in the training set
    print("\nBuilding vocabulary ...")
    vocab = build_vocab(CONFIG["train_path"], min_freq=CONFIG["min_freq"])
    print(f"Vocab size: {len(vocab):,}")
    with open(CONFIG["vocab_path"], "wb") as f:
        pickle.dump(vocab, f)

    # load STS validation/test sets used for evaluation during training
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

    train_mnr(model, vocab, sts_loaders["val"], CONFIG, DEVICE)
    run_evaluation(model, vocab, CONFIG, DEVICE)


if __name__ == "__main__":
    main()
