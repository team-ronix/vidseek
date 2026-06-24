"""Compare all 5 loss functions on the same transformer + same data.

Run all:    python train_compare.py
Run subset: python train_compare.py --only mnr cosent
"""

import argparse
import json
import os
import pickle

import torch

from data import (
    build_vocab_from_allnli,
    get_allnli_labeled_pair_loader,
    get_allnli_pair_loader,
    get_allnli_triplet_loader,
    get_sts_loaders,
)
from evaluate_search import evaluate
from losses import ContrastiveLoss, CoSENTLoss, CosineMSELoss, MultipleNegativesRankingLoss, TripletLoss
from models.model.transformer import Transformer
from trainer import Trainer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# shared settings for all approaches (same model, same data, same paths)
SHARED = {
    "allnli_path": "data/AllNLI/AllNLI.csv",
    "val_path":    "data/sts-222/stsb_validation.csv",
    "test_path":   "data/sts-222/stsb_test.csv",
    "vocab_path":  "vocab.pkl",
    "out_dir":     "compare_runs",
    "max_len":     128,
    "min_freq":    2,
    "num_workers": 0,
    "pin_memory":  True,
    "warmup_steps": 100,
    "weight_decay": 0.01,
    "clip_grad":    1.0,
    "pos_threshold": 0.3,
    # model architecture
    "d_model":  384,
    "n_layers": 6,
    "n_heads":  6,
    "d_ff":     1536,
    "dropout":  0.1,
    "pooling":  "mean",
}

# per-approach hyperparams
APPROACHES = {
    "mnr": {
        "loader_type": "pair",
        "loss_cls":    MultipleNegativesRankingLoss,
        "loss_kwargs": {"temperature": 0.05},
        "epochs":      20,
        "peak_lr":     3e-4,
        "batch_size":  128,
    },
    "triplet": {
        "loader_type": "triplet",
        "loss_cls":    TripletLoss,
        "loss_kwargs": {"margin": 0.5},
        "epochs":      20,
        "peak_lr":     1e-4,
        "batch_size":  64,
    },
    "contrastive": {
        "loader_type": "labeled",
        "loss_cls":    ContrastiveLoss,
        "loss_kwargs": {"margin": 0.5},
        "epochs":      20,
        "peak_lr":     1e-4,
        "batch_size":  128,
    },
    "cosent": {
        "loader_type": "labeled",
        "loss_cls":    CoSENTLoss,
        "loss_kwargs": {"scale": 20.0},
        "epochs":      20,
        "peak_lr":     2e-5,
        "batch_size":  128,
    },
    "cosine_mse": {
        "loader_type": "labeled",
        "loss_cls":    CosineMSELoss,
        "loss_kwargs": {},
        "epochs":      20,
        "peak_lr":     1e-4,
        "batch_size":  128,
    },
}


def build_model(vocab_size):
    return Transformer(
        vocab_size=vocab_size,
        d_model=SHARED["d_model"],
        n_layers=SHARED["n_layers"],
        n_heads=SHARED["n_heads"],
        d_ff=SHARED["d_ff"],
        max_len=SHARED["max_len"],
        dropout=SHARED["dropout"],
        pooling=SHARED["pooling"],
    ).to(DEVICE)


def build_loader(loader_type, vocab, batch_size):
    pin = SHARED["pin_memory"] and torch.cuda.is_available()
    kw = dict(
        path=SHARED["allnli_path"], vocab=vocab,
        batch_size=batch_size, max_len=SHARED["max_len"],
        num_workers=SHARED["num_workers"], pin_memory=pin,
    )
    if loader_type == "pair":
        return get_allnli_pair_loader(**kw)
    if loader_type == "triplet":
        return get_allnli_triplet_loader(**kw)
    return get_allnli_labeled_pair_loader(**kw)


def run_approach(name, vocab, sts_loaders):
    ap = APPROACHES[name]
    run_dir = os.path.join(SHARED["out_dir"], name)
    os.makedirs(run_dir, exist_ok=True)
    best_path = os.path.join(run_dir, "best_model.pt")
    ckpt_path = os.path.join(run_dir, "checkpoint.pt")
    eval_path = os.path.join(run_dir, "eval_results.json")

    print(f"\n{'='*62}")
    print(f"  [{name.upper()}]  lr={ap['peak_lr']:.0e}  bs={ap['batch_size']}  epochs={ap['epochs']}")
    print(f"{'='*62}")

    # skip if already finished
    if os.path.exists(eval_path):
        print(f"  Already done -- loading {eval_path}")
        with open(eval_path) as f:
            return json.load(f)

    model = build_model(len(vocab))
    criterion = ap["loss_cls"](**ap["loss_kwargs"])
    train_loader = build_loader(ap["loader_type"], vocab, ap["batch_size"])

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {n_params:,}  |  Training samples: {len(train_loader.dataset):,}")  # type: ignore

    trainer_cfg = {
        "epochs":          ap["epochs"],
        "peak_lr":         ap["peak_lr"],
        "warmup_steps":    SHARED["warmup_steps"],
        "weight_decay":    SHARED["weight_decay"],
        "clip_grad":       SHARED["clip_grad"],
        "checkpoint_path": ckpt_path,
        "best_model_path": best_path,
        "loader_type":     ap["loader_type"],
    }
    trainer = Trainer(model, criterion, train_loader, sts_loaders["val"], DEVICE, trainer_cfg)
    best_rho = trainer.fit()

    print(f"\n  Best val Spearman = {best_rho:.4f}")

    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    results = {"best_val_rho": float(best_rho)}

    for split, path in [("val", SHARED["val_path"]), ("test", SHARED["test_path"])]:
        if not os.path.exists(path):
            continue
        metrics = evaluate(
            model=model, csv_path=path, vocab=vocab,
            max_len=SHARED["max_len"], device=DEVICE,
            pos_threshold=SHARED["pos_threshold"],
        )
        results[split] = metrics
        print(f"  {split.upper()}: R@1={metrics['recall@1']:.4f}  R@5={metrics['recall@5']:.4f}  "
              f"R@10={metrics['recall@10']:.4f}  MRR={metrics['mrr']:.4f}  Spearman={metrics['spearman']:.4f}")

    with open(eval_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", choices=list(APPROACHES), default=list(APPROACHES),
                        metavar="APPROACH", help="run only these approaches (default: all)")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    os.makedirs(SHARED["out_dir"], exist_ok=True)

    if os.path.exists(SHARED["vocab_path"]):
        print(f"\nLoading vocabulary from {SHARED['vocab_path']} ...")
        with open(SHARED["vocab_path"], "rb") as f:
            vocab = pickle.load(f)
    else:
        print("\nBuilding vocabulary from AllNLI ...")
        vocab = build_vocab_from_allnli(SHARED["allnli_path"], min_freq=SHARED["min_freq"])
        with open(SHARED["vocab_path"], "wb") as f:
            pickle.dump(vocab, f)
    print(f"Vocab size: {len(vocab):,}")

    pin = SHARED["pin_memory"] and torch.cuda.is_available()
    sts_loaders = get_sts_loaders(
        paths={"val": SHARED["val_path"], "test": SHARED["test_path"]},
        vocab=vocab, batch_size=128, max_len=SHARED["max_len"],
        num_workers=SHARED["num_workers"], pin_memory=pin,
    )

    all_results = {}
    for name in args.only:
        all_results[name] = run_approach(name, vocab, sts_loaders)

    summary_path = os.path.join(SHARED["out_dir"], "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\n{'='*74}")
    print("  COMPARISON -- TEST SET")
    print(f"{'='*74}")
    print(f"  {'Approach':<14}  {'Spearman':>9}  {'R@1':>7}  {'R@5':>7}  {'R@10':>8}  {'MRR':>7}")
    print(f"  {'-'*64}")
    for name, res in all_results.items():
        t = res.get("test", {})
        print(f"  {name:<14}  "
              f"{t.get('spearman', 0):>9.4f}  "
              f"{t.get('recall@1', 0):>7.4f}  "
              f"{t.get('recall@5', 0):>7.4f}  "
              f"{t.get('recall@10', 0):>8.4f}  "
              f"{t.get('mrr', 0):>7.4f}")
    print(f"{'='*74}")
    print(f"\nPer-run results: {SHARED['out_dir']}/<approach>/eval_results.json")
    print(f"Summary:         {summary_path}")


if __name__ == "__main__":
    main()
