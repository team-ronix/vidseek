# entry point — configure dataset paths, model hyperparams, and training schedule,
# then run the MNR training loop and evaluate

import json
import os
import pickle

import torch
import torch.nn as nn
from tqdm import tqdm

from configs.default import MODEL_CONFIG, DATA_CONFIG, TRAIN_DEFAULTS
from data import build_vocab, get_pair_loader, get_sts_loaders
from evaluate_search import evaluate
from losses.mnr_loss import MultipleNegativesRankingLoss
from models import Transformer
from utils import make_cosine_with_warmup, save_checkpoint, load_checkpoint, spearman_on_sts

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = {
    # model architecture - from configs/default.py
    **MODEL_CONFIG,

    # data paths and loading - from configs/default.py
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


def run_one_batch(model, batch, criterion, device):
    # unpack (anchor, positive) pair batch, encode both sides, compute MNR loss
    ids_a, ids_b, mask_a, mask_b = [x.to(device) for x in batch]
    emb_a = model.encode(ids_a, mask_a, normalize=True)
    emb_b = model.encode(ids_b, mask_b, normalize=True)
    return criterion(emb_a, emb_b)


class Trainer:
    def __init__(self, model, criterion, train_loader, val_loader, device, cfg):
        self.model        = model
        self.criterion    = criterion
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device
        self.cfg          = cfg

    def build_optimizer(self):
        # apply weight decay only to weight matrices, not biases or layer norm params
        decay, no_decay = [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if "bias" in name or "norm" in name:
                no_decay.append(p)
            else:
                decay.append(p)
        return torch.optim.AdamW(
            [{"params": decay,    "weight_decay": self.cfg["weight_decay"]},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=self.cfg["peak_lr"], betas=(0.9, 0.999), eps=1e-8,
        )

    def fit(self):
        cfg          = self.cfg
        total_steps  = cfg["epochs"] * len(self.train_loader)
        warmup_steps = min(cfg.get("warmup_steps", 100), total_steps // 10)

        optimizer = self.build_optimizer()
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=make_cosine_with_warmup(total_steps, warmup_steps)
        )

        # try to resume from a saved checkpoint
        start_epoch, global_step, best_rho = load_checkpoint(
            cfg["checkpoint_path"], self.model, optimizer, scheduler, cfg["epochs"]
        )

        for epoch in range(start_epoch + 1, cfg["epochs"] + 1):
            self.model.train()
            total_loss = 0.0

            for batch in tqdm(self.train_loader, desc=f"  Epoch {epoch:02d}/{cfg['epochs']}", leave=False):
                optimizer.zero_grad()
                loss = run_one_batch(self.model, batch, self.criterion, self.device)
                loss.backward()
                # clip gradients to prevent exploding gradients
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg["clip_grad"])
                optimizer.step()
                scheduler.step()
                global_step += 1
                total_loss  += loss.item()

            avg_loss = total_loss / len(self.train_loader)
            val_rho  = spearman_on_sts(self.model, self.val_loader, self.device)
            lr_now   = scheduler.get_last_lr()[0]

            tag = ""
            if val_rho > best_rho:
                best_rho = val_rho
                torch.save(self.model.state_dict(), cfg["best_model_path"])
                tag = " *"

            # save checkpoint after updating best_rho so resume starts with the correct value
            save_checkpoint(cfg["checkpoint_path"], self.model, optimizer, scheduler,
                            epoch, global_step, best_rho)
            print(f"  Epoch {epoch:02d}/{cfg['epochs']}  lr={lr_now:.2e}  loss={avg_loss:.4f}  val_rho={val_rho:.4f}{tag}")

        return best_rho


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
