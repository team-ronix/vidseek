# generic training loop that works with any of our loss functions
# just set loader_type to "pair", "triplet", or "labeled" and it handles the rest

import torch
import torch.nn as nn
from tqdm import tqdm

from utils import make_cosine_with_warmup, save_checkpoint, load_checkpoint, spearman_on_sts


def run_one_batch(loader_type, model, batch, criterion, device):
    if loader_type == "pair":
        ids_a, ids_b, mask_a, mask_b = [x.to(device) for x in batch]
        emb_a = model.encode(ids_a, mask_a, normalize=True)
        emb_b = model.encode(ids_b, mask_b, normalize=True)
        return criterion(emb_a, emb_b)

    if loader_type == "triplet":
        ids_a, ids_p, ids_n, mask_a, mask_p, mask_n = [x.to(device) for x in batch]
        emb_a = model.encode(ids_a, mask_a, normalize=True)
        emb_p = model.encode(ids_p, mask_p, normalize=True)
        emb_n = model.encode(ids_n, mask_n, normalize=True)
        return criterion(emb_a, emb_p, emb_n)

    # labeled pairs (contrastive, cosent, cosine_mse)
    ids_a, ids_b, mask_a, mask_b, labels = [x.to(device) for x in batch]
    emb_a = model.encode(ids_a, mask_a, normalize=True)
    emb_b = model.encode(ids_b, mask_b, normalize=True)
    return criterion(emb_a, emb_b, labels)


class Trainer:
    def __init__(self, model, criterion, train_loader, val_loader, device, cfg,
                 compute_loss_fn=None):
        self.model = model
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.cfg = cfg
        self.compute_loss_fn = compute_loss_fn  # optional override for custom losses

    def get_loss(self, batch):
        if self.compute_loss_fn is not None:
            return self.compute_loss_fn(self.model, batch, self.criterion, self.device)
        return run_one_batch(self.cfg["loader_type"], self.model, batch, self.criterion, self.device)

    def build_optimizer(self):
        # don't apply weight decay to biases and layer norm params
        decay = []
        no_decay = []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if "bias" in name or "norm" in name:
                no_decay.append(p)
            else:
                decay.append(p)
        return torch.optim.AdamW(
            [{"params": decay, "weight_decay": self.cfg["weight_decay"]},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=self.cfg["peak_lr"], betas=(0.9, 0.999), eps=1e-8,
        )

    def fit(self):
        cfg = self.cfg
        total_steps = cfg["epochs"] * len(self.train_loader)
        warmup_steps = min(cfg.get("warmup_steps", 100), total_steps // 10)

        optimizer = self.build_optimizer()
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=make_cosine_with_warmup(total_steps, warmup_steps)
        )

        start_epoch, global_step, best_rho = load_checkpoint(
            cfg["checkpoint_path"], self.model, optimizer, scheduler, cfg["epochs"]
        )

        for epoch in range(start_epoch + 1, cfg["epochs"] + 1):
            self.model.train()
            total_loss = 0.0

            for batch in tqdm(self.train_loader, desc=f"  Epoch {epoch:02d}/{cfg['epochs']}", leave=False):
                optimizer.zero_grad()
                loss = self.get_loss(batch)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg["clip_grad"])
                optimizer.step()
                scheduler.step()
                global_step += 1
                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_loader)
            val_rho = spearman_on_sts(self.model, self.val_loader, self.device)
            lr_now = scheduler.get_last_lr()[0]

            save_checkpoint(cfg["checkpoint_path"], self.model, optimizer, scheduler,
                            epoch, global_step, best_rho)

            tag = ""
            if val_rho > best_rho:
                best_rho = val_rho
                torch.save(self.model.state_dict(), cfg["best_model_path"])
                tag = " *"
            print(f"  Epoch {epoch:02d}/{cfg['epochs']}  lr={lr_now:.2e}  loss={avg_loss:.4f}  val_rho={val_rho:.4f}{tag}")

        return best_rho
