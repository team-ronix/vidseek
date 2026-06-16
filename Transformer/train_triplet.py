import torch
import torch.nn as nn
import pandas as pd
import pickle
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from models.model.transformer import Transformer
from losses.triplet_loss import TripletLoss
from data import _make_ids, collate_triplet_fn


class HardNegativeDataset(Dataset):
    """Loads (anchor, positive, hard_negative) triplets from a CSV produced
    by evaluate_search.py's mine_hard_negatives()."""

    def __init__(self, df: pd.DataFrame, vocab, max_len: int = 128):
        self.triplets = []
        for _, row in df.iterrows():
            anc = str(row["anchor"])
            pos = str(row["positive"])
            neg = str(row["hard_negative"])
            self.triplets.append((
                _make_ids(anc, vocab, max_len),
                _make_ids(pos, vocab, max_len),
                _make_ids(neg, vocab, max_len),
            ))

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        ids_a, ids_p, ids_n = self.triplets[idx]
        return (
            torch.tensor(ids_a, dtype=torch.long),
            torch.tensor(ids_p, dtype=torch.long),
            torch.tensor(ids_n, dtype=torch.long),
        )


def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {DEVICE}")

    # Load vocab and model saved by train.py
    with open("vocab.pkl", "rb") as _vf:
        vocab = pickle.load(_vf)

    model = Transformer(
        vocab_size=len(vocab),
        d_model=256,
        n_layers=4,
        n_heads=4,
        d_ff=512,
        max_len=128,
        pooling="mean",
    ).to(DEVICE)
    model.load_state_dict(torch.load("best_model.pt", map_location=DEVICE))

    # Load the hard-negative CSV produced by evaluate_search.py
    hard_df = pd.read_csv("data/train_hard_negatives.csv")
    print(f"Hard-negative triplets : {len(hard_df):,}")

    dataset = HardNegativeDataset(hard_df, vocab, max_len=128)
    loader  = DataLoader(
        dataset, batch_size=64, shuffle=True, collate_fn=collate_triplet_fn
    )

    criterion = TripletLoss(margin=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # Fine-tune on hard negatives with TripletLoss
    # Few epochs suffice because the model is already pre-trained
    print("Fine-tuning on hard negatives ...")
    for epoch in range(1, 6):
        model.train()
        total_loss = 0.0
        for ids_a, ids_p, ids_n, mask_a, mask_p, mask_n in tqdm(loader, desc=f"Epoch {epoch}"):
            optimizer.zero_grad()
            emb_a = model.encode(ids_a.to(DEVICE), mask_a.to(DEVICE), normalize=True)
            emb_p = model.encode(ids_p.to(DEVICE), mask_p.to(DEVICE), normalize=True)
            emb_n = model.encode(ids_n.to(DEVICE), mask_n.to(DEVICE), normalize=True)
            loss  = criterion(emb_a, emb_p, emb_n)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg = total_loss / max(len(loader), 1)
        print(f"Epoch {epoch}/5 | Triplet Loss: {avg:.4f}")

    torch.save(model.state_dict(), "final_hard_tuned_model.pt")
    print("Saved final_hard_tuned_model.pt")


if __name__ == "__main__":
    main()
