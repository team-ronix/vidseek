import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model.encoder import Encoder


class STSTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_layers=4, n_heads=8,
                 d_ff=512, max_len=512, dropout=0.1):
        super(STSTransformer, self).__init__()

        self.encoder = Encoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            max_len=max_len,
            dropout=dropout,
        )

    def mean_pooling(self, token_embeddings, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        summed = (token_embeddings * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def encode_sentence(self, input_ids, attention_mask=None):
        if attention_mask is None:
            attention_mask = (input_ids != 0).long()

        src_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        hidden = self.encoder(input_ids, src_mask)
        return self.mean_pooling(hidden, attention_mask)

    def forward(self, input_ids_a, input_ids_b,
                mask_a=None, mask_b=None):
        emb_a = self.encode_sentence(input_ids_a, mask_a)
        emb_b = self.encode_sentence(input_ids_b, mask_b)

        similarity = F.cosine_similarity(emb_a, emb_b, dim=-1)
        return similarity
