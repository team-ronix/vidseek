import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)

        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** (i / d_model)))
                pe[pos, i+1] = math.cos(pos / (10000 ** (i / d_model)))

        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, heads=8):
        super().__init__()
        self.heads = heads
        self.d_k = d_model // heads

        self.qkv = nn.Linear(d_model, d_model * 3)
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.qkv(x).reshape(B, T, 3, self.heads, self.d_k)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        weights = torch.softmax(scores, dim=-1)

        out = (weights @ v).reshape(B, T, C)
        return self.fc(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = MultiHeadAttention(d_model)
        self.ff = FeedForward(d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.ff(x))
        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_layers=4):
        super().__init__()

        self.embed = Embeddings(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model)

        self.layers = nn.ModuleList([
            EncoderLayer(d_model) for _ in range(num_layers)
        ])

    def forward(self, x):
        x = self.embed(x)
        x = self.pos(x)

        for layer in self.layers:
            x = layer(x)

        return x


class SBERT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = Encoder(vocab_size)

    def forward(self, x):
        x = self.encoder(x)
        x = x.mean(dim=1)
        x = F.normalize(x, dim=1)
        return x