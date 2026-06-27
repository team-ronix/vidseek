# shared defaults so every training script uses the same model architecture
# and data paths 

MODEL_CONFIG = {
    "d_model":  384,    # embedding dimension
    "n_layers": 6,      # number of encoder layers
    "n_heads":  6,      # number of attention heads (d_model must be divisible by n_heads)
    "d_ff":     1536,   # feed-forward inner dimension (typically 4x d_model)
    "dropout":  0.1,
    "pooling":  "mean", # how to pool token vectors → sentence vector ("mean" or "max")
}

DATA_CONFIG = {
    "train_path": "data/allnli_specter/allnli_specter.csv",
    "val_path":    "data/sts_benchmark/stsb_validation.csv",
    "test_path":   "data/sts_benchmark/stsb_test.csv",
    "vocab_path":  "results/allnli_specter/vocab.pkl",
    "max_len":     128,   # max tokens per sentence (longer sequences are truncated)
    "min_freq":    2,     # words appearing fewer times than this are mapped to <unk>
    "batch_size":  128,
    "num_workers": 0,
    "pin_memory":  True,
}

# each training script can override these if needed
TRAIN_DEFAULTS = {
    "warmup_steps":  100,
    "weight_decay":  0.01,
    "clip_grad":     1.0,
    "pos_threshold": 0.3,   # min score to count a pair as "positive" during eval
}
