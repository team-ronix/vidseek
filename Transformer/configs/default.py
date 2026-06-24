# shared defaults so every training script uses the same model architecture
# and data paths without copy-pasting

MODEL_CONFIG = {
    "d_model":  384,
    "n_layers": 6,
    "n_heads":  6,
    "d_ff":     1536,
    "dropout":  0.1,
    "pooling":  "mean",
}

DATA_CONFIG = {
    "allnli_path": "data/AllNLI/AllNLI.csv",
    "val_path":    "data/sts-222/stsb_validation.csv",
    "test_path":   "data/sts-222/stsb_test.csv",
    "vocab_path":  "vocab.pkl",
    "max_len":     128,
    "min_freq":    2,
    "num_workers": 0,
    "pin_memory":  True,
}

# each script can override these if needed
TRAIN_DEFAULTS = {
    "warmup_steps":  100,
    "weight_decay":  0.01,
    "clip_grad":     1.0,
    "pos_threshold": 0.3,  # scores are in [-1, 1] now
}
