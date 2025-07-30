import torch

LLAMA2_CONFIG_7B = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 4096,  # Context length
    "emb_dim": 4096,         # Embedding dimension
    "n_heads": 1,           # Number of attention heads
    "n_layers": 6,          # Number of layers
    "hidden_dim": 118,     # NEW: Size of the intermediate dimension in FeedForward
    "dtype": torch.bfloat16  # NEW: Lower-precision dtype to reduce memory usage
}