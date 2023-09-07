from dataclasses import dataclass

@dataclass
class DecodingSampleConfig:
    q_seq_len: int # between 1, 32 (inclusive)
    kv_seq_len: int # between 512, 1024 (inclusive)
    batch_size: int # between 1, 32 (inclusive)
    repeat: int = 20
    warmup: int = 5

@dataclass
class ModelConfig:
    n_layers: int
    n_heads: int
    head_dim: int

