from dataclasses import dataclass

@dataclass
class DecodingSampleConfig:
    q_seq_len: int # between 1, 32 (inclusive)
    kv_seq_len: int # between 512, 1024 (inclusive)
    batch_size: int # between 1, 32 (inclusive)

@dataclass
class ModelConfig:
    n_layers: int
    n_heads: int
    head_dim: int

@dataclass
class BenchResult:
    model: str
    precision: str
    framework: str

@dataclass(frozen=True, eq=True)
class BenchResult:
    model: str
    framework: str
    precision: str
    median: float
    low: float
    high: float