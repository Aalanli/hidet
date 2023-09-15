import torch
from torch import nn


class LlamaMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ):
        super().__init__()
        self.w1 = torch.empty(hidden_size, intermediate_size * 2)
        self.w2 = torch.empty(intermediate_size, hidden_size)

        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )

    def build_fused_matmul(self):
        pass

    def build_reduce(self):
        pass

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x
