from typing import List, Tuple, Optional, Type
import torch
from torch import nn
from vllm.model_executor.input_metadata import InputMetadata
from vllm.sequence import SamplerOutput


class VllmForCausalLM(nn.Module):
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
    ) -> SamplerOutput:
        raise NotImplementedError()

    def load_weights(
        self,
        model_name_or_path: str,
        cache_dir: Optional[str] = None,
        load_format: str = "auto"
    ):
        raise NotImplementedError()


def register_vllm_model(name: str, module: Optional[Type[nn.Module]] = None):
    from vllm.model_executor.model_loader import _MODEL_REGISTRY

    if module is not None:
        _MODEL_REGISTRY[name] = module
    else:
        def decorator(cls):
            _MODEL_REGISTRY[name] = cls
            return cls

        return decorator


def register_hidet_implementations():
    from .llama import LlamaForCausalLM
    from .opt import OPTForCausalLM
    name2model = {
        'LlamaForCausalLM': LlamaForCausalLM,
        'OPTForCausalLM': OPTForCausalLM
    }
    for name, model in name2model.items():
        register_vllm_model(name, model)


def revert_implementations():
    from vllm.model_executor.models import LlamaForCausalLM, OPTForCausalLM
    name2model = {
        'LlamaForCausalLM': LlamaForCausalLM,
        'OPTForCausalLM': OPTForCausalLM
    }
    for name, model in name2model.items():
        register_vllm_model(name, model)
