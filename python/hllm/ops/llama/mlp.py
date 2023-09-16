from typing import List
import hidet
import torch
from hidet.ir.expr import cast
from hidet.ir.dtypes import f16
from hidet.ir.module import IRModule
from hidet.ir.primitives.runtime import request_cuda_workspace
from hllm.ops.base import Operator
from hidet.ir.library import tune


class LlamaMLPOperator(Operator):
    def __init__(self, seq: int, hidden_size: int, intermediate_size: int):
        super().__init__({
            'seq': seq,
            'hidden_size': hidden_size,
            'intermediate_size': intermediate_size,
        })
        self.seq: int = seq
        self.hidden_size: int = hidden_size
        self.intermediate_size: int = intermediate_size

    def dummy_params(self) -> List[torch.Tensor]:
        return [
            torch.empty(self.seq, self.hidden_size, dtype=torch.float16, device='cuda'),
            torch.empty(self.hidden_size, self.intermediate_size * 2, dtype=torch.float16, device='cuda'),
            torch.empty(self.intermediate_size, self.hidden_size, dtype=torch.float16, device='cuda'),
            torch.empty(self.seq, self.hidden_size, dtype=torch.float16, device='cuda'),
        ]

    def implement(self) -> List[IRModule]:
        return tune.extract_ir_modules(self.schedule)

    @tune.space(
        2,
        block_s=[16],
        block_m=[32, 64, 128],
        block_h=[32, 64, 128],
    )
    @tune.space(
        0,
        block_s=[16],
        block_m=[32],
        block_h=[32],
    )
    def schedule(self, block_s: int, block_m: int, block_h: int) -> IRModule:
        from hidet.lang import attrs
        from hidet.lang import tile as ti

        m_size = self.intermediate_size
        h_size = self.hidden_size

        reduce_block_h = 32
        reduce_block_s = block_s

        tune.check(self.seq <= block_s)
        tune.check(m_size % block_m == 0)
        tune.check(h_size % block_h == 0)
        tune.check(h_size % reduce_block_h == 0)

        with hidet.script_module() as script_module:
            @hidet.script
            def llama_ffn(x_ptr: ~f16, w1_ptr: ~f16, w2_ptr: ~f16, y_ptr: ~f16):
                attrs.func_kind = 'cuda_tile'
                attrs.cuda.block_dim = 256
                attrs.cuda.grid_dim = m_size // block_m

                pid = ti.program_id()

                x_ptrs = x_ptr + ti.grid(shape=[block_s, block_h], starts=[0, 0], strides=[h_size, 1])
                w1_ptrs = w1_ptr + ti.grid(shape=[block_h, block_m], starts=[0, pid * block_m], strides=[2 * m_size, 1])
                y1_lhs = ti.zeros([block_s, block_m], dtype=f16)
                y1_rhs = ti.zeros([block_s, block_m], dtype=f16)

                for k in range(h_size // block_h):
                    x = ti.load(x_ptrs)  # [block_s, block_h]
                    w1_lhs = ti.load(w1_ptrs)  # [block_h, block_m]
                    w1_rhs = ti.load(w1_ptrs + m_size)
                    y1_lhs += ti.dot(x, w1_lhs)
                    y1_rhs += ti.dot(x, w1_rhs)
                    x_ptrs += block_h
                    w1_ptrs += 2 * m_size * block_h

                y1 = ti.silu(y1_lhs) * y1_rhs  # [block_s, block_m]

                w2_ptrs = w2_ptr + ti.grid(shape=[block_m, block_h], starts=[pid * block_m, 0], strides=[h_size, 1])
                y_ptrs = y_ptr + ti.grid(shape=[block_s, block_h], starts=[pid * self.seq, 0], strides=[h_size, 1])

                mask = ti.grid(shape=[block_s, 1], starts=[0, 0], strides=[1, 0]) < self.seq

                for k in range(h_size // block_h):
                    w2 = ti.load(w2_ptrs)  # [block_m, block_h]
                    y = ti.dot(y1, w2)  # [block_s, block_h]
                    ti.store(ptr=y_ptrs, value=y, mask=mask)
                    w2_ptrs += block_h
                    y_ptrs += block_h

            @hidet.script
            def reduce(x_ptr: ~f16, y_ptr: ~f16):
                attrs.func_kind = 'cuda_tile'
                attrs.cuda.block_dim = 256
                attrs.cuda.grid_dim = h_size // reduce_block_h

                pid = ti.program_id()
                h_offsets = pid * reduce_block_h + ti.arange(0, reduce_block_h)

                x_ptrs = (
                    x_ptr
                    + ti.expand_dims(ti.arange(0, reduce_block_s), axis=1) * h_size
                    + h_offsets
                )
                acc = ti.zeros([reduce_block_s, reduce_block_h], dtype=f16)
                for k in range((m_size // block_m) * self.seq // reduce_block_s):
                    mask = (
                        ti.expand_dims(
                            ti.arange(0, reduce_block_s), axis=1
                        ) < m_size // block_m * self.seq - k * reduce_block_s
                    )
                    acc += ti.load(x_ptrs, mask=mask)
                    x_ptrs += reduce_block_s * h_size

                ti.store(
                    ptr=y_ptr + ti.expand_dims(ti.arange(0, reduce_block_s), axis=1) * h_size + h_offsets,
                    value=acc,
                    mask=ti.expand_dims(ti.arange(0, reduce_block_s), axis=1) < self.seq
                )

            @hidet.script
            def launch(x_ptr: ~f16, w1_ptr: ~f16, w2_ptr: ~f16, y_ptr: ~f16):
                attrs.func_kind = 'public'

                y1_ptr = cast(request_cuda_workspace(nbytes=(m_size // block_m) * self.seq * h_size * f16.nbytes), ~f16)

                llama_ffn(x_ptr, w1_ptr, w2_ptr, y1_ptr)
                reduce(y1_ptr, y_ptr)

        return script_module.ir_module()


def demo():
    hidet.option.search_space(0)
    hidet.option.save_lower_ir()
    hidet.option.cache_dir('./outs/cache')
    hidet.utils.clear_cache_dir('./hops')

    op = LlamaMLPOperator(16, 4096, 12288)
    dummy_params = op.dummy_params()
    op(*dummy_params)


if __name__ == '__main__':
    demo()
