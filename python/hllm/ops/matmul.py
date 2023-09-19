# from typing import List
# import hidet
# import torch
# from hidet.ir.expr import cast
# from hidet.ir.dtypes import f16
# from hidet.ir.module import IRModule
# from hidet.ir.primitives.runtime import request_cuda_workspace
# from hllm.ops.base import Operator
# from hidet.ir.library import tune
#
#
# class MatmulOperator(Operator):
#     def __init__(self, batch_size: int, m_size: int, n_size: int, k_size: int):
#         super().__init__({
#             'batch_size': batch_size,
#             'm_size': m_size,
#             'n_size': n_size,
#             'k_size': k_size,
#         })
#         self.batch_size: int = batch_size
#         self.m_size: int = m_size
#         self.n_size: int = n_size
#         self.k_size: int = k_size
#
#     def dummy_params(self) -> List[torch.Tensor]:
#         return [
#             torch.empty(self.batch_size, self.m_size, self.k_size, dtype=torch.float16, device='cuda'),
#             torch.empty(self.batch_size, self.k_size, self.n_size, dtype=torch.float16, device='cuda'),
#         ]
#
#     def implement(self) -> List[IRModule]:
#         return tune.extract_ir_modules(self.schedule)
#
#     @tune.space(
#         2,
#         block_m=[16],
#         block_n=[32, 64, 128],
#         block_k=[32, 64, 128],
#         k_parts=[1, 2, 4, 8, 16],
#         block_size=[128, 256]
#     )
#     def schedule(self, block_m: int, block_n: int, block_k: int, k_parts: int, block_size: int) -> IRModule:
#         from hidet.lang import attrs
#         from hidet.lang import tile as ti
#
#         k_part_size: int = (self.k_size + k_parts - 1) // k_parts
#         m_blocks: int = (self.m_size + block_m - 1) // block_m
#         n_blocks: int = (self.n_size + block_n - 1) // block_n
#
#         with hidet.script_module() as script_module:
#             @hidet.script
#             def matmul_kernel(
#                 a_ptr: ~f16,    # [batch_size, m_size, k_size]
#                 b_ptr: ~f16,    # [batch_size, k_size, n_size]
#                 c_ptr: ~f16     # [batch_size, k_parts, m_size, n_size]
#             ):
#                 attrs.func_kind = 'cuda_tile'
#                 attrs.cuda.block_dim = 256
#                 attrs.cuda.grid_dim = (
#                     ti.cdiv(self.m_size, block_m) * ti.cdiv(self.n_size, block_n),
#                     self.batch_size * k_parts
#                 )
#
#
#                 batch, k_part = ti.deserialize(ti.program_id(idx=1), shape=[self.batch_size, k_parts])
#                 m_part, n_part = ti.deserialize(ti.program_id(idx=0), shape=[m_blocks, n_blocks])
#                 m_offset = m_part * block_m
#                 n_offset = n_part * block_n
#                 k_offset = k_part * k_part_size
#
#                 a_ptrs = ti.compute(
#                     shape=[block_m, block_k],
#                     f_compute=lambda ii, kk: (
#                         a_ptr + batch * self.m_size * self.k_size + (m_offset + ii) * self.k_size + k_offset
#                     )
#                 )
#                 b_ptrs = ti.compute(
#                     shape=[block_m, block_k],
#                     f_compute=lambda kk, jj: (
#                         b_ptr + batch * self.k_size * self.n_size + k_offset * self.n_size + (n_offset + jj)
#                     )
#                 )
#                 a_mask = ti.compute(
#                     shape=[block_m, block_k],
#                     f_compute=lambda ii, kk: (
#                         (m_offset + ii) < self.m_size and (k_offset + kk) < self.k_size
#                     )
#                 )
#                 b_mask = ti.compute(
#                     shape=[block_m, block_k],
#                     f_compute=lambda kk, jj: (
#                         (k_offset + kk) < self.k_size and (n_offset + jj) < self.n_size
#                     )
#                 )
#
#                 for k in range(ti.cdiv(k_part_size, block_k)):
#                     a = ti.load(a_ptrs)
#
#
#
#             @hidet.script
#             def reduce(x_ptr: ~f16, y_ptr: ~f16):
#                 attrs.func_kind = 'cuda_tile'
#                 attrs.cuda.block_dim = 256
#                 attrs.cuda.grid_dim = h_size // reduce_block_h
#
#                 pid = ti.program_id()
#                 h_offsets = pid * reduce_block_h + ti.arange(0, reduce_block_h)
#
#                 x_ptrs = (
#                     x_ptr
#                     + ti.expand_dims(ti.arange(0, reduce_block_s), axis=1) * h_size
#                     + h_offsets
#                 )
#                 acc = ti.zeros([reduce_block_s, reduce_block_h], dtype=f16)
#                 for k in range((m_size // block_m) * self.seq // reduce_block_s):
#                     mask = (
#                         ti.expand_dims(
#                             ti.arange(0, reduce_block_s), axis=1
#                         ) < m_size // block_m * self.seq - k * reduce_block_s
#                     )
#                     acc += ti.load(x_ptrs, mask=mask)
#                     x_ptrs += reduce_block_s * h_size
#
#                 ti.store(
#                     ptr=y_ptr + ti.expand_dims(ti.arange(0, reduce_block_s), axis=1) * h_size + h_offsets,
#                     value=acc,
#                     mask=ti.expand_dims(ti.arange(0, reduce_block_s), axis=1) < self.seq
#                 )
#
#             @hidet.script
#             def launch(x_ptr: ~f16, w1_ptr: ~f16, w2_ptr: ~f16, y_ptr: ~f16):
#                 attrs.func_kind = 'public'
#
#                 y1_ptr = cast(request_cuda_workspace(nbytes=(m_size // block_m) * self.seq * h_size * f16.nbytes), ~f16)
#
#                 llama_ffn(x_ptr, w1_ptr, w2_ptr, y1_ptr)
#                 reduce(y1_ptr, y_ptr)
#
#         return script_module.ir_module()
#
#
# def demo():
#     hidet.option.search_space(0)
#     hidet.option.save_lower_ir()
#     hidet.option.cache_dir('./outs/cache')
#     hidet.utils.clear_cache_dir('./hops')
#
#     op = LlamaMLPOperator(16, 4096, 12288)
#     dummy_params = op.dummy_params()
#     op(*dummy_params)
#
#
# if __name__ == '__main__':
#     demo()
