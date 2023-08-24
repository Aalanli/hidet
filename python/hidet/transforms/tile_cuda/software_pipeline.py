from hidet.ir.func import Function
from hidet.ir.functors import IRRewriter
from hidet.ir.tile.ops.dot import Dot, SimtDot
from hidet.ir.tile.type import TileType
from hidet.ir.tools import TypeInfer
from hidet.ir.type import DataType
from hidet.transforms.base import TileFunctionPass


"""
Apply software pipeline to the load operators in a loop:
  
  for i in range(n) with other_args=..., ptr_args=inits:
      ...
      ptr = ptr_expr(ptr_args)
      x = load(ptr)
      ...
      yield iter_expr(ptr_args), ...

convert to:

  stages = 3
  
  buf = alloc_tensor([stages, ...])   # ... is the shape of x
  
  ptr_args0 = ptr_inits
  ptr0 = ptr_expr(ptr_args0)
  insert_slice_async(ptr0, buf, mask, other, axis=0, index=0)
  
  ptr_args1 = iter_expr(ptr_args0)
  ptr1 = ptr_expr(ptr_args1)
  insert_slice_async(ptr1, buf, mask, other, axis=0, index=1)
  
  async_wait(stages - 2)
  s0 = extract_slice(buf, axis=0, index=0)
  ptr_args2 = iter_expr(ptr_args1)
  
  for i in range(n) with (
     other_args=...,
     ptr_args = ptr_args2, 
     s = s0, 
     insert_index = 2, 
     extract_index = 1, 
   ):
     ...
     x = convert_layout(s, x's layout)
     ...
     ptr = ptr_expr(ptr_args)
     insert_slice_async(ptr, buf, mask=mask && i < n, other, axis=0, index=insert_index)
     ptr_args' = iter_expr(ptr_args)
     async_wait(staged - 2)
     s' = extract_slice(buf, axis=0, index=extract_index)
     insert_index' = (insert_index + 1) % stages
     extract_index' = (extract_index + 1) % stages
     yield ..., iter_expr(ptr_args), s', insert_index', extract_index'
  async_wait(0)

 Let depends(loop, x) be the loop's arguments that are depended by the computation of x

 To follow several steps to achieve above transformation:
 1. Find the pairs of (loop, (load_1, load_2, ...)) where the loads are directly in the loop body without nested loops.
 2. For each (loop, loads) pair, do the following:
    2.1. Let ptr_i be the pointer argument of the load_i. Find ptr_args = union{depends(loop, ptr_i) for all load_i}.
    2.2. Repeat ptr_args = depends(loop, ptr_args) until ptr_args is stable.
    2.3. Right before the loop, creates the shared memory and rematerialize the computation of ptr_args
            buf = alloc_tensor([stages, ...])   # ... is the shape of x
            ... (see above example)
    2.4. Update loop arguments to include ptr_args, s, insert_index, extract_index
    2.5. Replace the load(...) with convert_layout(s)
    2.6. At the end of the loop body, insert the code like:
            ptr = ptr_expr(ptr_args)
            insert_slice_async(ptr, buf, mask=mask && i < n, other, axis=0, index=insert_index)
            ptr_args' = iter_expr(ptr_args)
            async_wait(staged - 2)
            s' = extract_slice(buf, axis=0, index=extract_index)
            insert_index' = (insert_index + 1) % stages
            extract_index' = (extract_index + 1) % stages
    2.7. Update yield statement to
            yield ..., iter_expr(ptr_args), s', insert_index', extract_index'
    2.8. Add async_wait(0) after the loop body
"""


class SoftwarePipelineRewriter(IRRewriter):
    def __init__(self):
        super().__init__()
