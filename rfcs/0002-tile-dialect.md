- Feature Name: Tile Dialect for Hidet Script
- Start Date: 2023-07-25
- RFC PR: N/A
- GitHub Issue: N/A

# Summary
[summary]: #summary

Todo: A simple introduction to the programming model and its corresponding CUDA counterparts.

Tile-based programming: use tile and scalar as basic arthimatic data types.
Corase-grained operations: describe what a thread block should do instead of what each thread should do.
Serial execution: the execution of each operation can be thought as serial execution.

# Motivation
[motivation]: #motivation

Why are we doing this? What use cases does it support? What is the expected outcome?

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

Explain the proposal as if it was already included in hidet and you were teaching it to a Hidet user. 

## An example

TODO: use a concrete example to explain what the new dialect will look like. 

```python
import hidet
from hidet.dtypes import float32
from hidet.lang import attrs, spatial

# use a submodule in `hidet.script` like `block` or `tile` or anything similar
# to store the specialized operations for the dialect.
from hidet.lang import tile as T 

BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 16

# similar to other hidet script functions, we use `hidet.script` to decorate the functions written 
# in block dialect
@hidet.script    
def matmul(
    m_size: int, n_size: int, k_size: int,    # the size of matmul: a[M, K] x b[K, N] = c[M, N]
    ptr_a: ~float32, ptr_b: ~float32, ptr_c: ~float32    # ~float32: "a pointer to float32 data"
):  
    # `cuda_tile` indicate this is a function writting in block dialect (alternative: 'cuda_block')
    attrs.func_kind = 'cuda_tile'    
    
    pid = T.program_id()    # get the program id (i.e., the block index in the context of cuda programming)

    bcnt_m = (m_size + BLOCK_M - 1) / BLOCK_M
    bcnt_n = (n_size + BLOCK_N - 1) / BLOCK_N

    offset_m, offset_n = spatial(bcnt_m, bcnt_n).map(pid)  # we can also use task mapping in the dialect
    
    # We can use T.arange(...) to create a tile and support the arthimatic computation between tile
    # and scalar with the broadcasting sematics as in numpy
    range_m = offset_m * BLOCK_M + T.arange(BLOCK_M)  
    range_n = offset_n * BLOCK_N + T.arange(BLOCK_N)
    range_k = T.arange(BLOCK_K)

    # get the offset of the a/b tile
    ptrs_a = ptr_a + range_m[:, None] * k_size + range_k[None, :]  # [BLOCK_M, BLOCK_K]
    ptrs_b = ptr_b + range_k[:, None] * n_size + range_n[None, :]  # [BLOCK_K, BLOCK_N]

    # define a accumulator
    acc = T.zeros([BLOCK_M, BLOCK_N], dtype=float32)
    
    # main k-loop
    for k in range(k_size / BLOCK_K):
        # load tile a and b
        a = T.load(ptrs_a)
        b = T.load(ptrs_b)
        # matrix multiply accumulate
        c += T.dot(a, b)
        # iterate to the next tile 
        ptrs_a += BLOCK_K
        ptrs_b += BLOCK_K * n_size
    
    # write back
    ptrs_c = ptr_c + range_m[:, None] * n_size + range_n[None, :]  # [BLOCK_M, BLOCK_N]
    T.store(ptrs_c, acc)
```

## todo: more sections

to teach the kernel developers how to write a kernel in the dialect.

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

We should include a more comprehensive introduction to the new dialect in the reference-level explanation.

Let's temporarily define the dialect in the `hidet.ir.dialects.tile` submodule, and put the new types, layouts, etc.
in `hidet.ir.dialects.tile.type` and `hidet.ir.dialects.tile.layout` submodules.

We need to formally define:

## the type system of the dialect (scalar, tile)

```python
from typing import List
from hidet.ir.type import BaseType
from hidet.ir.layout import DataLayout


class TileLayout:
  def __init__(self, shape):
    self.shape: List[int] = shape # all tile shapes are static integers known at compile time


class RegisterLayout(TileLayout):
  def __init__(self, shape, num_threads: int, mapping: DataLayout):
    super().__init__(shape)
    self.num_threads: int = num_threads
    self.mapping: DataLayout = mapping


class SharedLayout(TileLayout):
  def __init__(self, shape):
    super().__init__(shape)


# define some subclass of TileLayout to represent different tile layouts

class TileType:
  dtype: BaseType
  layout: TileLayout

# for scalar types, our existing scalar type system should be enough
```

## Operations

For the operations that have corresponding expressions like addition, subtraction, multiplication, division,
we directly use the existing Expr subclass (like Add) in `hidet.ir.expr` module to represent them.

For the operations like reduce, dot, etc. we define a new subclass of Expr to represent them.

```python
from typing import List
from hidet.ir.type import DataType
from hidet.ir.builders import StmtBuilder
from hidet.ir.expr import Expr, Var

class ConstructionContext:
  pass

class TileOperator:
  def __init__(self):
    pass

  def implement_cuda(self, cc: ConstructionContext, *args: Expr) -> Var:
    raise NotImplementedError()


# the operators can be classified and put in different submodules like `hidet.ir.dialects.tile.ops.reduce`,
# `hidet.ir.dialects.tile.ops.create`, or `hidet.ir.dialects.tile.ops.dot`, etc.

class ReduceOp(TileOperator):
  pass


class DotOp(TileOperator):
  pass


class ScanOp(TileOperator):
  pass


class ElementwiseOp(TileOperator):
  pass


class ZerosOp(TileOperator):
  def __init__(self, shape: List[int], dtype: DataType):
    super().__init__()
    self.shape = shape
    self.dtype = dtype

# ...


class CallTileOp(Expr):
  op: TileOperator
```

In the tile dialect, there will only be the `CallTileOp` expression. 
For statements, there will be the following statements:
1. IfStmt
2. ForStmt
3. ForMappingStmt
4. WhileStmt
5. ReturnStmt
```python


```
## the primitive operations: load and store, dot, reduce, ... 
## the IR design for the new dialect. 

### What new IR do we need? How to integrate the IR in our current IR? 
### What passes do we need to 

a) achieve the necessary optimizations and,
b) lower this dialect IR to our existing hidet tensor program IR?

# Drawbacks
[drawbacks]: #drawbacks

Why should we *not* do this?

# Rationale and alternatives
[rationale-and-alternatives]: #rationale-and-alternatives

- Why is this design the best in the space of possible designs?
- What other designs have been considered and what is the rationale for not choosing them?
- What is the impact of not doing this?

# Prior art
[prior-art]: #prior-art

Discuss prior art, both the good and the bad, in relation to this proposal.
A few examples of what this can include are:

- Does this feature exist in other ML compilers or languages and discuss the experience their community has had?
- For community proposals: Is this done by some other community and what were their experiences with it?
- For other teams: What lessons can we learn from what other communities have done here?
- Papers: Are there any published papers or great posts that discuss this? 
  If you have some relevant papers to refer to, this can serve as a more detailed theoretical background.

If there is no prior art, that is fine - your ideas are interesting to us whether they are 
  brand new or if it is an adaptation from other languages.

Note that while precedent set by other languages is some motivation, it does not on its own motivate an RFC.
Please also take into consideration that TVM intentionally diverges from other compilers.

# Unresolved questions
[unresolved-questions]: #unresolved-questions

- What parts of the design do you expect to resolve through the RFC process before this gets merged?
- What parts of the design do you expect to resolve through the implementation of this feature before stabilization?
- What related issues do you consider out of scope for this RFC that could be addressed in the future 
  independently of the solution that comes out of this RFC?

# Future possibilities
[future-possibilities]: #future-possibilities

Think about what the natural extension and evolution of your proposal would
be and how it would affect the language and project as a whole in a holistic
way. Try to use this section as a tool to more fully consider all possible
interactions with the project and language in your proposal.
Also consider how this all fits into the roadmap for the project
and of the relevant sub-team.

This is also a good place to "dump ideas", if they are out of scope for the
RFC you are writing but otherwise related.

If you have tried and cannot think of any future possibilities,
you may simply state that you cannot think of anything.

Note that having something written down in the future-possibilities section
is not a reason to accept the current or a future RFC; such notes should be
in the section on motivation or rationale in this or subsequent RFCs.
The section merely provides additional information.

