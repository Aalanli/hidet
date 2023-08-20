from typing import List, Dict, Optional, Tuple

from hidet.ir.expr import Var, Call, var
from hidet.ir.func import Function
from hidet.ir.functors import IRRewriter, IRVisitor
from hidet.ir.stmt import LetStmt, DeclareStmt, AssignStmt
from hidet.ir.tile.ops import (
    Arange, Full, Broadcast, BinaryTileOp, ReduceOp, Dot, ExpandDims, SimtDot, Store, Construct, convert_layout
)
from hidet.ir.tile.expr import CallTileOp
from hidet.ir.primitives import lookup_primitive_function
from hidet.ir.tile.layout import (
    TileLayout,
    SharedLayout,
    BlockLayout,
    DotOperandLayout,
    FlattenBlockLayout,
    BlockDotOperandLayout,
)
from hidet.ir.tile.type import TileType
from hidet.ir.tools import TypeInfer, simplify_to_int
from hidet.utils import prod, is_power_of_two
from hidet.utils import same_list
from hidet.transforms.base import TileFunctionPass
from hidet.transforms.tile_generic.convert_tile_expr_to_let import convert_to_let


class Alloc:
    def __init__(self, v: Var, nbytes: int):
        self.var: Var = v
        self.nbytes: int = nbytes
        self.offset: Optional[int] = None


class ExtractSharedMemoryAllocation(IRVisitor):
    """
    let a = alloc_shared(5 bytes)
      let b = alloc_shared(10 bytes)
        use(b)
      let c = alloc_shared(20 bytes)
        use(c)

    then, we will have:
    allocations = [a, b, c]
    edges = {a: [b, c], b: [a], c: [a]}
    """
    alloc_shared_func_var: Var = lookup_primitive_function('cuda_alloc_shared').var

    def __init__(self):
        super().__init__()
        self.allocations: List[Alloc] = []
        self.edges: Dict[Alloc, List[Alloc]] = {}
        self.alloc2offset: Dict[Alloc, int] = {}

        self.let_scoped_alloc: List[Alloc] = []

    def plan(self, max_nbytes: int):
        # determine the offset of each allocation in the shared memory so that the allocations with shared lifetime
        # will be placed in different regions of the shared memory (e.g., no overlap)
        # we use a greedy algorithm to do this:
        #
        # repeat the following process until all allocations are placed:
        # 1. find the allocation with the largest size
        # 2. get the allocations connected to it (with shared lifetime) that have been placed already
        # 3. place the current allocation in the first available region that is large enough to hold it

        # sort the allocations by size in descending order
        allocations: List[Alloc] = list(sorted(self.allocations, key=lambda alloc: alloc.nbytes, reverse=True))

        for u in allocations:
            for v in self.edges[u]:
                events: List[Tuple[int, int]] = []


    def visit_LetStmt(self, stmt: LetStmt):
        num_enqueued: int = 0

        for bind_var, bind_value in zip(stmt.bind_vars, stmt.bind_values):
            if isinstance(bind_value, Call) and bind_value.func_var == self.alloc_shared_func_var:
                nbytes: int = simplify_to_int(bind_value.args[0])
                u_alloc = Alloc(bind_var, nbytes)
                self.allocations.append(u_alloc)
                self.edges[u_alloc] = []
                for v_alloc in self.let_scoped_alloc:
                    self.edges[u_alloc].append(v_alloc)
                    self.edges[v_alloc].append(u_alloc)
                self.let_scoped_alloc.append(u_alloc)
                num_enqueued += 1

        self.visit(stmt.body)

        for _ in range(num_enqueued):
            self.let_scoped_alloc.pop()


class PlanSharedMemoryRewriter(IRRewriter):
    def __init__(self):
        super().__init__()

    def visit_LetStmt(self, stmt: LetStmt):
        pass


class PlanSharedMemoryPass(TileFunctionPass):
    def process_tile_func(self, func: Function) -> Function:
        return self.apply_rewriter_list(func, [PlanSharedMemoryRewriter()])
