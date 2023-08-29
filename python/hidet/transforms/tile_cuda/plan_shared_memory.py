from typing import List, Dict, Optional, Tuple

import hidet.cuda.capability
from hidet.ir.type import TensorPointerType, PointerType
from hidet.ir.expr import Var, Call, Expr, var
from hidet.ir.func import Function
from hidet.ir.functors import IRRewriter, IRVisitor
from hidet.ir.stmt import LetStmt, DeclareStmt, AssignStmt, SeqStmt
from hidet.ir.module import IRModule
from hidet.ir.tile.ops import Arange, Full, Broadcast, BinaryTileOp, ReduceOp, Dot, ExpandDims, SimtDot, Store
from hidet.ir.tile.ops import Construct, convert_layout
from hidet.ir.tile.expr import CallTileOp
from hidet.ir.primitives import lookup_primitive_function
from hidet.ir.primitives.cuda.smem import dynamic_shared_memory
from hidet.ir.primitives.cuda.tile import alloc_shared
from hidet.ir.tile.layout import TileLayout, SharedLayout, BlockLayout, DotOperandLayout, FlattenBlockLayout
from hidet.ir.tile.layout import BlockDotOperandLayout
from hidet.ir.tile.type import TileType
from hidet.ir.tools import TypeInfer, simplify_to_int
from hidet.utils import prod, is_power_of_two
from hidet.utils import same_list
from hidet.transforms.base import FunctionPass
from hidet.transforms.declare_to_let import DeclareToLetRewriter, UpliftLetBodyRewriter

alloc_name = 'cuda_alloc_shared'

class Alloc:
    def __init__(self, v: Var, nbytes: int):
        self.var: Var = v
        self.nbytes: int = nbytes
        self.offset: Optional[int] = None


def check_alloc_shared_call(call: Call) -> bool:
    return isinstance(call, Call) and call.func_var.name == 'cuda_alloc_shared'


class SharedMemoryAllocator(IRVisitor):
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

    def __init__(self):
        super().__init__()
        self.allocations: List[Alloc] = []
        self.edges: Dict[Alloc, List[Alloc]] = {}
        self.alloc2offset: Dict[Alloc, int] = {}

        self.let_scoped_alloc: List[Alloc] = []
        self.var2alloc: Dict[Var, Alloc] = {}

        self.dynamic_smem_size: Optional[int] = None

    def plan(self, max_nbytes: int, alignment: int = 16):
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
        max_allocated: int = 0

        for u in allocations:
            events: List[Tuple[int, int]] = []
            for v in self.edges[u]:
                assert isinstance(u, Alloc)
                assert isinstance(v, Alloc)
                if v.offset is not None:
                    aligned_nbytes = (v.nbytes + alignment - 1) // alignment * alignment
                    events.append((v.offset, 1))
                    events.append((v.offset + aligned_nbytes, -1))
            events.append((0, 0))
            events.append((max_nbytes, 0))
            events = sorted(events, key=lambda event: event[0])
            cnt = 0
            for i in range(len(events)):
                cnt += events[i][1]
                if cnt == 0 and i < len(events) - 1:
                    space = events[i + 1][0] - events[i][0]
                    if space >= u.nbytes:
                        u.offset = events[i][0]
                        max_allocated = max(max_allocated, u.offset + u.nbytes)
                        break
            else:
                raise RuntimeError('Cannot find a valid shared memory allocation plan.')

        self.dynamic_smem_size = max_allocated

    def visit_LetStmt(self, stmt: LetStmt):
        num_enqueued: int = 0

        for bind_var, bind_value in zip(stmt.bind_vars, stmt.bind_values):
            if isinstance(bind_value, Call) and check_alloc_shared_call(bind_value):
                nbytes: int = simplify_to_int(bind_value.args[0])
                u_alloc = Alloc(bind_var, nbytes)
                self.allocations.append(u_alloc)
                self.var2alloc[bind_var] = u_alloc
                self.edges[u_alloc] = []
                for v_alloc in self.let_scoped_alloc:
                    self.edges[u_alloc].append(v_alloc)
                    self.edges[v_alloc].append(u_alloc)
                self.let_scoped_alloc.append(u_alloc)
                num_enqueued += 1

        self.visit(stmt.body)

        for _ in range(num_enqueued):
            self.let_scoped_alloc.pop()


class AllocSharedMarkerVisitor(IRVisitor):
    def __init__(self):
        super().__init__()
        self.var2alloc: Dict[Var, int] = {}

    def mark(self, v: Var, nbytes: int):
        if v in self.var2alloc:
            raise ValueError('Variable {} has already been marked.'.format(v.name))
        self.var2alloc[v] = nbytes

    def is_call_to_alloc_shared(self, call: Expr) -> bool:
        return isinstance(call, Call) and call.func_var.name == alloc_name

    def get_nbytes(self, call: Expr) -> int:
        assert isinstance(call, Call) and call.func_var.name == alloc_name
        return int(call.args[0])

    def visit_LetStmt(self, stmt: LetStmt):
        for bind_var, bind_value in zip(stmt.bind_vars, stmt.bind_values):
            if self.is_call_to_alloc_shared(bind_value):
                self.mark(bind_var, self.get_nbytes(bind_value))
                self.visit(bind_value.args)
            else:
                self.visit(bind_value)
        self.visit(stmt.body)

    def visit_DeclareStmt(self, stmt: DeclareStmt):
        if stmt.init and self.is_call_to_alloc_shared(stmt.init):
            self.mark(stmt.var, self.get_nbytes(stmt.init))
        else:
            super().visit_DeclareStmt(stmt)

    def visit_AssignStmt(self, stmt: AssignStmt):
        if isinstance(stmt.value, Call) and self.is_call_to_alloc_shared(stmt.value):
            self.mark(stmt.var, int(stmt.value.args[0]))
        else:
            super().visit_AssignStmt(stmt)

    def visit_Call(self, call: Call):
        if call.func_var.name == 'cuda_alloc_shared':
            raise ValueError('cuda_alloc_shared should directly assigned to a variable via let or declare.')
        super().visit_Call(call)


class CanonicalizeAllocSharedRewriter(IRRewriter):
    def __init__(self):
        super().__init__()
        self.marker = AllocSharedMarkerVisitor()

    def visit_Function(self, func: Function):
        self.marker.visit(func)
        return super().visit_Function(func)

    def visit_DeclareStmt(self, stmt: DeclareStmt):
        if stmt.var in self.marker.var2alloc:
            nbytes = self.marker.var2alloc[stmt.var]
            return DeclareStmt(stmt.var, init=alloc_shared(nbytes))
        else:
            return super().visit_DeclareStmt(stmt)

    def visit_AssignStmt(self, stmt: AssignStmt):
        if stmt.var in self.marker.var2alloc:
            return SeqStmt([])
        else:
            return super().visit_AssignStmt(stmt)


class PlanSharedMemoryRewriter(IRRewriter):
    def __init__(self):
        super().__init__()
        self.allocator = SharedMemoryAllocator()

    def __call__(self, ir_module: IRModule):
        self.allocator.visit(ir_module)
        self.allocator.plan(max_nbytes=self.get_max_smem_size())
        return super().__call__(ir_module)

    @staticmethod
    def get_max_smem_size():
        from hidet.cuda.capability import capability
        return capability().sharedMemPerBlock

    def visit_Function(self, func: Function):
        func = super().visit_Function(func)
        if func.kind == 'cuda_kernel':
            func.attrs['cuda.dynamic_smem_bytes'] = self.allocator.dynamic_smem_size
        return func

    def visit_LetStmt(self, stmt: LetStmt):
        bind_values: List[Expr] = []
        for bind_var, bind_value in zip(stmt.bind_vars, stmt.bind_values):
            if isinstance(bind_value, Call) and check_alloc_shared_call(bind_value):
                offset: int = self.allocator.var2alloc[bind_var].offset
                assert offset is not None
                assert isinstance(bind_var, Var)
                if isinstance(bind_var.type, TensorPointerType):
                    dtype = bind_var.type.tensor_type.dtype
                elif isinstance(bind_var.type, PointerType):
                    dtype = bind_var.type
                else:
                    raise NotImplementedError()
                bind_values.append(dynamic_shared_memory(offset, dtype=dtype))
            else:
                bind_values.append(bind_value)
        body = self.visit(stmt.body)
        if same_list(stmt.bind_values, bind_values) and body is stmt.body:
            return stmt
        else:
            return LetStmt(stmt.bind_vars, bind_values, body)


class PlanSharedMemoryPass(FunctionPass):
    def process_func(self, func: Function) -> Function:
        return self.apply_transforms(
            func, [
                CanonicalizeAllocSharedRewriter(),
                DeclareToLetRewriter(),
                UpliftLetBodyRewriter(),
                PlanSharedMemoryRewriter()
            ]
        )


def plan_shared_memory_pass() -> FunctionPass:
    return PlanSharedMemoryPass()
