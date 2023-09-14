from typing import Dict, List, Tuple
from collections import defaultdict
from hidet.ir.type import sizeof
from hidet.ir.expr import Var
from hidet.ir.stmt import LetStmt, Stmt, DeclareStmt
from hidet.ir.func import Function
from hidet.ir.functors import IRVisitor, IRRewriter
from hidet.ir.tools import TypeInfer
from hidet.ir.tile.type import TileType
from hidet.ir.tile.expr import CallTileOp
from hidet.ir.tile.stmt import PureForStmt, YieldStmt
from hidet.ir.tile.ops.convert_layout import ConvertLayout
from hidet.ir.tile.ops.smem import AllocTensor, InsertSliceAsync, ExtractSlice, StoreShared
from hidet.transforms.base import TileFunctionPass
from hidet.transforms.tile import annotations
from hidet.utils import prod

"""
Plan the shared memory allocation.

Problem:
The shared memory is a precious resource in GPU, we need to allocate shared memory for tensors in shared scope to
make the tensors with non-overlapping lifetime to share the same shared memory space.
In tile dialect, we have the operator AllocTensor to allocate a tensor in shared memory, it has an attribute
"global_offset" to specify the offset of the shared memory region in the whole shared memory space window.
This pass tries to find the best global offset for each AllocTensor operator to minimize the total shared memory usage
while making sure that the tensors with overlapping lifetime do not share the same shared memory space.

Solution:
1. Associate the tile variables in shared scope with the AllocTensor operators.
   For example:
   ```
       let a = alloc_tensor([3, 8])
       let b = alloc_tensor([3, 16])
       let c = extract_slice(a, axis=0, index=0)
       let d = convert_layout(b, scope=shared, layout=...)
   ```
   We will have the following association:
       {a, c}: alloc_tensor([3, 8])
       {b, d}: alloc_tensor([3, 16])
2. Analyze the lifespan of each tile variable in shared scope, the lifespan is a pair of scope numbering
3. For each pair of AllocTensor operators, we check if any variables associated with them have overlapping lifespan.
4. With the overlapping information, use a greedy algorithm to find the best global offset for each AllocTensor operator
"""


class AssociateVisitor(IRVisitor):
    def __init__(self):
        super().__init__()
        self.type_infer = TypeInfer()
        self.var2alloc: Dict[Var, AllocTensor] = {}
        self.alloc2var: Dict[AllocTensor, List[Var]] = defaultdict(list)

    def is_alloc_tensor(self, e):
        return isinstance(e, CallTileOp) and isinstance(e.op, AllocTensor)

    def as_alloc_tensor(self, e) -> AllocTensor:
        assert isinstance(e, CallTileOp) and isinstance(e.op, AllocTensor)
        return e.op

    def associate(self, var: Var, alloc: AllocTensor):
        self.var2alloc[var] = alloc
        self.alloc2var[alloc].append(var)

    def visit_LetStmt(self, stmt: LetStmt):
        for bind_var, bind_value in zip(stmt.bind_vars, stmt.bind_values):
            assert isinstance(bind_var, Var)
            if not (isinstance(bind_var.type, TileType) and bind_var.type.scope.is_shared()):
                continue
            assert isinstance(bind_value, CallTileOp)
            op = bind_value.op
            if isinstance(op, AllocTensor):
                self.associate(bind_var, op)
            elif isinstance(op, InsertSliceAsync):
                assert isinstance(op.dst, Var)
                self.associate(bind_var, self.var2alloc[op.dst])
            elif isinstance(op, ExtractSlice):
                assert isinstance(op.src, Var)
                self.associate(bind_var, self.var2alloc[op.src])
            elif isinstance(op, ConvertLayout):
                assert isinstance(op.x, Var)
                # op.x must be a shared tensor, otherwise this op will be resolved in ResolveConvertLayoutPass
                self.associate(bind_var, self.var2alloc[op.x])
            elif isinstance(op, StoreShared):
                assert isinstance(op.dst, Var)
                self.associate(bind_var, self.var2alloc[op.dst])
            else:
                raise NotImplementedError(op.__class__.__name__)
        self.visit(stmt.body)

    def visit_PureForStmt(self, stmt: PureForStmt):
        for arg, let_var, value in zip(stmt.args, stmt.let_vars, stmt.values):
            if not (isinstance(arg.type, TileType) and arg.type.scope.is_shared()):
                continue
            assert isinstance(value, Var) and value in self.var2alloc
            self.var2alloc[arg] = self.var2alloc[value]
            self.var2alloc[let_var] = self.var2alloc[value]

        self.pure_for_stmts.append(stmt)
        self.visit(stmt.body)
        self.pure_for_stmts.pop()
        self.visit(stmt.let_body)

    def visit_YieldStmt(self, stmt: YieldStmt):
        for_stmt = self.pure_for_stmts[-1]
        for arg, value in zip(for_stmt.args, stmt.values):
            if not (isinstance(arg.type, TileType) and arg.type.scope.is_shared()):
                continue
            assert arg in self.var2alloc
            assert value in self.var2alloc


class LifeSpan:
    def __init__(self, left: int = int(1e9), right: int = int(-1e9)):
        self.left: int = left
        self.right: int = right

    def __str__(self):
        return [self.left, self.right].__str__()

    def expand(self, clock: int):
        self.left = min(self.left, clock)
        self.right = max(self.right, clock)

    def merge(self, other):
        self.left = min(self.left, other.left)
        self.right = max(self.right, other.right)

    def intersect_with(self, other) -> bool:
        return self.left <= other.right and other.left <= self.right


class LifeSpanAnalyzer(IRVisitor):
    def __init__(self):
        super().__init__()
        self.var2lifespan: Dict[Var, LifeSpan] = {}
        self.clock: int = 0

    def visit(self, node):
        super().visit(node)
        if isinstance(node, Stmt):
            # after every statement, we enter the next clock
            self.clock += 1

    def visit_Var(self, v: Var):
        if isinstance(v.type, TileType) and not v.type.scope.is_shared():
            # only analyze the lifespan of shared variables
            return

        if v not in self.var2lifespan:
            self.var2lifespan[v] = LifeSpan(self.clock, self.clock)
        else:
            self.var2lifespan[v].expand(self.clock)

    def visit_LetStmt(self, stmt: LetStmt):
        for bind_var, bind_value in zip(stmt.bind_vars, stmt.bind_values):
            self.visit(bind_var)
            self.visit(bind_value)
            self.clock += 1
        self.visit(stmt.body)

    def visit_PureForStmt(self, stmt: PureForStmt):
        for arg, value in zip(stmt.args, stmt.values):
            self.visit(arg)
            self.visit(value)
            self.clock += 1
        self.visit(stmt.body)
        for let_var in stmt.let_vars:
            self.visit(let_var)
        self.clock += 1
        self.visit(stmt.let_body)

    def visit_DeclareStmt(self, stmt: DeclareStmt):
        raise RuntimeError('SSA form is required')


class ApplyPlanRewriter(IRRewriter):
    def __init__(self, alloc2offset: Dict[AllocTensor, int], dynamic_smem_nbytes: int):
        super().__init__()
        self.alloc2offset: Dict[AllocTensor, int] = alloc2offset
        self.dynamic_smem_nbytes: int = dynamic_smem_nbytes

    def visit_Function(self, func: Function):
        func = super().visit_Function(func)
        if func.kind == 'cuda_tile':
            func.attrs['cuda.dynamic_smem_bytes'] = self.dynamic_smem_nbytes
        return func

    def visit_AllocTensor(self, e: AllocTensor):
        assert e in self.alloc2offset
        ret = AllocTensor(e.dtype, e.shape, e.layout)
        ret.annotations[annotations.global_offset] = self.alloc2offset[e]
        return ret


Edges = Dict[AllocTensor, List[AllocTensor]]
Plan = Tuple[Dict[AllocTensor, int], int]


class PlanSharedMemoryPass(TileFunctionPass):
    @staticmethod
    def get_max_smem_size():
        from hidet.cuda.capability import capability

        return capability().sharedMemPerBlock

    def analyze_alloc_edges(
        self,
        allocations: List[AllocTensor],
        alloc2var: Dict[AllocTensor, List[Var]],
        var2lifespan: Dict[Var, LifeSpan]
    ) -> Dict[AllocTensor, List[AllocTensor]]:
        # if (u, v) in edges, then u and v have overlap
        edges: Edges = defaultdict(list)
        for u in allocations:
            for v in allocations:
                if u is v:
                    continue
                u_span, v_span = LifeSpan(), LifeSpan()
                for u_var in alloc2var[u]:
                    u_span.merge(var2lifespan[u_var])
                for v_var in alloc2var[v]:
                    v_span.merge(var2lifespan[v_var])
                if u_span.intersect_with(v_span) and v not in edges[u]:
                    edges[u].append(v)
                    edges[v].append(u)
        return edges

    def plan(self, allocations: List[AllocTensor], edges: Edges, max_nbytes: int, alignment=16) -> Plan:
        def alloc_nbytes(alloc: AllocTensor) -> int:
            local_shape: List[int] = alloc.layout.local_shape()
            return prod(local_shape) * sizeof(alloc.dtype)

        allocations: List[AllocTensor] = list(sorted(allocations, key=alloc_nbytes, reverse=True))
        plan: Dict[AllocTensor, int] = {}
        allocated: int = 0

        for u in allocations:
            # event: (offset, delta)
            events: List[Tuple[int, int]] = []
            for v in edges[u]:
                assert isinstance(u, AllocTensor)
                assert isinstance(v, AllocTensor)
                if v in plan:
                    aligned_nbytes = (alloc_nbytes(v) + alignment - 1) // alignment * alignment
                    events.append((plan[v], 1))
                    events.append((plan[v] + aligned_nbytes, -1))
            events.append((0, 0))
            events.append((max_nbytes, 0))
            events = sorted(events, key=lambda event: event[0])
            cnt = 0
            for i in range(len(events)):
                cnt += events[i][1]
                if cnt == 0 and i < len(events) - 1:
                    space = events[i + 1][0] - events[i][0]
                    if space >= alloc_nbytes(u):
                        plan[u] = events[i][0]
                        allocated = max(allocated, plan[u] + alloc_nbytes(u))
                        break
            else:
                raise RuntimeError('Cannot find a valid shared memory allocation plan.')
        return plan, allocated

    def process_tile_func(self, func: Function) -> Function:
        # step 1
        associate_visitor = AssociateVisitor()
        associate_visitor.visit(func)

        # step 2
        lifespan_analyzer = LifeSpanAnalyzer()
        lifespan_analyzer.visit(func)

        # step 3
        alloc2var: Dict[AllocTensor, List[Var]] = associate_visitor.alloc2var
        var2lifespan: Dict[Var, LifeSpan] = lifespan_analyzer.var2lifespan
        allocations: List[AllocTensor] = list(associate_visitor.alloc2var.keys())
        edges: Edges = self.analyze_alloc_edges(allocations, alloc2var, var2lifespan)

        # step 4
        plan: Dict[AllocTensor, int]
        allocated: int
        plan, allocated = self.plan(allocations, edges, max_nbytes=self.get_max_smem_size())

        rewriter = ApplyPlanRewriter(plan, allocated)
        func = rewriter(func)

        return func


def plan_shared_memory_pass() -> TileFunctionPass:
    return PlanSharedMemoryPass()
