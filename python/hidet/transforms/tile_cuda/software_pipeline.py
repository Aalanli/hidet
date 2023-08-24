from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict
from hidet.ir.expr import Expr
from hidet.ir.dtypes import int32
from hidet.ir.func import Function
from hidet.ir.functors import IRRewriter, IRVisitor
from hidet.ir.tile.ops import AllocTensor, Load, InsertSliceAsync, AsyncCommitGroup, ExtractSlice, AsyncWait
from hidet.ir.tile.ops import ConvertLayout
from hidet.ir.tile.expr import TileOp
from hidet.ir.tile.type import TileType
from hidet.ir.tile.layout import SharedLayout
from hidet.ir.tile.stmt import PureForStmt
from hidet.ir.tools import TypeInfer, rewrite, collect
from hidet.ir.type import DataType
from hidet.ir.expr import Var
from hidet.ir.builders import StmtBuilder
from hidet.transforms.base import TileFunctionPass
from hidet.ir.stmt import LetStmt, Stmt, EvaluateStmt, SeqStmt
from hidet.ir.tile.stmt import PureForStmt, YieldStmt
from hidet.transforms.tile_generic.utils.definition_analyzer import DefinitionAnalyzer, VarDefinition, LetDefinition

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


class DetectLoadInLoopVisitor(IRVisitor):
    def __init__(self):
        super().__init__()
        self.loop2loads: Dict[PureForStmt, List[Load]] = defaultdict(list)
        self.loop2yields: Dict[PureForStmt, List[YieldStmt]] = defaultdict(list)

    def visit_Load(self, e: Load):
        if len(self.pure_for_stmts) > 0:
            loop = self.pure_for_stmts[-1]
            self.loop2loads[loop].append(e)
        super().visit_Load(e)

    def visit_YieldStmt(self, stmt: YieldStmt):
        if len(self.pure_for_stmts) > 0:
            loop = self.pure_for_stmts[-1]
            self.loop2yields[loop].append(stmt)
        super().visit_YieldStmt(stmt)


class DependencyGraphConstructor(IRVisitor):
    def __init__(self):
        super().__init__()
        self.depends: Dict[Var, List[Var]] = {}

    def get_direct_depends(self, e: Expr):
        self.memo.clear()
        self.visit(e)
        return [v for v in self.memo if isinstance(v, Var)]

    def add_depends(self, user: Var, depends: List[Var]):
        if user not in self.depends:
            self.depends[user] = []
        for v in depends:
            if v not in self.depends[user]:
                self.depends[user].append(v)

    def visit_LetStmt(self, stmt: LetStmt):
        for bind_var, bind_value in zip(stmt.bind_vars, stmt.bind_values):
            self.add_depends(bind_var, self.get_direct_depends(bind_value))
        self.visit(stmt.body)

    def visit_PureForStmt(self, stmt: PureForStmt):
        for arg, value in zip(stmt.args, stmt.values):
            self.add_depends(arg, self.get_direct_depends(value))
        self.pure_for_stmts.append(stmt)
        self.visit(stmt.body)
        self.pure_for_stmts.pop()
        self.visit(stmt.let_body)

    def visit_YieldStmt(self, stmt: YieldStmt):
        loop_stmt = self.pure_for_stmts[-1]
        for let_var, yield_value in zip(loop_stmt.let_vars, stmt.yields):
            self.add_depends(let_var, self.get_direct_depends(yield_value))


class Rematerializer:
    def __init__(self, args: List[Var], bind_vars: List[Var], bind_values: List[Expr], results: List[Var]):
        super().__init__()
        self.args: List[Var] = args
        self.bind_vars: List[Var] = bind_vars
        self.bind_values: List[Expr] = bind_values
        self.results: List[Var] = results

    @staticmethod
    def create(loop: PureForStmt, for_args: List[Var], values: List[Expr]):
        given_values = values
        values: List[Var] = [v for v in given_values if isinstance(v, Var)]
        assert len(values) == len(given_values), "Expected all values to be Var"

        definitions: Dict[Var, VarDefinition] = DefinitionAnalyzer().analyze(loop.body)
        bind_vars: List[Var] = []
        bind_values: List[Expr] = []

        # bfs starting from the given values to its dependencies to get the computations from for_args and external
        # variables (outside the loop) to the values
        visited: Set[Var] = set(values)
        queue: List[Var] = values.copy()

        while queue:
            u = queue.pop(0)

            if u not in definitions:
                # external variable defined outside the for loop, keep it as is
                continue
            else:
                # for variable defined inside the for loop
                definition = definitions[u]
                if not isinstance(definition, LetDefinition):
                    raise RuntimeError(f"Expected LetDefinition, got {definition}")
                if definition.bind_var in visited:
                    # already in the computation (bind_vars, bind_values), skip
                    continue
                bind_vars.append(definition.bind_var)
                bind_values.append(definition.bind_value)
                visited.add(definition.bind_var)
                queue.append(definition.bind_var)

        # inplace reverse the order of bind_vars and bind_values
        bind_vars.reverse()
        bind_values.reverse()
        return Rematerializer(for_args, bind_vars, bind_values, values)

    def rematerialize(self, updated_args: List[Var]) -> Tuple[List[Var], List[Expr], List[Var]]:
        bind_vars = []
        bind_values = []
        remap = {a: b for a, b in zip(self.args, updated_args)}
        for bind_var, bind_value in zip(self.bind_vars, self.bind_values):
            updated_bind_value = rewrite(bind_value, remap)
            updated_bind_var = Var(bind_var.hint, bind_var.type)
            remap[bind_var] = updated_bind_var
            bind_vars.append(updated_bind_var)
            bind_values.append(updated_bind_value)
        results = [remap[r] if r in remap else r for r in self.results]
        return bind_vars, bind_values, results


class SoftwarePipelineRewriter(IRRewriter):
    def __init__(self, loop, yield_stmt, loads, dependency_graph, num_stages=3):
        super().__init__()
        self.loop: PureForStmt = loop
        self.yield_stmt: YieldStmt = yield_stmt
        self.loads: List[Load] = loads
        self.dependency_graph: Dict[Var, List[Var]] = dependency_graph
        self.num_stages: int = num_stages
        self.type_infer = TypeInfer()

        # the information that will be used by Load and YieldStmt visitors, filled by PureForStmt visitor
        self.loop_args: List[Var] = []
        self.args_count: Dict[str, int] = {}
        self.for_extra_args: List[Var] = []

    def depends(self, users: List[Var]):
        # find all the variables that are the dependencies of users
        stack: List[Var] = list(users)
        visited: Set[Var] = set(users)
        while len(stack) > 0:
            u = stack.pop()
            for v in self.dependency_graph[u]:
                if v not in visited:
                    stack.append(v)
                    visited.add(v)
        return list(visited)

    def get_load_args(self) -> List[Var]:
        """ step 2.1: find the arguments of the loads """
        load_args: Set[Var] = set()
        for load in self.loads:
            assert isinstance(load.ptr, Var)
            load_args.add(load.ptr)
            if load.mask:
                assert isinstance(load.mask, Var)
                load_args.add(load.mask)
            if load.other:
                assert isinstance(load.other, Var)
                load_args.add(load.other)
        return list(load_args)

    def get_for_extra_args(self, all_for_args) -> List[Var]:
        """ step 2.2: find the self-contained set of arguments to compute load args as well as themselves """
        load_args = self.get_load_args()
        for_extra_args = load_args
        while True:
            orig_num = len(for_extra_args)
            for_extra_args = [v for v in self.depends(users=load_args) if v in all_for_args]
            if len(for_extra_args) == orig_num:
                # converged to a self-contained set of arguments to compute themselves during loop iteration
                break
        for_extra_args = [v for v in self.depends(users=load_args) if v in all_for_args]
        return for_extra_args

    def rematerialized_prefetch(self) -> Tuple[List[Stmt], List[Var], List[Var]]:
        stmts = []

        # allocate shared memory buffers
        load_types: List[TileType] = [self.type_infer(load.make_call()) for load in self.loads]
        buffer_vars: List[Var] = []
        for load_type in load_types:
            shape = [self.num_stages] + load_type.shape
            buffer_type = TileType(load_type.type, shape, layout=SharedLayout(shape))
            buffer_var = Var('smem', type=buffer_type)
            buffer_vars.append(buffer_var)
            stmts.append(LetStmt(buffer_var, AllocTensor(shape).make_call()))

        # construct the rematerializer for for_args calculation
        load_args = self.get_load_args()
        load_args_remat = Rematerializer.create(loop=self.loop, for_args=self.for_extra_args, values=load_args)
        for_args_remat = Rematerializer.create(
            loop=self.loop, for_args=self.for_extra_args, values=self.yield_stmt.yields
        )

        # rematerialize the load args and loop args
        arg2init: Dict[Var, Var] = {arg: value for arg, value in zip(self.loop.args, self.loop.values)}
        current_for_args: List[Var] = [arg2init[arg] for arg in self.for_extra_args]
        for i in range(self.num_stages - 1):
            # rematerialize the load arguments computations
            bind_vars, bind_values, remat_load_args = load_args_remat.rematerialize(current_for_args)
            stmts.append(LetStmt(bind_vars, bind_values))

            # rematerialize the load operations
            load_arg_map: Dict[Expr, Var] = {a: b for a, b in zip(load_args, remat_load_args)}
            for idx, load in enumerate(self.loads):
                # calculate the ptr, mask, and other arguments
                ptr = load_arg_map[load.ptr]
                mask = load_arg_map[load.mask] if load.mask else None
                other = load_arg_map[load.other] if load.other else None
                buf_var = buffer_vars[idx]
                op = InsertSliceAsync(ptr=ptr, dst=buf_var, index=int32(i), mask=mask, other=other, axis=0)
                new_buf_var = Var(buf_var.hint, type=buf_var.type)
                buffer_vars[idx] = new_buf_var
                stmts.append(LetStmt(new_buf_var, op.make_call()))
                stmts.append(EvaluateStmt(AsyncCommitGroup().make_call()))

            # rematerialize the loop arguments computations
            bind_vars, bind_values, current_for_args = for_args_remat.rematerialize(current_for_args)
            stmts.append(LetStmt(bind_vars, bind_values))

        # extract the first stage
        stmts.append(EvaluateStmt(AsyncWait(self.num_stages - 2)))
        tile_vars: List[Var] = []
        for idx, buf_var in enumerate(buffer_vars):
            shape = load_types[idx].shape
            op = ExtractSlice(buf_var, index=int32(0), axis=0, layout=SharedLayout(shape)).make_call()
            tile_var = Var('ext_slice', type=self.type_infer(op.make_call()))
            stmts.append(LetStmt(tile_var, op.make_call()))
            tile_vars.append(tile_var)

        # iterate the for_args
        bind_vars, bind_values, current_for_args = for_args_remat.rematerialize(current_for_args)
        stmts.append(LetStmt(bind_vars, bind_values))

        return stmts, current_for_args, tile_vars

    def update_loop_args_values(self, stmt: PureForStmt, tile_vars: List[Var], current_for_args: List[Var]):
        # prepare the new loop args and init values
        self.args_count['original'] = len(stmt.args)
        loop_args: List[Var] = stmt.args.copy()
        loop_values: List[Expr] = stmt.values.copy()
        # insert_index and extract_index
        self.args_count['num_indices'] = 2
        loop_args.append(Var('insert_index', type=int32))
        loop_values.append(int32(self.num_stages - 1))
        loop_args.append(Var('extract_index', type=int32))
        loop_values.append(int32(self.num_stages - 2))
        # extracted slices
        self.args_count['tile_vars'] = len(tile_vars)
        for tile_var in tile_vars:
            loop_args.append(Var(tile_var.hint, type=tile_var.type))
            loop_values.append(tile_var)
        # extra loop args used to compute load args
        self.args_count['extra_args'] = len(current_for_args)
        for current_for_arg in current_for_args:
            loop_args.append(Var(current_for_arg.hint, type=current_for_arg.type))
            loop_values.append(current_for_arg)
        return loop_args, loop_values

    def visit_PureForStmt(self, stmt: PureForStmt):
        if stmt is not self.loop:
            return super().visit_PureForStmt(stmt)

        stmts: List[Stmt] = []

        arg2value: Dict[Var, Expr] = {arg: value for arg, value in zip(stmt.args, stmt.values)}

        # step 2.1 to 2.2: the for args that are used to compute the load arguments
        self.for_extra_args: List[Var] = self.get_for_extra_args(all_for_args=stmt.args)

        # step 2.3: hoist the loading logic out of the loop body
        new_stmts, current_for_args, tile_vars = self.rematerialized_prefetch()
        stmts.extend(new_stmts)

        loop_args_values = self.update_loop_args_values(stmt, tile_vars, current_for_args)
        loop_args: List[Var] = loop_args_values[0]
        loop_values: List[Expr] = loop_args_values[1]

        # step 2.4 to 2.7 in separate visit methods
        self.loop_args = loop_args
        self.pure_for_stmts.append(stmt)
        body = self.visit(stmt.body)
        self.pure_for_stmts.pop()

        # update let vars
        let_vars = stmt.let_vars.copy()
        for idx in range(len(stmt.args), len(loop_args)):
            let_vars.append(Var(loop_args[idx].hint, type=loop_args[idx].type))

        # step 2.8
        let_body = self.visit(stmt.let_body)
        let_body = SeqStmt([
            EvaluateStmt(AsyncWait(int32(0))),
            let_body
        ])
        for_stmt = PureForStmt(
            args=loop_args,
            values=loop_values,
            loop_var=stmt.loop_var,
            extent=stmt.extent,
            body=body,
            let_vars=let_vars,
            let_body=let_body,
        )

        # concatenate the stmts
        body = for_stmt
        for s in reversed(stmts):
            if isinstance(s, LetStmt):
                if s.body is None:
                    body = LetStmt(s.bind_vars, s.bind_values, body)
                else:
                    body = SeqStmt([s, body])
            else:
                body = SeqStmt([s, body])

        # so far, we have finished the software pipelining
        return body

    def visit_Load(self, e: Load):
        if e not in self.loads:
            return super().visit_Load(e)
        # step 2.5: replace the load(...) with convert_layout(s, load's layout)
        idx = self.loads.index(e)
        ptr_type: TileType = self.type_infer(e.ptr)
        cvt = ConvertLayout(
            x=self.loop_args[self.args_count['original'] + self.args_count['num_indices'] + idx],
            layout=ptr_type.layout
        )
        return cvt

    def visit_YieldStmt(self, stmt: YieldStmt):
        if self.pure_for_stmts[-1] is not self.loop:
            return super().visit_YieldStmt(stmt)

        # 2.6: update the end of loop body
        for_extra_args: List[Var] = self.loop_args[-self.args_count['extra_args']:]



class SoftwarePipelinePass(TileFunctionPass):
    def process_tile_func(self, func: Function) -> Function:
        # step 1: find all the (loop, loads) pairs
        loop_loads_detector = DetectLoadInLoopVisitor()
        loop_loads_detector.visit(func)
        loop2loads = loop_loads_detector.loop2loads
        loop2yields = loop_loads_detector.loop2yields

        # step 2: for each pair (loop, loads), rewrite the loop
        for loop, loads in loop2loads.items():
            if len(loop2yields[loop]) != 1:
                # we expect there is only one yield statement in the loop to apply software pipelining
                continue
            yield_stmt = loop2yields[loop][0]

            # analyze dependency graph: depends[u] = [v | u depends on v directly]
            dep_graph_constructor = DependencyGraphConstructor()
            dep_graph_constructor.visit(func)
            dependency_graph = dep_graph_constructor.depends

            # step 2.1 to 2.8: rewrite the loop
            rewriter = SoftwarePipelineRewriter(loop, yield_stmt, loads, dependency_graph)
            func = rewriter(func)
        return func


def software_pipeline_pass() -> TileFunctionPass:
    return SoftwarePipelinePass()
