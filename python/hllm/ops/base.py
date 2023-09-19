from typing import List, Optional, Dict, Union, Tuple, Type
from tqdm import tqdm
import os
import hidet
import torch
from tabulate import tabulate
from hidet.ir.module import IRModule
from hidet.runtime import CompiledFunction, CompiledModule, compiled_module_exists
from hidet.drivers import build_ir_module
from hidet.utils.multiprocess import parallel_imap

Attr = Union[str, int]


class Operator:
    def __init__(self, attrs: Dict[str, Attr]):
        self.attrs = attrs
        self._func: Optional[CompiledFunction] = None

    def __str__(self):
        return '{}({})'.format(
            type(self).__name__,
            ', '.join('{}={}'.format(k, v) for k, v in self.attrs.items())
        )

    def __call__(self, *args):
        if self._func is None:
            self._func = self.build()
        self._func(*args)

    def get_cache_dir(self):
        space = hidet.option.get_search_space()
        op_name = type(self).__name__.lower()
        attr_str = '_'.join('{}_{}'.format(k, v) for k, v in self.attrs.items())
        return hidet.utils.cache_dir('hops', 'space_{}'.format(space), op_name, attr_str)

    def build(self) -> CompiledFunction:
        # check the in-memory cache
        if operator_cache.has(self):
            return operator_cache.get(self)

        # check the on-disk cache
        cache_dir = self.get_cache_dir()

        if os.path.exists(os.path.join(cache_dir, 'best.txt')):
            with open(os.path.join(cache_dir, 'best.txt'), 'r') as f:
                best_idx = int(f.read())
                best_module_dir = os.path.join(cache_dir, 'candidates', str(best_idx))
                compiled_module = CompiledModule(best_module_dir)
                func: CompiledFunction = compiled_module['launch']
        else:
            params: List[torch.Tensor] = self.dummy_params()
            ir_modules: List[IRModule] = self.implement()

            # build each ir module
            candidates: List[Tuple[Dict[str, str], CompiledFunction]] = []
            candidates_dir = os.path.join(cache_dir, 'candidates')

            with hidet.option.context():
                # set the arch to the current one, so that the sub-processes will not query the cuda runtime,
                # which will trigger the cuda initialization error
                hidet.option.cuda.arch(hidet.option.cuda.get_arch())

                def build_job(job: Tuple[int, IRModule]):
                    from hidet.transforms.tile.exceptions import SharedMemoryPlanningError
                    try:
                        build_ir_module(
                            ir_module=job[1],
                            output_dir=os.path.join(candidates_dir, str(job[0])),
                            target='cuda'
                        )
                    except SharedMemoryPlanningError:
                        pass

                for _ in tqdm(
                    parallel_imap(
                        func=build_job,
                        jobs=list(enumerate(ir_modules)),
                        num_workers=1 if not hidet.option.get_parallel_build() else None
                    ),
                    desc='Building operator: {}'.format(self),
                    total=len(ir_modules),
                ):
                    pass

            for idx, ir_module in enumerate(ir_modules):
                ir_module_dir = os.path.join(cache_dir, 'candidates', str(idx))
                if compiled_module_exists(ir_module_dir):
                    compiled_module = CompiledModule(ir_module_dir)
                    compiled_function: CompiledFunction = compiled_module['launch']
                    candidates.append((getattr(ir_module, '_tuning_kwargs', {}), compiled_function))

            if len(candidates) == 0:
                raise RuntimeError('No valid candidate for {}.'.format(self))
            elif len(candidates) == 1:
                best_idx = 0
            else:
                # benchmark each compiled function
                latencies: List[float] = []
                for kwargs, func in candidates:
                    latencies.append(hidet.utils.benchmark_func(lambda: func(*params), warmup=3, repeat=5, number=20))

                # generate summary
                table = []
                headers = ['index'] + list(candidates[0][0].keys()) + ['latency']
                for idx, (kwargs, func) in enumerate(candidates):
                    table.append([str(idx)] + list(kwargs.values()) + ['{:.3f}'.format(latencies[idx])])
                table = sorted(table, key=lambda x: float(x[-1]))
                summary = tabulate(table, headers=headers)
                with open(os.path.join(cache_dir, 'summary.txt'), 'w') as f:
                    f.write(summary)

                # write the best
                best_idx = table[0][0]

            with open(os.path.join(cache_dir, 'best.txt'), 'w') as f:
                f.write(str(best_idx))

            func: CompiledFunction = candidates[int(best_idx)][1]

        # set the in-memory cache and return
        operator_cache.set(self, func)
        return func

    def dummy_params(self) -> List[torch.Tensor]:
        raise NotImplementedError()

    def implement(self) -> List[IRModule]:
        raise NotImplementedError()


class OperatorCache:
    def __init__(self):
        self.cache: Dict[Tuple[Type[Operator], Tuple[Tuple[str, Attr], ...]], CompiledFunction] = {}

    def get_key(self, op: Operator) -> Tuple[Type[Operator], Tuple[Tuple[str, Attr], ...]]:
        return type(op), tuple((k, v) for k, v in op.attrs.items())

    def set(self, op: Operator, func: CompiledFunction):
        key = self.get_key(op)
        self.cache[key] = func

    def has(self, op: Operator) -> bool:
        return self.get_key(op) in self.cache

    def get(self, op: Operator) -> CompiledFunction:
        return self.cache[self.get_key(op)]


operator_cache = OperatorCache()
