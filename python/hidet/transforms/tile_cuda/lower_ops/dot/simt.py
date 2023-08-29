from typing import List, Union

from hidet.ir.expr import Expr
from hidet.ir.tile.ops.dot import SimtDot
from hidet.ir.tile.layout import BlockDotOperandLayout, BlockLayout
from ..registry import TileOpImpl, Buffer, register_impl


@register_impl(SimtDot)
class SimtDotImpl(TileOpImpl):
    def implement(self, op: SimtDot, args: List[Union[Buffer, Expr]], output: Buffer):
        a: Buffer = args[0]
        b: Buffer = args[1]
        c: Buffer = args[2]
        d: Buffer = output
        a_layout, b_layout, c_layout, d_layout = a.layout, b.layout, c.layout, d.layout

        # check layout
        assert (
            isinstance(a_layout, BlockDotOperandLayout)
            and isinstance(b_layout, BlockDotOperandLayout)
            and isinstance(c_layout, BlockLayout)
            and isinstance(d_layout, BlockLayout)
            and a_layout.parent == c_layout and b_layout.parent == c_layout and c_layout == d_layout
        )

        k_size = a.shape[1]
        with self.for_grid(d.local_shape) as d_indices:
            self.buffer_store(d.var, d_indices, c[d_indices])

        with self.for_range(k_size) as k:
            with self.for_grid(d.local_shape) as d_indices:
                a_indices = [d_indices[0], k]
                b_indices = [k, d_indices[1]]
                self.buffer_store(d.var, d_indices, d[d_indices] + a[a_indices] * b[b_indices])

