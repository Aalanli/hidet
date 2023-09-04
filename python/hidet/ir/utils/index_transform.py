# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List, Optional
from ..expr import Expr, convert


def index_serialize(indices: List[Expr], shape: List[int], ranks: Optional[List[int]] = None) -> Expr:
    """
    Serialize the logical indices in a tensor with given shape to a linear index in linear memory space.
    The ranks indices the rank of each dimension of the tensor.
    ranks = [0, 1, 2, 3] of shape[3, 4, 5, 6] indicates that the last dimension is the fastest changing dimension.
    ranks = [3, 2, 1, 0] of shape[3, 4, 5, 6] indicates that the first dimension is the fastest changing dimension.
    More generally, ranks = [r_0, r_1, ..., r_n] of shape[n_0, n_1, ..., n_n] indicates that rank r_n is the fastest
    changing dimension, rank r_{n-1} is the second fastest changing dimension, and so on.
    """
    if len(shape) == 0:
        return convert(0)
    if ranks is None:
        ranks = list(range(len(shape)))
    scalar_index: Expr = convert(0)
    acc = 1
    for rank in reversed(ranks):
        idx_value = indices[rank]
        extent = shape[rank]
        scalar_index += idx_value * acc
        acc *= extent
    return scalar_index


def index_deserialize(scalar_index: Expr, shape: List[int], ranks: Optional[List[int]] = None) -> List[Expr]:
    if len(shape) == 0:
        return []
    if ranks is None:
        ranks = list(range(len(shape)))
    indices = []
    acc = 1
    for idx, rank in enumerate(reversed(ranks)):
        extent = shape[rank]
        if idx < len(shape) - 1:
            indices.append(scalar_index // acc % extent)
        else:
            indices.append(scalar_index // acc)
        acc *= extent
    return list(reversed(indices))
