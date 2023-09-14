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
from typing import Union
from hidet.ir.type import DataType, PointerType
from hidet.ir.stmt import BlackBoxStmt


def printf(format_string, *args):
    """
    usage:
    printf("%d %d\n", expr_1, expr_2)
    """
    format_string = format_string.replace('\n', '\\n')
    if len(args) > 0:
        arg_string = ', '.join(['{}'] * len(args))
        template_string = f'printf("{format_string}", {arg_string});'
    else:
        template_string = f'printf("{format_string}");'
    # if '\n' in format_string:
    #     raise ValueError('Please use printf(r"...\\n") instead of printf("...\\n").')
    return BlackBoxStmt(template_string, *args)


def format_string_from_dtype(dtype: Union[DataType, PointerType]) -> str:
    from hidet.ir.dtypes import float32, float64, int8, int16, int32, int64, uint8, uint16, uint32, uint64

    if isinstance(dtype, PointerType):
        return '%p'
    elif dtype in [float32, float64]:
        return '%.2f'
    elif dtype in [int8, int16, int32]:
        return '%d'
    elif dtype in [uint8, uint16, uint32]:
        return '%u'
    elif dtype is int64:
        return '%lld'
    elif dtype is uint64:
        return '%llu'
    else:
        raise ValueError(
            'Can not use printf to print "{}" directly, consider converting it to other '
            'more standard type first.'.format(dtype)
        )
