import pytest

import numpy as np
from numpy import testing
import torch


def test_torch_flatten():
    DEVICE = 'cuda'

    class Model0(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, *args):
            abs_1 = torch.abs(args[0])
            flatten = abs_1.flatten()
            return flatten

    model_0 = Model0()
    output_names_0 = ['v0_0']

    class Model1(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, *args):
            abs_1 = torch.abs(args[0])
            flatten = abs_1.flatten()
            return (abs_1, flatten)

    model_1 = Model1()
    output_names_1 = ['v5_0', 'v0_0']

    data = np.array(
        [
            6,
            3,
            4,
            5,
            4,
            7,
            5,
            5,
            3,
            3,
            4,
            4,
            6,
            3,
            7,
            5,
            4,
            3,
            5,
            6,
            3,
            7,
            7,
            5,
            6,
            6,
            5,
            4,
            5,
            6,
            5,
            3,
            3,
            5,
            4,
            5,
            3,
            7,
            6,
            6,
            6,
            4,
            5,
            3,
            7,
            4,
            4,
            6,
            5,
            3,
            7,
        ],
        dtype=np.int8,
    )
    input_data_0 = [data]

    optmodel_0 = torch.compile(model_0, fullgraph=True, backend='hidet', mode=None)
    model_out_0 = optmodel_0(*[torch.from_numpy(v).to(DEVICE) for v in input_data_0])
    model_out_0 = (
        [v.to(DEVICE).detach() for v in model_out_0]
        if isinstance(model_out_0, tuple)
        else [model_out_0.to(DEVICE).detach()]
    )
    model_out_0 = [v.cpu().resolve_conj().numpy() if v.is_conj() else v.cpu().numpy() for v in model_out_0]
    output_0 = dict(zip(output_names_0, model_out_0))

    input_data_1 = [data]

    optmodel_1 = torch.compile(model_1, fullgraph=True, backend='hidet', mode=None)
    model_out_1 = optmodel_1(*[torch.from_numpy(v).to(DEVICE) for v in input_data_1])
    model_out_1 = (
        [v.to(DEVICE).detach() for v in model_out_1]
        if isinstance(model_out_1, tuple)
        else [model_out_1.to(DEVICE).detach()]
    )
    model_out_1 = [v.cpu().resolve_conj().numpy() if v.is_conj() else v.cpu().numpy() for v in model_out_1]
    output_1 = dict(zip(output_names_1, model_out_1))
    output_name_dict = {'v0_0': 'v0_0'}

    for tensor_name_0, tensor_name_1 in output_name_dict.items():
        testing.assert_allclose(
            output_0[tensor_name_0], output_1[tensor_name_1], rtol=1, err_msg=f'at {tensor_name_0}, {tensor_name_1}'
        )


if __name__ == '__main__':
    pytest.main([__file__])
