import hidet
from hidet.testing import capture_stdout


def test_debug_print():
    from hidet.lang import attrs
    from hidet.lang import tile as ti

    with hidet.script_module() as script_module:
        @hidet.script
        def use_arange():
            attrs.func_kind = 'cuda_tile'
            attrs.cuda.grid_dim = 1
            attrs.cuda.block_dim = 128

            a = ti.arange(0, 16)
            ti.debug_print(a)

            b = ti.full(1, [4, 4])
            ti.debug_print(b)

            c = ti.full(1, [4, 4, 4])
            ti.debug_print(c)

            d = ti.full(1, [16, 16])
            ti.debug_print(d)

    expected_output = """
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
[[1, 1, 1, 1]
 [1, 1, 1, 1]
 [1, 1, 1, 1]
 [1, 1, 1, 1]]
[[1, 1, 1, 1]
 [1, 1, 1, 1]
 [1, 1, 1, 1]
 [1, 1, 1, 1]]

[[1, 1, 1, 1]
 [1, 1, 1, 1]
 [1, 1, 1, 1]
 [1, 1, 1, 1]]

[[1, 1, 1, 1]
 [1, 1, 1, 1]
 [1, 1, 1, 1]
 [1, 1, 1, 1]]

[[1, 1, 1, 1]
 [1, 1, 1, 1]
 [1, 1, 1, 1]
 [1, 1, 1, 1]]

[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
    """.strip()
    func = script_module.build()
    hidet.cuda.synchronize()
    with capture_stdout() as captured:
        func()
        hidet.cuda.synchronize()

    actual_output: str = str(captured).strip()
    assert actual_output == expected_output
