
def program_id():
    from hidet.ir.primitives.cuda import blockIdx
    return blockIdx.x


def num_programs():
    from hidet.ir.primitives.cuda import blockDim
    return blockDim.x
