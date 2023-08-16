

def flatten_indices():
    pass


def unflatten_indices(global_index, shape):
    s = 1
    indices = []
    for i, extent in enumerate(shape):
        indices.append(global_index // s % extent)
        s *= extent
    return indices

