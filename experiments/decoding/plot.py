# %%
from typing import List
from decoding_config import DecodingSampleConfig, BenchResult
from torch_gpt2 import get_numbers_gpt2
from torch_llama import get_numbers_llama7b
from matplotlib import pyplot as plt
import numpy as np

configs = [
    DecodingSampleConfig(q_seq_len=32, kv_seq_len=512, batch_size=1),
    DecodingSampleConfig(q_seq_len=4, kv_seq_len=512, batch_size=1),
    DecodingSampleConfig(q_seq_len=1, kv_seq_len=512, batch_size=1),
]


llama_numbers = get_numbers_llama7b(configs)
gpt2_numbers = get_numbers_gpt2(configs)
gpt2_medium_numbers = get_numbers_gpt2(configs, 'gpt2-medium')

def join_lists(*args):
    assert all(len(args[0]) == len(a) for a in args)
    lst = [[] for _ in range(len(args[0]))]
    for i in range(len(args[0])):
        for a in args:
            lst[i].extend(a[i])
    return lst

numbers: List[List[BenchResult]] = join_lists(llama_numbers, gpt2_numbers, gpt2_medium_numbers)

for n, config in zip(numbers, configs):
    models = set()
    frameworks = set()
    for k in n:
        models.add(k.model)
        frameworks.add(k.framework)
    models = list(models)
    models.sort()
    frameworks = list(frameworks)
    frameworks.sort()
    x = np.arange(len(models))
    width = 0.15
    multiplier = 0

    fix, ax = plt.subplots(layout='constrained')
    for f in frameworks:
        same_framework = [k for k in n if k.framework == f]
        same_framework.sort(key=lambda x: x.model) # sort by model
        model_times = [v.median for v in same_framework]
        offset = width * multiplier
        rects = ax.bar(x + offset, model_times, width, label=f)
        ax.bar_label(rects, padding=3)
        multiplier += 1
    
    ax.set_ylabel('time (ms)')
    ax.set_title('Decoding time by model and framework, q_seq_len={}, kv_seq_len={}, batch_size={}'.format(config.q_seq_len, config.kv_seq_len, config.batch_size))
    ax.set_xticks(x + width, models)
    ax.legend()
    plt.show()

