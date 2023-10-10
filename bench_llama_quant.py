# %%
from hidet.testing.models.llama import generate, convert_model
from hidet.runtime.storage import current_memory_pool

from hidet.testing.models.llama import *
hidet.option.cache_dir('./llama_quant_benchmark/outs/cache')

def get_compiled_model(name='decapoda-research/llama-7b-hf', device='cuda', opt=False):
    tok = LlamaTokenizer.from_pretrained(name)

    with torch.device("cuda"):  # reduce the time to load the model
        model = hfLm.from_pretrained(name, torch_dtype=torch.float16)

    model.cpu()
    torch.cuda.empty_cache()

    config = model.config

    model: nn.Module = convert_model(model, device=device)

    flow_graph = build_flow_graph(model, device=device)

    if opt:
        with hidet.graph.PassContext() as ctx:
            ctx.set_precision('int8')
            # ctx.set_use_attention(True)
            ctx.set_parallel_k(search=True)
            ctx.reduce_cuda_compile_mem()
            flow_graph = hidet.graph.optimize(flow_graph)
    print(flow_graph)
    compiled = flow_graph.build(space=2)
    return compiled, config, tok

hidet.option.search_space(2)
import time
import random
from matplotlib import pyplot as plt

device = 'cuda'
dtype = hidet.float16

model, config, tokenizer = get_compiled_model(device='cuda', opt=True)


generate('In the beginning was the Word.', model, tokenizer, config, num_tokens=12)
# %%
position_ids = hidet.arange(0, config.max_position_embeddings, dtype=hidet.int32, device=device).unsqueeze(0)
num_tokens = 128
num_prefill = 0
make_past = lambda: hidet.zeros(
    [1, config.num_key_value_heads, num_prefill, config.hidden_size // config.num_key_value_heads], device=device, dtype=dtype
)
past_keys_values = [make_past() for _ in range(config.num_hidden_layers * 2)]

inputs = [hidet.asarray([[random.randint(0, 32000)]], dtype=hidet.int32, device=device) for _ in range(num_tokens)]

outputs = []
times = []
org_t = time.time()
for i in range(num_tokens):
    input_ids = inputs[i]
    t = time.time()
    y = model(input_ids, position_ids, *past_keys_values)
    times.append(time.time() - t)
    # input_ids = y[0][:, -1:].to(dtype=hidet.int32)
    # outputs.append(input_ids[0, -1].item())
    past_keys_values = y[1:]
    print(past_keys_values[0].shape)
org_t = time.time() - org_t

print(f'org_t: {org_t}')
print(f'avg_t: {sum(times) / len(times)}')

plt.plot(times)
plt.show()


# fp16 - 128 tokens
# org_t: 3.048099994659424
# avg_t: 0.0044784750789403915

# fp16 - 128 tokens with parallel_k
# org_t: 2.862935781478882
# avg_t: 0.003960812464356422

# int8 - 128 tokens
# org_t: 2.2229115962982178
# avg_t: 0.0038476772606372833

# int8 - 128 tokens with parallel_k 'search'
# org_t: 2.125953435897827
# avg_t: 0.004375133663415909

# int8 - 128 tokens with parallel_k=4
# org_t: 1.5948965549468994
# avg_t: 0.012420829385519028

# int8 - 128 tokens with parallel_k=4 and flash attention
# org_t: 1.9178423881530762
# avg_t: 0.004179380834102631

# int8 - prefill 128, decode 128, batch size 1, k_parks=4, flash-attn
# org_t: 2.031648635864258
# avg_t: 0.01493494026362896

# int8 - prefill 128, decode 128, batch size 1, k_parks=4
# org_t: 1.767362117767334
# avg_t: 0.013012891635298729

# ---torch---
# decode 128 tokens with 0 prefill, batch size 1
# torch-llama: 3.195237979888916
# torch-compile-llama: 2.881284008026123

# 128 prefill, decode 128, batch size 1
# torch-llama: 3.2373869705200193
# torch-compile-llama: 2.905135974884033
