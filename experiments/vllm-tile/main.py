import time
import hidet
from vllm import LLM, SamplingParams
from hllm import register_hidet_implementations, revert_implementations
import nvtx

register_hidet_implementations()

# Create an LLM.
with nvtx.annotate('create LLM'):
    # llm = LLM(model="facebook/opt-125m")
    llm = LLM(model="lmsys/vicuna-7b-v1.5")
# llm = LLM(model="hidet-llama")
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.

fmt = '{:>10} {:>10} {:>12} {:>15}'
print(fmt.format('requests', 'tokens', 'latency', 'throughput'))

for num_requests in [16]:
    for max_tokens in [1024]:
        # Sample prompts.
        prompts = [""] * num_requests
        # Create a sampling params object.
        sampling_params = SamplingParams(max_tokens=max_tokens, ignore_eos=True)

        hidet.cuda.synchronize()
        t1 = time.time()
        with nvtx.annotate('generate {} {}'.format(num_requests, max_tokens)):
            outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        hidet.cuda.synchronize()
        t2 = time.time()
        latency = t2 - t1
        print(fmt.format(
            num_requests,
            max_tokens,
            '{:.3f} secs'.format(latency),
            '{:.3f} reqs/s'.format(num_requests / latency)
        ))
