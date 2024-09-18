import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='torch.utils._pytree')
warnings.filterwarnings('ignore', category=FutureWarning, module='huggingface_hub.file_download')


from generate import _load_model, model_forward, prefill
import argparse
import time
import torch
from tqdm import tqdm
from pathlib import Path

# torch._inductor.config.coordinate_descent_tuning = True
# torch._inductor.config.triton.unique_kernel_names = True
# torch._inductor.config.fx_graph_cache = True # Experimental feature to reduce compilation times, will be on by default in future
# torch._dynamo.config.cache_size_limit=64
# torch._inductor.config.max_autotune=True
# torch._inductor.config.max_autotune_gemm_backends="TRITON"

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', type=str, default="../gpt-fast/checkpoints/codellama/CodeLlama-7b-Python-hf/model.pth",help='model')
parser.add_argument('--T', type=int, default=20, help='repeat times')
parser.add_argument('--B', type=int, default=1, help='batch size')
parser.add_argument('--P', type=int, default=128, help='prefix length')
parser.add_argument('--M', type=int, default=1536, help='max length')
parser.add_argument('--D', type=int, default=1, help='dec length')
parser.add_argument('--W', type=int, default=0, help='Warmup Steps')
# parser.add_argument('--E', action='store_true', default=False, help='use ea model')
args = parser.parse_args()
print(args)
PREFIX_LEN = args.P
MAX_LEN = args.M
DEC_LEN = args.D
# MODEL_NAME = args.base-model-path
DTYPE = torch.bfloat16
DEVICE = "cuda:0"
T = args.T
WARM_UP = 10
tree_choices =  [[0],[0,0]]
# Set the number of warm-up steps

model = _load_model(Path(args.checkpoint_path), DEVICE, DTYPE, False)

global model_forward, logits_to_prob
model_forward = torch.compile(model_forward, mode="max-autotune", fullgraph=True)
# prefill = torch.compile(prefill, fullgraph=True, dynamic=True,  mode="max-autotune")
model.eval()
# pytorch_profiler.start()
# Capture the CUDA graph
input_ids = torch.randint(low=3, high=30000, size=(1, PREFIX_LEN), device=DEVICE)
# Avoid modifying the input_ids in-place
input_ids = input_ids.clone()
max_block_size = MAX_LEN
input_pos = torch.arange(0, input_ids.shape[1], device=input_ids.device)

with torch.device(DEVICE):
    model.setup_caches(max_batch_size=args.B, max_seq_length=MAX_LEN)


with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=False, enable_math=True):
    next_token = prefill(model, input_ids.view(args.B, -1), input_pos).clone()


torch.cuda.synchronize()
times = []
input_id = torch.randint(low=3, high=30000, size=(1, DEC_LEN), device=DEVICE)
input_pos=torch.arange(input_ids.shape[1], input_ids.shape[1]+DEC_LEN, device=input_ids.device)
for i in tqdm(range(T)):
    #with Timer("naive base"):
    t1 = time.time()
    with torch.no_grad():
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True): # Actually better for Inductor to codegen attention here
            model_forward(model, input_id, input_pos)
    torch.cuda.synchronize()
    times.append(time.time()-t1)

t = sum(times[args.W:])
# print("Max Length :{}, Decode Length :{}, Prefix Length :{}, inference time:{}s".format(MAX_LEN, DEC_LEN, PREFIX_LEN, t/ (T-args.W)))
print("{},    {}".format(DEC_LEN, t/ (T-args.W)))



