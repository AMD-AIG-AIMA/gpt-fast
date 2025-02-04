from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# print(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from model import Transformer
from torchinfo import summary
import  torch


from evaluate import _load_model
from pathlib import Path
import torch.nn.functional as F
# torch.manual_seed(1234)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float16
torch.set_default_device(device)
hf_input = torch.rand(size=(1,20, 2048), dtype=torch.bfloat16, device=device)
qwen_input = hf_input
# llava_input = torch.rand(size=(1,40, 896), dtype=torch.bfloat16, device='cuda:0')
hf_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype=dtype, device_map=device
)
qwen_checkpoint_path = Path("/home/parfashi/Multimodal-SpecDec/gpt-fast/checkpoints/Qwen/Qwen2.5-VL-3B-Instruct/model.pth")
# llava_checkpoint_path = "/home/parfashi/Multimodal-SpecDec/gpt-fast/checkpoints/lmms-lab/llava-onevision-qwen2-0.5b-si/model.pth"

class Qwen2_5_LanguageModel(torch.nn.Module):  
    def __init__(self, base_model, lm_head):  
        super(Qwen2_5_LanguageModel, self).__init__()  
        self.base_model = base_model  
        self.lm_head = lm_head  
  
    def forward(self, inputs_embeds):  
        output = self.base_model(inputs_embeds=inputs_embeds)  
        logits = self.lm_head(output.last_hidden_state)  
        return logits




gf_qwen_model = _load_model(qwen_checkpoint_path, device, dtype, use_tp=False)
gf_qwen_model.setup_caches(max_batch_size=1, max_seq_length=40)
# gf_llava_model = _load_model(llava_checkpoint_path, 'cuda:0', torch.float16, use_tp=False)

for param in gf_qwen_model.parameters():  
    param.requires_grad = False 
    
 

hf_llm = Qwen2_5_LanguageModel(hf_model.model, hf_model.lm_head)

for param in hf_llm.parameters():  
    param.requires_grad = False  
# summary(hf_llm)
# summary(gf_model)]
err =[]
argmax_diff = []

for i in range(100):
    hf_input = torch.rand(size=(1,40, 2048), dtype=dtype, device=device)
    qwen_input = hf_input
    input_pos = torch.arange(0, hf_input.shape[1], device=device)
    hf_output= hf_llm(hf_input)
    gf_qwen_output = gf_qwen_model(qwen_input, input_pos, embedded=True)

    # print(gf_llava_model(llava_input, input_pos,embedded=True))
    err.append(F.mse_loss(hf_output , gf_qwen_output).item())
    hf_tokens = hf_output.argmax(dim=-1)
    gf_qwen_tokens = gf_qwen_output.argmax(dim=-1)
    
    if (hf_tokens - gf_qwen_tokens).nonzero().numel() != 0:
        argmax_diff.append((hf_tokens - gf_qwen_tokens).nonzero())
        # print(hf_tokens, gf_qwen_tokens)

print('avg: {}, max: {}, min {}, number of times a token was misplaced: {} '.format(sum(err)/len(err), max(err), min(err), len(argmax_diff)))