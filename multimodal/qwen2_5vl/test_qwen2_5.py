from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# print(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from model import Transformer, precompute_freqs_cis_for_qwen2_5, flatten_freqs
from torchinfo import summary
import  torch
from multimodal.qwen2_5vl.preprocessing import get_qwen2_5vl_message_template, get_processor, process_chat_template, process_visual_inputs, set_model_inputs, prepare_input_embeds, get_position_ids
from multimodal.qwen2_5vl.builder import get_qwen_vision_model
from multimodal.mm_config import QwenVisionModelArgs


from evaluate import _load_model
from pathlib import Path
import torch.nn.functional as F
# torch.manual_seed(1234)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float
torch.set_default_device(device)

message = get_qwen2_5vl_message_template()
processor = get_processor()
text = process_chat_template(processor, message)
inputs = process_visual_inputs(processor, text, device=device, dtype=dtype)
model_inputs = set_model_inputs(**inputs)


gf_vision_model = get_qwen_vision_model(
            name="Qwen2.5-VL-3B-Instruct",
            checkpoint_path='checkpoints/Qwen/Qwen2.5-VL-3B-Instruct/vision_modules.pth',
            device=device,
            dtype=dtype
            )

# summary(gf_vision_model)


# llava_input = torch.rand(size=(1,40, 896), dtype=torch.bfloat16, device='cuda:0')
hf_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype=dtype, device_map=device
)
qwen_checkpoint_path = Path("/home/parfashi/Multimodal-SpecDec/gpt-fast/checkpoints/Qwen/Qwen2.5-VL-3B-Instruct/model.pth")
# llava_checkpoint_path = "/home/parfashi/Multimodal-SpecDec/gpt-fast/checkpoints/lmms-lab/llava-onevision-qwen2-0.5b-si/model.pth"

gf_qwen_model = _load_model(qwen_checkpoint_path, device, dtype, use_tp=False)
# gf_llava_model = _load_model(llava_checkpoint_path, 'cuda:0', torch.float16, use_tp=False)



for param in gf_vision_model.parameters():  
    param.requires_grad = False 
    
inputs_embeds = prepare_input_embeds(model_inputs, gf_qwen_model, gf_vision_model, 151655,device, dtype)
mm_config = QwenVisionModelArgs.from_name('Qwen2.5-VL-3B-Instruct')

position_ids, _ = get_position_ids(model_inputs,mm_config)

# print(position_ids, position_ids.shape)

# blegh
gf_qwen_model.setup_caches(max_batch_size=1, max_seq_length=inputs_embeds.shape[1]*2)

prefill_freqs_cis = precompute_freqs_cis_for_qwen2_5(gf_qwen_model.config.block_size, gf_qwen_model.config.dim // gf_qwen_model.config.n_head, gf_qwen_model.config.rope_base, dtype, gf_qwen_model.config.rope_scaling, position_ids, [16, 24, 24])
# print(prefill_freqs_cis[...,0].shape)

gf_qwen_model.freqs_cis = prefill_freqs_cis
#TODO: write a set_freqs_cis function? No, add position_ids to setup_cache


for param in gf_qwen_model.parameters():  
    param.requires_grad = False 
    
 
for param in hf_model.parameters():  
    param.requires_grad = False  
# summary(hf_llm)
# summary(gf_model)]
max_new_tokens = 16

input_pos = torch.arange(0, inputs_embeds.shape[1], device=device)
try:
    hf_output = hf_model.generate(**inputs, max_new_tokens=max_new_tokens, return_dict_in_generate=True, output_logits =True)
except:
    pass
# print(hf_output)

gf_qwen_tokens = torch.tensor([])
gf_qwen_logits = torch.tensor([])
for i in range(max_new_tokens):
    gf_qwen_output = gf_qwen_model(inputs_embeds, input_pos, embedded=True)
    gf_qwen_logit = gf_qwen_output[:,-1,:].unsqueeze(1)
    gf_qwen_token = gf_qwen_logit.argmax(dim=-1)
    gf_qwen_embedding = gf_qwen_model.tok_embeddings(gf_qwen_token)
    
    inputs_embeds = torch.cat([inputs_embeds, gf_qwen_embedding], dim=1)
    input_pos = torch.cat([input_pos, torch.tensor([input_pos[-1]+1])])
    gf_qwen_tokens = torch.cat([gf_qwen_tokens, gf_qwen_token], dim=1)
    gf_qwen_logits = torch.cat([gf_qwen_logits, gf_qwen_logit], dim=1)

hf_logits = torch.stack(hf_output.logits, dim=1)
hf_tokens = hf_output.sequences[0,-max_new_tokens:]
# generated_ids_trimmed = [
#     out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
# ]
# output_text = processor.batch_decode(
#     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
# )
mse_error = F.mse_loss(gf_qwen_logits,hf_logits)


print('hf_logits: {}, gf_qwen_logits: {}, MSE Error: {}'.format(hf_logits, gf_qwen_logits, mse_error))


print('Is new_token the same?', (hf_tokens - gf_qwen_tokens).nonzero().numel()==0)
