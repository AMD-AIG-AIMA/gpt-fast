from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from torchinfo import summary
import  torch
from pathlib import Path
import torch.nn.functional as F
from builder import get_qwen_vision_model
from qwen_vl_utils import process_vision_info

# default processer

# torch.manual_seed(1234)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float
torch.set_default_device(device)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")


messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)


image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to(device)


# llava_input = torch.rand(size=(1,40, 896), dtype=torch.bfloat16, device='cuda:0')
hf_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype=dtype, device_map=device
)
qwen_checkpoint_path = Path("/home/parfashi/Multimodal-SpecDec/gpt-fast/checkpoints/Qwen/Qwen2.5-VL-3B-Instruct/vision_modules.pth")
# llava_checkpoint_path = "/home/parfashi/Multimodal-SpecDec/gpt-fast/checkpoints/lmms-lab/llava-onevision-qwen2-0.5b-si/model.pth"



gf_qwen_model = get_qwen_vision_model("Qwen2.5-VL-3B-Instruct", qwen_checkpoint_path, device, dtype)
# gf_qwen_model.setup_caches(max_batch_size=1, max_seq_length=40)
# gf_llava_model = _load_model(llava_checkpoint_path, 'cuda:0', torch.float16, use_tp=False)

gf_qwen_model.requires_grad_(False) 
hf_model.requires_grad_(False) 

hf_vision = hf_model.visual


# summary(hf_llm)
# summary(gf_model)]
err =[]
image_grid_thw = inputs.image_grid_thw
for i in range(5):
    hf_input = (torch.rand(size=inputs.pixel_values.shape, dtype=dtype, device=device) - 0.5)*2.
    qwen_input = hf_input
    hf_output= hf_vision(hf_input, grid_thw=image_grid_thw)
    gf_qwen_output = gf_qwen_model(hf_input, grid_thw=image_grid_thw)

    # print(gf_llava_model(llava_input, input_pos,embedded=True))
    err.append(F.mse_loss(hf_output , gf_qwen_output).item())


print(sum(err)/len(err), max(err), min(err))