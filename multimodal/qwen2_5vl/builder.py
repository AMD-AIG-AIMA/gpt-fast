
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import asdict
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
import sys
import os
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mm_config import QwenVisionModelArgs 

def get_qwen_vision_model(name, checkpoint_path, device, dtype):
    config_dc = QwenVisionModelArgs.from_name(name)
    config = Qwen2_5_VLVisionConfig(**asdict(config_dc))
    vision_model = Qwen2_5_VisionTransformerPretrainedModel._from_config(config)
    return vision_model


if __name__ == '__main__':
    try:
        get_qwen_vision_model(
            name="Qwen2.5-VL-3B-Instruct",
            checkpoint_path='checkpoints/Qwen/Qwen2.5-VL-3B-Instruct/vision_modules.pth',
            device='cuda',
            dtype=torch.float16
        )
    except:
        raise ImportError('Failed to import Qwen2.5 VL vision model')
    

