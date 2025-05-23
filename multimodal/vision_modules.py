from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union, Dict, Type

import math

import torch
import torch.nn as nn
from PIL import Image
from dataclasses import dataclass
from pathlib import Path
import re
from dataclasses import asdict


from multimodal.llava.preprocessing import (process_images, 
                                            tokenizer_image_token, 
                                            get_image_features, 
                                            embed_multimodal_tokens, 
                                            IMAGE_TOKEN_INDEX)
from multimodal.qwen2_5vl.preprocessing import process_prompt_for_qwen2_5vl, get_processor, prepare_input_embeds
from multimodal.mm_config import QwenVisionModelArgs 

@dataclass 
class VisionModelOutput:
    """Common output format for vision models"""
    input_ids: torch.LongTensor
    embeddings: torch.FloatTensor

class VisionModule(ABC, nn.Module):
    """Abstract base class for vision modules.
    
    All vision module implementations should inherit from this class
    and implement the required abstract methods.
    """
    _registry: Dict[str, Type['VisionModule']] = {}
    
    def __init__(self, 
                 config,
                 dtype: torch.dtype = torch.float16,
                 device: Optional[Union[str, torch.device]] = None):
        super().__init__()
        ABC.__init__(self)
        self.config = config
        self.dtype = dtype

        if device is not None:
            self._device = torch.device(device)
        else:
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        self._is_loaded = False
        self.to(self._device)
        self.image_token = None
    

    @abstractmethod
    def preprocess_images(self, images: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
        """Preprocess input images into model-ready format.
        
        Args:
            images: Single image or list of images to preprocess
            
        Returns:
            Tensor of preprocessed images
        """
        pass

    @abstractmethod
    def forward(self, images: torch.Tensor) -> VisionModelOutput:
        """Run forward pass of the model.
        
        Args:
            images: Preprocessed image tensor
            
        Returns:
            Model outputs including embeddings and hidden states
        """
        pass

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load model weights from checkpoint"""
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=self._device, weights_only=True)
        self.load_state_dict(state_dict)

    def eval_mode(self):
        self.requires_grad_(False) 
        self.eval()
        
    @property 
    def is_loaded(self) -> bool:
        """Whether the model weights are loaded"""
        return self._is_loaded
        
    @property
    def device(self) -> torch.device:
        """Device the model parameters are on"""
        return self._device

    @device.setter 
    def device(self, device: Union[str, torch.device]):
        """Set model device"""
        self._device = torch.device(device)
        self.to(self._device)

    @classmethod
    def register(cls, model_name: str):
        def register_model_cls(model_cls):
            cls._registry[model_name] = model_cls
            return model_cls
        return register_model_cls
    
    @classmethod
    def from_name(cls, model_name: str, **kwargs):
        mapped_name = [name for name in cls._registry if name.lower() in model_name.lower()]
        if len(mapped_name) == 0:
            raise ValueError(f"Model {model_name} not found in registry. Available models: {list(cls._registry.keys())}")
        if len(mapped_name) > 1:
            mapped_name.sort(key=len, reverse=True)
            assert len(mapped_name[0]) != len(mapped_name[1]), model_name # make sure only one 'best' match
        return cls._registry[mapped_name[0]](**kwargs)

def prune_mm_tokens(embedded, prune_method=None, prune_ratio=0.0, dim=1):
        """Prune Multimodal tokens from the embedded tensor.
        
        Args:
            embedded: Tensor containing multimodal token embeddings
            prune_method: Method for pruning (None, 'random', 'structured')
            prune_ratio: Ratio of tokens to prune (0.0 to 1.0)
            dim: Dimension along which to prune tokens (default: 1)
            
        Returns:
            Pruned embedded tensor
        """
        if prune_method is None or prune_ratio <= 0.0:
            return embedded
        
        # Make a copy to avoid modifying the original
        pruned = embedded.clone()
        
        # Get dimensions
        seq_length = pruned.size(dim)
        
        keep_ratio = 1.0 - prune_ratio
        num_keep = int(seq_length * keep_ratio)
        
        if prune_method == 'random':
            # Randomly select tokens to keep
            indices = torch.randperm(seq_length, device=pruned.device)[:num_keep].to(pruned.device)
            indices, _ = torch.sort(indices)  # Sort to maintain sequence order
            
            pruned = torch.index_select(pruned, dim, indices)
                
        elif prune_method == 'structured':
            # Keep a structured pattern (e.g., 3 out of every 4 tokens)
            stride = int(1.0 / keep_ratio)
            to_remove = torch.arange(stride-1, seq_length, stride, device=pruned.device)
            mask = torch.ones(seq_length, device=pruned.device, dtype=torch.bool)
            mask[to_remove] = False

            # Select the indices where mask is True
            keep_indices = torch.where(mask)[0]
            pruned = torch.index_select(pruned, dim, keep_indices)
        
        return pruned
    

@VisionModule.register("llava")
class LlavaVisionModule(VisionModule):
    def __init__(
        self,
        config,
        checkpoint_path: Optional[Path] = None,
        dtype: torch.dtype = torch.float16,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__(config, dtype, device)
        from multimodal.llava.builder import build_vision_tower, build_vision_resampler, build_vision_projector
        self.delay_load = getattr(config, "delay_load", False)
        self.vision_tower = build_vision_tower(config, delay_load=self.delay_load)
        self.vision_resampler = build_vision_resampler(config, vision_tower=self.vision_tower)
        self.mm_projector = build_vision_projector(config, vision_cfg=self.vision_tower)
        self.vision_tower = self.vision_tower.to(dtype=dtype, device=device)
        self.vision_resampler = self.vision_resampler.to(dtype=dtype, device=device)
        self.mm_projector = self.mm_projector.to(dtype=dtype, device=device)
        self.image_newline = None
        if "unpad" in getattr(config, "mm_patch_merge_type", ""):
            self.image_newline = nn.Parameter(torch.empty(config.hidden_size, dtype=dtype))
        if checkpoint_path is not None:
            self.load_checkpoint(Path(checkpoint_path))
        self.image_token = "<image>"

    def preprocess_images(self, images):
        pass

    def forward(self,
        prompt,
        tokenizer,
        images,
        embed_tokens,
        prune_method=None,
        prune_ratio=0.0,
        ):
        image_tensor = process_images(images, self.vision_tower.image_processor, self.config)
        image_tensor = [_image.to(dtype=self.dtype, device=self._device) for _image in image_tensor]
        input_ids = tokenizer_image_token(
                            prompt, 
                            tokenizer,
                            IMAGE_TOKEN_INDEX,
                            return_tensors="pt"
                        ).unsqueeze(0).to(self._device)
        image_features = get_image_features(input_ids,
                                    image_tensor,
                                    config=self.config,
                                    vision_tower=self.vision_tower,
                                    mm_projector=self.mm_projector,
                                    modalities=["image"],
                                    image_newline=self.image_newline, 
                                    image_sizes=[img.size for img in images])
        if prune_method is not None and prune_ratio > 0.0:
            image_features = [prune_mm_tokens(image_features, prune_method, prune_ratio, dim=0) for image_features in image_features]
        embeds = embed_multimodal_tokens(input_ids, None, None, None, 
                                        image_features, 
                                        self.config, 
                                        embed_tokens, 
                                        self._device, 
                                        modalities=["image"])
        return input_ids, embeds.to(dtype=self.dtype)
    
@VisionModule.register("Qwen2.5")
class Qwen2_5VisionModule(VisionModule):
    def __init__(
          self,
          config,
          checkpoint_path: Optional[Path] = None,
          dtype: torch.dtype = torch.float16,
          device: Optional[Union[str, torch.device]] = None,
        ):
        super().__init__(config, dtype, device)
        try:
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel
            from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
        except:
            raise ImportError(f"Qwen2.5 VL series of models are not supported with current version of transformers. Update transformers to the latest version.")
        
        name = Path(checkpoint_path).parent.name
        self._processor_id = Path(*Path(checkpoint_path).parent.parts[-2:])
        config_dc = QwenVisionModelArgs.from_name(name)
        vision_config = Qwen2_5_VLVisionConfig(**asdict(config_dc))
        self.vision_model = Qwen2_5_VisionTransformerPretrainedModel._from_config(vision_config)
        if checkpoint_path is not None:
             self.vision_model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        self.vision_model = self.vision_model.to(device=self._device, dtype=self.dtype)
        self.image_factor = vision_config.image_factor[0]
        self.min_pixels = vision_config.min_pixels[0]
        self.max_pixels = vision_config.max_pixels[0]
        self.max_ratio = vision_config.max_ratio
        self.image_token = "<image>"
        
    def preprocess_images(self, images, prune_ratio):
        processed_images = []
        for image in images:
            processed_images.append(self.resize(image,prune_ratio))
        return processed_images
    def forward(self,
        prompt,
        tokenizer,
        images,
        embed_tokens,
        prune_method=None,
        prune_ratio=0.0,
        ):
        if prune_method=='random':
            raise NotImplementedError('Random pruning is not implemented for Qwen2.5 VL series of models.')

        prompt = process_prompt_for_qwen2_5vl(prompt)
        processor = get_processor(self._processor_id)
        images = self.preprocess_images(images, prune_ratio)
        inputs = processor(
            text=[prompt],
            images=images,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device=self._device, dtype=self.dtype)
        
        inputs_embeds = prepare_input_embeds(inputs['input_ids'], inputs['pixel_values'], inputs['image_grid_thw'], embed_tokens, self.vision_model,151655,self._device,self.dtype)

        return inputs, inputs_embeds
    
    def resize(self, image, prune_ratio):
        resized_height, resized_width = self.get_resize_dims(image.height, image.width)
        if prune_ratio > 0.0:
            resized_height = int(resized_height*math.sqrt(1-prune_ratio))
            resized_width = int(resized_width*math.sqrt(1-prune_ratio))
        return image.resize((resized_width, resized_height))
    
    def get_resize_dims(self, height: int, width: int):
        if max(height, width) / min(height, width) > self.max_ratio:
            raise ValueError(
                f"absolute aspect ratio must be smaller than {self.max_ratio}, got {max(height, width) / min(height, width)}"
            )
        h_bar = max(self.image_factor, self.round_by_factor(height, self.image_factor))
        w_bar = max(self.image_factor, self.round_by_factor(width, self.image_factor))
        if h_bar * w_bar > self.max_pixels:
            beta = math.sqrt((height * width) / self.max_pixels)
            h_bar = self.floor_by_factor(height / beta, self.image_factor)
            w_bar = self.floor_by_factor(width / beta, self.image_factor)
        elif h_bar * w_bar < self.min_pixels:
            beta = math.sqrt(self.min_pixels / (height * width))
            h_bar = self.ceil_by_factor(height * beta, self.image_factor)
            w_bar = self.ceil_by_factor(width * beta, self.image_factor)
        return h_bar, w_bar
    
    def round_by_factor(self, x: int, factor: int):
        return round(x / factor) * factor
    
    def ceil_by_factor(self, x: int, factor: int):
        return int(math.ceil(x / factor) * factor)
    
    def floor_by_factor(self, x: int, factor: int):
        return int(math.floor(x / factor) * factor)
      

@VisionModule.register("llama")
class LlamaVisionModule(VisionModule):
    def __init__(
          self,
          config,
          checkpoint_path: Optional[Path] = None,
          dtype: torch.dtype = torch.float16,
          device: Optional[Union[str, torch.device]] = None):
        super().__init__(config, dtype, device)
        from transformers import MllamaVisionModel, MllamaProcessor
        self.vision_model = MllamaVisionModel._from_config(config).to(dtype=dtype, device=device)
        # self.vision_model = MllamaVisionModel.from_pretrained(str(checkpoint_path.parent))
        self.multi_modal_projector = nn.Linear(
            config.vision_output_dim,
            config.text_hidden_size,
            bias=True,
            dtype=dtype, device=device
        )
        
        self.processor = MllamaProcessor.from_pretrained(str(checkpoint_path.parent))
        if checkpoint_path is not None:
            self.load_checkpoint(Path(checkpoint_path))
        self.image_token = "<|image|>"

    def preprocess_images(self, images):
        pass

    def forward(self,
        prompt,
        tokenizer,
        images,
        embed_tokens,
        prune_method=None,
        prune_ratio=0.0,
        ):
        # Replace <image>, <image1>, <image2>, ... with <|image|>
        prompt = re.sub(r'<image\s*\d*>', '<|image|>', prompt)
        if len(images) == 0:
            inputs = self.processor(text=prompt, return_tensors="pt").to(self._device)
            cross_attention_states = None
            self.cross_attention_masks = {}
        else:
            inputs = self.processor(
                images=images,
                text=prompt,
                return_tensors="pt"
            ).to(self._device)
            vision_outputs = self.vision_model(
                    pixel_values=inputs.pixel_values,
                    aspect_ratio_ids=inputs.aspect_ratio_ids,
                    aspect_ratio_mask=inputs.aspect_ratio_mask,
                    output_hidden_states=False,
                    output_attentions=False,
                    return_dict=True,
            )
            cross_attention_states = vision_outputs[0]
            cross_attention_states = self.multi_modal_projector(cross_attention_states).reshape(
                -1, cross_attention_states.shape[-2], self.config.text_hidden_size
            )
            if prune_method is not None and prune_ratio > 0.0:
                cross_attention_states = prune_mm_tokens(cross_attention_states, prune_method, prune_ratio)

            cross_attention_mask, out_mask = self._prepare_cross_attention_mask(
                inputs.cross_attention_mask,
                num_vision_tokens=cross_attention_states.shape[1],
                dtype=self.dtype,
            )
            self.cross_attention_masks = {'cross_attention_mask': cross_attention_mask, 'cross_attention_mask_out': out_mask}
        return inputs.input_ids, cross_attention_states

    def _prepare_cross_attention_mask(
            self,
            cross_attention_mask: torch.Tensor,
            num_vision_tokens: int,
            dtype: str,
        ):
        # reshape so it can be used by attn module
        batch_size, text_total_length, *_ = cross_attention_mask.shape
        cross_attention_mask = cross_attention_mask.repeat_interleave(num_vision_tokens, dim=3)
        cross_attention_mask = cross_attention_mask.view(batch_size, text_total_length, -1)
        cross_attention_mask = cross_attention_mask.unsqueeze(1)

        # invert the mask
        inverted_cross_attn_mask = (1.0 - cross_attention_mask).to(dtype)
        cross_attention_mask = inverted_cross_attn_mask.masked_fill(
            inverted_cross_attn_mask.to(torch.bool), torch.finfo(dtype).min
        )

        # apply full-row bias, which return 4D tensor of shape [B, H, S1, 1] where value is 0 if the a full row in cross attn mask's
        # last dimension contains negative infinity values, otherwise it's 1
        negative_inf_value = torch.finfo(dtype).min
        full_text_row_masked_out_mask = (
            (cross_attention_mask != negative_inf_value).any(dim=-1).type_as(cross_attention_mask)[..., None]
        )
        cross_attention_mask *= full_text_row_masked_out_mask

        return cross_attention_mask, full_text_row_masked_out_mask
