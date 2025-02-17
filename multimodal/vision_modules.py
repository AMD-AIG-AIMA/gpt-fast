from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union, Dict, Type
import torch
import torch.nn as nn
from PIL import Image
from dataclasses import dataclass
from pathlib import Path
import re

from multimodal.llava.preprocessing import process_images, tokenizer_image_token, prepare_inputs_labels_for_multimodal, IMAGE_TOKEN_INDEX

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
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device is not None:
            self._device = torch.device(device)
        self._is_loaded = False
        self.to(self._device)
    

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
    

@VisionModule.register("llava")
class LlavaVisionModule(VisionModule):
    def __init__(
        self,
        config,
        checkpoint_path: Optional[Path] = None,
        dtype: torch.dtype = torch.float16,
        device: Optional[Union[str, torch.device]] = None
    ):
        super().__init__(config, dtype, device)
        from multimodal.llava.builder import build_vision_tower, build_vision_resampler, build_vision_projector
        self.delay_load = getattr(config, "delay_load", False)
        self.vision_tower = build_vision_tower(config, delay_load=self.delay_load)
        self.vision_resampler = build_vision_resampler(config, vision_tower=self.vision_tower)
        self.mm_projector = build_vision_projector(config, vision_cfg=self.vision_tower)
        self.image_newline = None
        if "unpad" in getattr(config, "mm_patch_merge_type", ""):
            self.image_newline = nn.Parameter(torch.empty(config.hidden_size, dtype=dtype))
        if checkpoint_path is not None:
            self.load_checkpoint(Path(checkpoint_path))

    def preprocess_images(self, images):
        pass

    def forward(self,
        prompt,
        tokenizer,
        images,
        embed_tokens,
        ):
        image_tensor = process_images(images, self.vision_tower.image_processor, self.config)
        image_tensor = [_image.to(dtype=self.dtype, device=self._device) for _image in image_tensor]
        input_ids = tokenizer_image_token(
                            prompt, 
                            tokenizer,
                            IMAGE_TOKEN_INDEX,
                            return_tensors="pt"
                        ).unsqueeze(0).to(self._device)
        embeds = prepare_inputs_labels_for_multimodal(input_ids, None, None, None, None, 
                                    image_tensor,
                                    config=self.config,
                                    vision_tower=self.vision_tower,
                                    mm_projector=self.mm_projector,
                                    embed_tokens=embed_tokens,
                                    device=self._device,
                                    modalities=["image"],
                                    image_newline=self.image_newline, 
                                    image_sizes=[img.size for img in images])
        return input_ids, embeds[4].to(dtype=self.dtype)
    

@VisionModule.register("llama")
class LlamaVisionModule(VisionModule):
    def __init__(
        self,
        config,
        checkpoint_path: Optional[Path] = None,
        dtype: torch.dtype = torch.float16,
        device: Optional[Union[str, torch.device]] = None
    ):
        super().__init__(config, dtype, device)
        from transformers import MllamaVisionModel, MllamaProcessor
        self.vision_model = MllamaVisionModel._from_config(config)
        # self.vision_model = MllamaVisionModel.from_pretrained(str(checkpoint_path.parent))
        self.multi_modal_projector = nn.Linear(
            config.vision_output_dim,
            config.text_hidden_size,
            bias=True,
        )
        
        self.processor = MllamaProcessor.from_pretrained(str(checkpoint_path.parent))
        if checkpoint_path is not None:
            self.load_checkpoint(Path(checkpoint_path))

    def preprocess_images(self, images):
        pass

    def forward(self,
        prompt,
        tokenizer,
        images,
        embed_tokens,
        ):
        # Replace <image>, <image1>, <image2>, ... with <|image|>
        prompt = re.sub(r'<image\s*\d*>', '<|image|>', prompt)
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

        cross_attention_mask, out_mask = self._prepare_cross_attention_mask(
            inputs.cross_attention_mask,
            num_vision_tokens=self.vision_model.num_patches,
            dtype=self.dtype,
        )
        self.cross_attention_mask = {'mask': cross_attention_mask, 'out_mask': out_mask}
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