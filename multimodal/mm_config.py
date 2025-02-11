from dataclasses import dataclass, field
from typing import Optional, List


def get_default_pinpoints() -> List[List[int]]:
    return [
        [384, 384], [384, 768], [384, 1152], [384, 1536], [384, 1920], [384, 2304],
        [768, 384], [768, 768], [768, 1152], [768, 1536], [768, 1920], [768, 2304],
        [1152, 384], [1152, 768], [1152, 1152], [1152, 1536], [1152, 1920], [1152, 2304],
        [1536, 384], [1536, 768], [1536, 1152], [1536, 1536], [1536, 1920], [1536, 2304],
        [1920, 384], [1920, 768], [1920, 1152], [1920, 1536], [1920, 1920], [1920, 2304],
        [2304, 384], [2304, 768], [2304, 1152], [2304, 1536], [2304, 1920], [2304, 2304]
    ]

@dataclass
class MultimodalModelArgs:
    mm_newline_position: str = "one_token"
    image_aspect_ratio: str = "anyres_max_9" 
    image_crop_resolution: Optional[int] = None
    image_grid_pinpoints: List[List[int]] = field(default_factory=get_default_pinpoints)
    image_split_resolution: Optional[int] = None
    image_token_index: int = 151646
    mm_hidden_size: int = 1152
    mm_patch_merge_type: str = "spatial_unpad"
    mm_projector_type: str = "mlp2x_gelu"
    mm_resampler_type: Optional[str] = None
    mm_spatial_pool_mode: str = "bilinear"
    mm_use_im_patch_token: bool = False
    mm_use_im_start_end: bool = False
    mm_vision_select_feature: str = "patch"
    mm_vision_select_layer: int = -2
    mm_vision_tower: str = "google/siglip-so400m-patch14-384"
    tokenizer_padding_side: str = "right"
    use_mm_proj: bool = True
    hidden_size: int = 3584 # This is the hidden dimension from the LLM


    def __post_init__(self):
        pass

    @classmethod
    def from_name(cls, name: str):
        if name in mm_transformer_config:
            return cls(**mm_transformer_config[name])
        # fuzzy search
        config = [config for config in mm_transformer_config if config.lower() in str(name).lower()]

        # We may have two or more configs matched (e.g. "7B" and "Mistral-7B"). Find the best config match,
        # take longer name (as it have more symbols matched)
        if len(config) > 1:
            config.sort(key=len, reverse=True)
            assert len(config[0]) != len(config[1]), name # make sure only one 'best' match
            
        return cls(**mm_transformer_config[config[0]])


mm_transformer_config = {
    "llava-onevision-qwen2-72b-si": dict(
        mm_newline_position= "one_token",
        image_aspect_ratio= "anyres_max_9",
        image_crop_resolution= None,
        image_grid_pinpoints= get_default_pinpoints(),
        image_split_resolution= None,
        image_token_index= 151646,
        mm_hidden_size= 1152,
        mm_patch_merge_type= "spatial_unpad",
        mm_projector_type= "mlp2x_gelu",
        mm_resampler_type= None,
        mm_spatial_pool_mode= "bilinear",
        mm_use_im_patch_token= False,
        mm_use_im_start_end= False,
        mm_vision_select_feature= "patch",
        mm_vision_select_layer= -2,
        mm_vision_tower= "google/siglip-so400m-patch14-384",
        tokenizer_padding_side= "right",
        use_mm_proj= True,
        hidden_size=8192,
    ),
    "llava-onevision-qwen2-7b-si": dict(
        mm_newline_position= "one_token",
        image_aspect_ratio= "anyres_max_9",
        image_crop_resolution= None,
        image_grid_pinpoints= get_default_pinpoints(),
        image_split_resolution= None,
        image_token_index= 151646,
        mm_hidden_size= 1152,
        mm_patch_merge_type= "spatial_unpad",
        mm_projector_type= "mlp2x_gelu",
        mm_resampler_type= None,
        mm_spatial_pool_mode= "bilinear",
        mm_use_im_patch_token= False,
        mm_use_im_start_end= False,
        mm_vision_select_feature= "patch",
        mm_vision_select_layer= -2,
        mm_vision_tower= "google/siglip-so400m-patch14-384",
        tokenizer_padding_side= "right",
        use_mm_proj= True,
        hidden_size=3584,
    ),
    "llava-onevision-qwen2-0.5b": dict(
        mm_newline_position= "one_token",
        image_aspect_ratio= "anyres_max_9",
        image_crop_resolution= None,
        image_grid_pinpoints= get_default_pinpoints(),
        image_split_resolution= None,
        image_token_index= 151646,
        mm_hidden_size= 1152,
        mm_patch_merge_type= "spatial_unpad",
        mm_projector_type= "mlp2x_gelu",
        mm_resampler_type= None,
        mm_spatial_pool_mode= "bilinear",
        mm_use_im_patch_token= False,
        mm_use_im_start_end= False,
        mm_vision_select_feature= "patch",
        mm_vision_select_layer= -2,
        mm_vision_tower= "google/siglip-so400m-patch14-384",
        tokenizer_padding_side= "right",
        use_mm_proj= True,
        hidden_size=896,
    ),
    
    "Qwen2.5-VL-3B-Instruct": dict(
    depth= 32,
    hidden_act= "silu",
    hidden_size= 1280,
    intermediate_size= 3420,
    num_heads= 16,
    in_chans= 3,
    out_hidden_size= 2048,
    patch_size= 14,
    spatial_merge_size= 2,
    spatial_patch_size= 14,
    window_size= 112,
    fullatt_block_indexes= [
      7,
      15,
      23,
      31
    ],
    tokens_per_second= 2,
    temporal_patch_size= 2
    ),
    "Qwen2.5-VL-7B-Instruct": dict(
    depth= 32,
    hidden_act= "silu",
    hidden_size= 1280,
    intermediate_size= 3420,
    num_heads= 16,
    in_chans= 3,
    out_hidden_size= 3584,
    patch_size= 14,
    spatial_merge_size= 2,
    spatial_patch_size= 14,
    window_size= 112,
    fullatt_block_indexes= [
      7,
      15,
      23,
      31
    ],
    tokens_per_second= 2,
    temporal_patch_size= 2
    ),
    "Qwen2.5-VL-72B-Instruct": dict(
    depth= 32,
    hidden_act= "silu",
    hidden_size= 1280,
    intermediate_size= 3456,
    num_heads= 16,
    in_chans= 3,
    out_hidden_size= 8192,
    patch_size= 14,
    spatial_merge_size= 2,
    spatial_patch_size= 14,
    window_size= 112,
    fullatt_block_indexes= [
      7,
      15,
      23,
      31
    ],
    tokens_per_second= 2,
    temporal_patch_size= 2
    ),
}

@dataclass
class QwenVisionModelArgs:
    depth: int = 32,
    hidden_act: str = "silu",
    hidden_size: int = 1280,
    intermediate_size: int = 3420,
    num_heads: int = 16,
    in_chans: int = 3,
    out_hidden_size: int = 2048,
    patch_size: int = 14,
    spatial_merge_size: int = 2,
    spatial_patch_size: int = 14,
    window_size: int = 112,
    fullatt_block_indexes: list = [
      7,
      15,
      23,
      31
    ],
    tokens_per_second: int = 2,
    temporal_patch_size: int = 2


    def __post_init__(self):
        pass

    @classmethod
    def from_name(cls, name: str):
        if name in mm_transformer_config:
            return cls(**mm_transformer_config[name])
        raise ValueError("Please set the Qwen config manually")