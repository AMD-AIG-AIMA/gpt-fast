from dataclasses import dataclass, field
from typing import Optional, List, Dict, Type

from transformers import MllamaConfig


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
class Config:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

@dataclass
class MultimodalModelArgs:
    """Base class for multimodal model arguments"""
    # Registry for model types
    _registry: Dict[str, Type['MultimodalModelArgs']] = field(default_factory=dict)

    @classmethod
    def register(cls, model_name: str):
        """Decorator to register model argument classes"""
        def register_model_cls(model_cls):
            if not hasattr(cls, '_registry'):
                cls._registry = {}
            cls._registry[model_name] = model_cls
            return model_cls
        return register_model_cls

    @classmethod
    def from_name(cls, name: str):
        """Create model args instance from name"""
        # Find matching model type in registry
        mapped_name = [n for n in cls._registry if n.lower() in name.lower()]
        if len(mapped_name) == 0:
            raise ValueError(f"Model {name} not found in registry. Available models: {list(cls._registry.keys())}")
        if len(mapped_name) > 1:
            mapped_name.sort(key=len, reverse=True)
            assert len(mapped_name[0]) != len(mapped_name[1]), name  # make sure only one 'best' match
        
        # Get config overrides
        if name in mm_transformer_config:
            config = mm_transformer_config[name]
        else:
            # fuzzy search
            configs = [c for c in mm_transformer_config if c.lower() in name.lower()]
            configs.sort(key=len, reverse=True)
            assert len(configs) > 0, f"No config found for {name}"
            config = mm_transformer_config[configs[0]]
            
        return cls._registry[mapped_name[0]](**config)


@MultimodalModelArgs.register("llava")
@dataclass
class LlavaMultimodalModelArgs(MultimodalModelArgs):
    """Arguments specific to Llava models"""
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


@MultimodalModelArgs.register("llama")
@dataclass
class LlamaMultimodalModelArgs(MultimodalModelArgs,MllamaConfig):
    """Arguments specific to Llama models"""
    _name_or_path: str = ""
    add_cross_attention: bool = False
    attention_heads: int = 16
    chunk_size_feed_forward: int = 0
    architectures = None
    bad_words_ids = None
    begin_suppress_tokens = None
    bos_token_id = None
    cross_attention_hidden_size = None
    decoder_start_token_id = None
    diversity_penalty: float = 0.0
    do_sample: bool = False
    early_stopping: bool = False
    encoder_no_repeat_ngram_size: int = 0
    eos_token_id = None
    exponential_decay_length_penalty = None
    finetuning_task = None
    forced_bos_token_id = None
    forced_eos_token_id = None
    hidden_act: str = "gelu"
    hidden_size: int = 1280
    image_size: int = 560
    intermediate_layers_indices = [3,7,15,23,30]
    intermediate_size: int = 5120
    is_decoder: bool = False
    is_encoder_decoder: bool = False
    supported_aspect_ratios = [
        [1,1],[1,2],[1,3],[1,4],
        [2,1],[2,2],[3,1],[4,1]
    ]
    length_penalty: float = 1.0
    max_length: int = 20
    max_num_tiles: int = 4
    min_length: int = 0
    model_type: str = "mllama_vision_model"
    no_repeat_ngram_size: int = 0
    norm_eps: float = 1e-05
    num_beam_groups: int = 1
    num_beams: int = 1
    num_channels: int = 3
    num_global_layers: int = 8
    num_hidden_layers: int = 32
    num_return_sequences: int = 1
    output_attentions: bool = False
    output_hidden_states: bool = False
    output_scores: bool = False
    pad_token_id = None
    patch_size: int = 14
    prefix = None
    problem_type = None
    pruned_heads = {}
    remove_invalid_values: bool = False
    repetition_penalty: float = 1.0
    return_dict: bool = True
    return_dict_in_generate: bool = False
    sep_token_id = None
    suppress_tokens = None
    task_specific_params = None
    temperature: float = 1.0
    tf_legacy_loss: bool = False
    tie_encoder_decoder: bool = False
    tie_word_embeddings: bool = True
    tokenizer_class =None
    top_k: int = 50
    top_p: float = 1.0
    torch_dtype: str = "bfloat16"
    torchscript: bool = False
    typical_p: float = 1.0
    use_bfloat16: bool = False
    vision_output_dim: int = 7680
    text_config = Config(initializer_range=0.02)
    vision_config = Config()
    max_aspect_ratio_id = 8
    text_hidden_size=4096
    _attn_implementation_internal = None
    


# Config dictionary can be simplified to only include overrides
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
    "llava-onevision-qwen2-0.5b-si": dict(
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
    "llama-3.2-11b-vision-instruct": dict(
        attention_heads=16,
        hidden_act="gelu",
        hidden_size=1280,
        image_size=560,
    ),
}