from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor

def qwen2_5vl_process_vision_info():
    pass

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

def get_qwen2_5vl_message_template(image_path=None, text_prompt=None):
    message= messages.copy()
    if image_path:
        message[0]["content"][0]["image"] = image_path
    if text_prompt:
        message[0]["content"][1]["text"] = text_prompt 
    return message

def get_processor(processor_id = "Qwen/Qwen2.5-VL-3B-Instruct"):
    return AutoProcessor.from_pretrained(processor_id)

def process_chat_template(processor, message):
    return processor.apply_chat_template(
        message, tokenize=False, add_generation_prompt=True
    )

def process_visual_inputs(processor, text, device):
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device=device)
    return inputs

def set_model_inputs(
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        **kwargs,
    ):
    
    if inputs_embeds is not None and cache_position[0] == 0:
        model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
    else:
        model_inputs = {"input_ids": input_ids, "inputs_embeds": None}


    model_inputs.update(
        {
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "pixel_values_videos": pixel_values_videos,
            "image_grid_thw": image_grid_thw,
            "video_grid_thw": video_grid_thw,
            "cache_position": cache_position,
            "second_per_grid_ts": second_per_grid_ts,
        }
    )
    return model_inputs

def prepare_input_embeds(input_ids, model, vision_model, model_inputs, config):
    inputs_embeds = model.embed_tokens(input_ids)
    if pixel_values is not None:
        pixel_values = pixel_values.type(vision_model.dtype)
        image_embeds = vision_model(pixel_values, grid_thw=model_inputs.image_grid_thw)
        n_image_tokens = (input_ids == config.image_token_id).sum().item()
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        mask = input_ids == config.image_token_id
        mask_unsqueezed = mask.unsqueeze(-1)
        mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
        image_mask = mask_expanded.to(inputs_embeds.device)

        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        
    return inputs_embeds

def get_rope_index():
    pass
def prepare_inputs_for_generation():
    pass

    
    
    
    