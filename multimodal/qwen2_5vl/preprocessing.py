from transformers import AutoProcessor
import torch
import re
def qwen2_5vl_process_vision_info():
    pass
IMAGE_TOKEN_PATTERN = '<|vision_start|><|image_pad|><|vision_end|>'
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

def process_visual_inputs(processor, text, device, dtype):
    from qwen_vl_utils import process_vision_info
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device=device, dtype=dtype)
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

def prepare_input_embeds(input_ids, pixel_values, image_grid_thw, embed_tokens, vision_model, image_token_id , device, dtype):
    inputs_embeds = embed_tokens(input_ids)
    inputs_embeds = inputs_embeds.to(device=device, dtype=dtype)
    if pixel_values is not None:
        pixel_values = pixel_values.to(device=device, dtype=dtype)
        image_embeds = vision_model(pixel_values, grid_thw=image_grid_thw)
        n_image_tokens = (input_ids == image_token_id).sum().item()
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        mask = input_ids == image_token_id
        mask_unsqueezed = mask.unsqueeze(-1)
        mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
        image_mask = mask_expanded.to(device=device)

        image_embeds = image_embeds.to(device=device, dtype=dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
    return inputs_embeds.detach()

def get_rope_index(input_ids, image_grid_thw, mm_config, attention_mask=None):
        if input_ids.ndim < 2:
            input_ids = input_ids.unsqueeze(0)
        spatial_merge_size = mm_config.spatial_merge_size
        image_token_id = 151655
        vision_start_token_id = 151652
        mrope_position_deltas = []
        if input_ids is not None and image_grid_thw is not None:
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index = 0
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums = 0
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images = image_nums
                for _ in range(image_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed = input_tokens.index(image_token_id, st)
                    else:
                        ed = len(input_tokens) + 1

                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    second_per_grid_t = 0
                    image_index += 1
                    remain_images -= 1

                    
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                    expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

                    time_tensor = expanded_range * second_per_grid_t * mm_config.tokens_per_second

                    time_tensor_long = time_tensor.long()
                    t_index = time_tensor_long.flatten()

                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )

            return position_ids, mrope_position_deltas
        
def get_position_ids(model_inputs, mm_config):
    position_ids, rope_deltas = get_rope_index(
        model_inputs['input_ids'],
        model_inputs['image_grid_thw'],
        mm_config,
        model_inputs['attention_mask'],
    )
    return position_ids, rope_deltas
    
    
    
def embed_token_multimodal_qwen2_5vl(
    prompt,
    processor_id,
    device,
    images,
    vision_modules,
    embed_tokens,
    dtype,
):  
    # print('before prompt', prompt)
    image_pattern = re.compile(r'<image(?:\s+\d+)?>')
    prompt = re.sub(image_pattern, IMAGE_TOKEN_PATTERN, prompt)
    # print('after prompt', prompt)
    processor = get_processor(processor_id)
    inputs = processor(
        text=[prompt],
        images=images,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device=device, dtype=dtype)
    inputs_embeds = prepare_input_embeds(inputs['input_ids'], inputs['pixel_values'], inputs['image_grid_thw'], embed_tokens, vision_modules,151655,device,dtype)
    # print(inputs['input_ids'].shape, inputs_embeds.shape)

    return inputs, inputs_embeds