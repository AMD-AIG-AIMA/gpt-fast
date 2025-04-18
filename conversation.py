from transformers import AutoProcessor

class HFConversationTemplate:
    def __init__(self, processor):
        self.processor = processor
        self.messages = []
        self.roles = ["user", "assistant"]
        self.system_message = None
        try:
            eos_token_id = processor.eos_token_id
            eos_token = processor.eos_token
            self.bos_token = processor.bos_token if hasattr(processor, "bos_token") else None
        except:
            eos_token_id = processor.tokenizer.eos_token_id
            eos_token = processor.tokenizer.eos_token
            self.bos_token = processor.tokenizer.bos_token if hasattr(processor.tokenizer, "bos_token") else None
        self.stop_token_ids = [eos_token_id]
        self.stop_str = eos_token

    def set_system_message(self, message):
        self.system_message = message

    def append_message(self, role, message):
        if message is not None:
            self.messages.append({"role": role, "content": message})

    def get_prompt(self, add_generation_prompt=False):
        all_messages = []
        if self.system_message:
            all_messages.append({
                "role": "system",
                "content": self.system_message
            })
        all_messages.extend(self.messages)
        return self.processor.apply_chat_template(all_messages, add_generation_prompt=add_generation_prompt, tokenize=False)
    
    def get_prompt_for_generation(self):            
        return self.get_prompt(add_generation_prompt=True)

def get_conversation_template(checkpoint_path):
    try:
        model_path = f"{checkpoint_path.parent}"
        if not (checkpoint_path.parent/"chat_template.json").exists():
            from huggingface_hub import hf_hub_download

            file_path = hf_hub_download(
                repo_id="Qwen/Qwen2.5-VL-3B-Instruct",
                filename="chat_template.json",
                local_dir=model_path,
                local_dir_use_symlinks=False  # set to False to copy instead of symlink
            )

            
            print(f"Chat template for {checkpoint_path.parent.parent.name}/{checkpoint_path.parent.name} not found, downloaded from Qwen/Qwen2.5-VL-3B-Instruct and saved at:", file_path)
        processor = AutoProcessor.from_pretrained(model_path)

        return HFConversationTemplate(processor)
    except Exception as e:
        raise NotImplementedError(f"The HF processor is not implemented for model {checkpoint_path}! Check this error: \n\n{e}")