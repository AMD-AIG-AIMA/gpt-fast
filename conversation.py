MODEL_TO_TEMPLATE = {
    "llama-2-7b-chat": "llama-2-chat",
    "llama-2-13b-chat": "llama-2-chat",
    "llama-2-70b-chat": "llama-2-chat",
    "mistral-7b-instruct": "mistral",
    "mixtral-8x7b-instruct": "mistral",
    "openchat-3.5": "openchat",
    "qwen2": "qwen2",
    # Add more mappings as needed
}
from fastchat.model import get_conversation_template as get_fastchat_template

class HFConversationTemplate:
    def __init__(self, processor):
        self.processor = processor
        self.messages = []
        self.roles = ["user", "assistant"]
        self.system_message = None
        self.stop_token_ids = [processor.tokenizer.eos_token_id]
        self.stop_str = processor.tokenizer.eos_token

    def set_system_message(self, message):
        self.system_message = message

    def append_message(self, role, message):
        if message is not None and role == "user":
            self.messages.append({"role": role, "content": [{"type": "text", "text": message}]})

    def get_prompt(self):
        all_messages = []
        if self.system_message:
            all_messages.append({
                "role": "system",
                "content": self.system_message
            })
        all_messages.extend(self.messages)
        return self.processor.apply_chat_template(all_messages, add_generation_prompt=True)

def get_model_template(model_name):
    for key, template in MODEL_TO_TEMPLATE.items():
        if key in model_name.lower():
            return template
    return -1  # Default to llama-2-chat if no match found

def get_conversation_template(checkpoint_path):
    # Try to find the model template in FastChat
    model_template = get_model_template(checkpoint_path.parent.name)
    if model_template != -1:
        return get_fastchat_template(model_template)
    else:
        # Fallback to custom template with HF processor
        from transformers import AutoProcessor
        model_id = f"{checkpoint_path.parent.parent.name}/{checkpoint_path.parent.name}"
        processor = AutoProcessor.from_pretrained(model_id)
        return HFConversationTemplate(processor)