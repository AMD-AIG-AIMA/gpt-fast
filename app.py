import time
from pathlib import Path

import gradio as gr
import argparse
import torch
import re
from PIL import Image as PILImage
from urllib.request import urlopen

# Import necessary components from evaluate.py 
from evaluate import process_one_question
from tokenizer import get_tokenizer
from conversation import get_conversation_template
from utils import load_model
from multimodal.vision_modules import VisionModule


def warmup(model, tokenizer, vision_modules=None, draft_model=None, draft_vision_modules=None):
    conv = get_conversation_template(args.checkpoint_path)

    sys_p = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
    conv.system_message = sys_p
    
    # Create a simple question for warmup
    warmup_question = {
        "question_id": "warmup",
        "turns": ["Hello, can you introduce yourself?"],
        "images": [
            PILImage.open(urlopen("https://www.google.com/images/branding/googlelogo/1x/googlelogo_color_272x92dp.png"))
        ]
    }
    
    # For Llama Vision models, we need to set a cross_attention_seq_length
    cross_seq_len = args.cross_attention_seq_length
    if cross_seq_len is None and hasattr(model.config, 'cross_attention_layers') and len(model.config.cross_attention_layers) > 0:
        # Default value for Llama 3.2 Vision models
        cross_seq_len = 7000
    
    # Process the question using the process_questions function from evaluate.py
    process_one_question(
        warmup_question, 
        model, 
        tokenizer, 
        conv,  
        args.max_new_tokens, 
        args.temperature, 
        args.top_k, 
        draft_model, 
        args.speculate_k, 
        args.device,
        vision_modules=vision_modules,
        draft_vision_modules=draft_vision_modules,
        max_cache_size=None,
        draft_max_cache_size=None,
        cross_attention_seq_length=cross_seq_len
    )

def bot(history, temperature, top_k, use_speculative, session_state, speculate_k=5, streaming=True):
    if not history:
        return history, "0.00 tokens/s", "0.00", "0", session_state
    
    try:
        pure_history = session_state.get("pure_history", [])
        current_msg_images = session_state.get("current_msg_images", [])
        # Get start positions from session state or initialize if not present
        start_pos = session_state.get("start_pos", {"target": 0, "draft": 0})
        
        # Get or create the conversation template
        if "conv" not in session_state:
            # First time - create conversation and set system message
            conv = get_conversation_template(args.checkpoint_path)
            sys_p = "You are a helpful, respectful and honest assistant."
            conv.system_message = sys_p
            
            # Initialize the conversation with previous turns if any
            if len(pure_history) > 1:
                for i in range(len(pure_history) - 1):
                    if pure_history[i][0] and pure_history[i][1]:
                        conv.append_message(conv.roles[0], pure_history[i][0])
                        conv.append_message(conv.roles[1], pure_history[i][1])
                        
            session_state["conv"] = conv
        else:
            # Get existing conversation
            conv = session_state["conv"]
        
        if vision_modules is not None and len(current_msg_images) > 0:
            pure_history[-1][0] = pure_history[-1][0] + " " + vision_modules.image_token
        
        generated_text = ""
        history[-1][1] = ""  # Initialize with empty response
        
        # Format the question
        question = {
            "question_id": "user_query",
            "turns": [pure_history[-1][0]],
            "images": current_msg_images
        }    
        # For Llama Vision models, we need to set a cross_attention_seq_length
        cross_seq_len = args.cross_attention_seq_length
        if cross_seq_len is None and hasattr(model.config, 'cross_attention_layers') and len(model.config.cross_attention_layers) > 0:
            # Default value for Llama 3.2 Vision models
            cross_seq_len = 7000
        
        generator = process_one_question(
            question, 
            model, 
            tokenizer, 
            conv, 
            args.max_new_tokens, 
            temperature, 
            top_k, 
            draft_model if use_speculative else None, 
            speculate_k, 
            args.device,
            vision_modules=vision_modules,
            draft_vision_modules=draft_vision_modules if use_speculative else None,
            max_cache_size=None,
            draft_max_cache_size=None,
            cross_attention_seq_length=cross_seq_len,
            start_pos=start_pos,
            streaming=streaming
        )
        
        # Process yielded tokens
        tokens_generated = 0
        start_time = time.time()
        acceptance_lengths = []
        updated_start_pos = start_pos
        results = {}
        try:
            if streaming:
                while True:
                    token = next(generator)
                    # Decode the token and add to generated text
                    if use_speculative and token.numel() > 1:
                        token_list = token[0].tolist() if token.dim()>1 else token.tolist()
                        token_text = tokenizer.decode(token_list[:-1])
                        last_token = tokenizer.decode(token_list[-1:])
                        token_text = f"<span style='color: rgb(0, 124, 151);'>{token_text}</span>" + last_token 
                    else:
                        token_text = tokenizer.decode(token[0].tolist() if token.dim()>1 else token.tolist())
                    
                    generated_text += token_text
                    tokens_generated += token.numel()
                    # Update the history for display
                    history[-1][1] = generated_text
                    
                    # Yield updated history to UI
                    yield history, "Generating...", "Generating...", str(tokens_generated), session_state
            else:
                while True:
                    # generate is a generator, so we need to call next to get the final output
                    _ = next(generator)
        except StopIteration as e:
            # The generator will raise StopIteration when done, with return values as e.value
            updated_start_pos, results = e.value
            if not streaming:
                # Update history with the final generated text from results
                history[-1][1] = results["generated_text"]
        # Update state and finalize
        session_state["start_pos"] = updated_start_pos
        
        elapsed_time = results["walltime"]
        generated_text = results["generated_text"]
        tokens_generated = results["tokens_generated"]
        tokens_per_second = tokens_generated / elapsed_time
        
        # Calculate mean acceptance length for speculative decoding
        mean_acceptance = "N/A"
        if use_speculative and "acceptance_lengths" in results and results["acceptance_lengths"]:
            acceptance_lengths = results["acceptance_lengths"]
            mean_acceptance = f"{sum(acceptance_lengths) / len(acceptance_lengths):.2f}"
        
        # Update conversation
        conv.append_message(conv.roles[1], generated_text)
        session_state["conv"] = conv
        
        # Update history
        pure_history[-1][1] = generated_text
        session_state["pure_history"] = pure_history
        
        # Clear current message images after processing to prevent reuse
        session_state["current_msg_images"] = []
        
        # Final yield with stats
        yield history, f"{tokens_per_second:.2f} tokens/s", mean_acceptance, str(tokens_generated), session_state
        
    except Exception as e:
        import traceback
        error_msg = f"Error processing your request: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        history[-1][1] = f"I encountered an error processing your request. Please try again or check the logs for details."
        yield history, "0.00 tokens/s", "0.00", "0", session_state

def user(user_message, history, session_state, img=None):
    if history is None:
        history = []
    
    pure_history = session_state.get("pure_history", [])
    
    # Handle image input - only if it's a newly uploaded image
    if img is not None and img not in session_state.get("processed_images", []):
        # This is a new image, add it to current_msg_images
        session_state["current_msg_images"] = [img]
        
        # Keep track of processed images to avoid reprocessing
        processed_images = session_state.get("processed_images", [])
        processed_images.append(img)
        session_state["processed_images"] = processed_images
    
    pure_history += [[user_message, None]]
    session_state["pure_history"] = pure_history
    
    return "", history + [[user_message, None]], session_state

def regenerate(history, session_state):
    if not history:
        return history, None, "0.00 tokens/s", "0.00", session_state
    
    pure_history = session_state.get("pure_history", [])
    pure_history[-1][-1] = None
    session_state["pure_history"] = pure_history
    
    if len(history) > 1:
        new_history = history[:-1]
        last_user_message = history[-1][0]
        return new_history + [[last_user_message, None]], None, "0.00 tokens/s", "0.00", session_state
    
    history[-1][1] = None
    return history, None, "0.00 tokens/s", "0.00", session_state

def clear(history, session_state):
    # Reset all state
    session_state["pure_history"] = []
    session_state["images"] = []
    session_state["current_msg_images"] = []
    session_state["processed_images"] = []
    # Reset position tracking
    session_state["start_pos"] = {"target": 0, "draft": 0}
    
    # Clear model caches
    if 'model' in globals():
        model.clear_cache()
    if 'draft_model' in globals() and draft_model is not None:
        draft_model.clear_cache()
    
    return [], [], "0.00 tokens/s", "0.00", session_state

def update_gallery(img, state):
    if img is not None:
        # Replace any existing images with just this one
        state["current_msg_images"] = [img]
    
    # Return the current images for display
    return state["current_msg_images"], state

# Update argument parser with multimodal options
parser = argparse.ArgumentParser()
parser.add_argument(
    "--checkpoint_path",
    type=Path,
    required=True,
    help="The path to the model checkpoint. This can be a local folder or a Hugging Face repo ID.",
)
parser.add_argument(
    "--draft_checkpoint_path",
    type=Path,
    default=None,
    help="The path to the draft model checkpoint for speculative decoding.",
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="Device to use for computation (cuda or cpu)",
)
parser.add_argument(
    "--dtype",
    type=str,
    default="fp16",
    choices=["fp32", "fp16", "bf16"],
    help="Data type for model weights and activations",
)
parser.add_argument(
    "--max_new_tokens",
    type=int,
    default=512,
    help="The maximum number of new generated tokens.",
)
parser.add_argument(
    "--temperature",
    type=float,
    default=0.8,
    help="Temperature for sampling",
)
parser.add_argument(
    "--top_k",
    type=int,
    default=1,
    help="Top-k for sampling",
)
parser.add_argument(
    "--speculate_k",
    type=int,
    default=5,
    help="Speculative execution depth",
)
parser.add_argument(
    "--cross_attention_seq_length",
    type=int,
    default=None,
    help="Maximum cross-attention sequence length for vision models",
)
parser.add_argument(
    "--random_seed", 
    default=None, 
    type=int, 
    help="Random seed")
args = parser.parse_args()

if args.random_seed:
    torch.manual_seed(args.random_seed)
# Initialize model and tokenizer
print(f"Loading model from {args.checkpoint_path} on {args.device}")
precision = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype]

# Load target model
model = load_model(args.checkpoint_path, args.device, precision, use_tp=False)
model.requires_grad_(False)
model._device = args.device

# Check if model is multimodal
multimodal = getattr(model.config, "mm_config", None) is not None
vision_modules = None

if multimodal:
    print("Loading vision modules for multimodal model")
    torch.set_default_dtype(precision)
    torch.set_default_device(args.device)
    vision_checkpoints = args.checkpoint_path.parent / "vision_modules.pth"
    vision_modules = VisionModule.from_name(
        args.checkpoint_path.parent.name, 
        config=model.config.mm_config,
        checkpoint_path=vision_checkpoints,
        dtype=precision,
        device=args.device
    )
    vision_modules.eval_mode()

# Load draft model if specified
draft_model = None
draft_vision_modules = None
if args.draft_checkpoint_path:
    print(f"Loading draft model from {args.draft_checkpoint_path}")
    draft_model = load_model(args.draft_checkpoint_path, args.device, precision, use_tp=False)
    draft_model.requires_grad_(False)
    draft_model._device = args.device
    
    # Check if draft model is multimodal
    draft_multimodal = getattr(draft_model.config, "mm_config", None) is not None
    if draft_multimodal:
        print("Loading vision modules for draft multimodal model")
        draft_vision_checkpoints = args.draft_checkpoint_path.parent / "vision_modules.pth"
        draft_vision_modules = VisionModule.from_name(
            args.draft_checkpoint_path.parent.name, 
            config=draft_model.config.mm_config,
            checkpoint_path=draft_vision_checkpoints,
            dtype=precision,
            device=args.device
        )
        draft_vision_modules.eval_mode()

# Load tokenizer
tokenizer = get_tokenizer(args.checkpoint_path.parent / "tokenizer.model", args.checkpoint_path)

# Warmup the model
warmup(model, tokenizer, vision_modules, draft_model, draft_vision_modules)

custom_css = """
#speed textarea {
    color: rgb(237, 28, 36);   
    font-size: 25px; 
}
#tokens textarea {
    color: rgb(242, 101, 34);   
    font-size: 25px; 
}
"""


with gr.Blocks(css=custom_css) as demo:
    gs = gr.State({
        "pure_history": [], 
        "images": [], 
        "current_msg_images": [],
        "start_pos": {"target": 0, "draft": 0}
    })
    
    # Display model information
    model_name = args.checkpoint_path.parent.name
    model_type = "Multimodal" if multimodal else "Text-only"
    gr.Markdown(f'''## {model_type} Chat Interface - {model_name}''')
    
    if args.draft_checkpoint_path:
        draft_model_name = args.draft_checkpoint_path.parent.name
        gr.Markdown(f'''### Using speculative decoding with draft model: {draft_model_name}''')
    
    with gr.Row():
        speed_box = gr.Textbox(label="Genetation Speed", elem_id="speed", interactive=False, value="0.00 tokens/s")
        acceptance_box = gr.Textbox(label="Mean Acceptance Length", elem_id="speed", interactive=False, value="0.00")
        tokens_box = gr.Textbox(label="Tokens Generated", elem_id="tokens", interactive=False, value="0")
    
    
    with gr.Row():
        if multimodal:
            chatbot = gr.Chatbot(height=600, show_label=False, scale=2)
            with gr.Column(scale=1):
                img_input = gr.Image(type="pil", label="Upload Image", sources=["upload", "webcam"])
                # Add visual indication of uploaded images
                image_gallery = gr.Gallery(label="Current message images", show_label=False, visible=False, elem_id="image_gallery")
        else:
            # For text-only models, use the full width for chatbot
            chatbot = gr.Chatbot(height=600, show_label=False)
            # Create hidden image components to maintain API compatibility
            img_input = gr.Image(type="pil", visible=False)
            image_gallery = gr.Gallery(visible=False)

    msg = gr.Textbox(label="Your input", placeholder="Type your message here...")
    
    with gr.Row():
        send_button = gr.Button("Send")
        stop_button = gr.Button("Stop")
        regenerate_button = gr.Button("Regenerate")
        clear_button = gr.Button("Clear")

    with gr.Row():
        use_speculative = gr.Checkbox(label="Use Speculative Decoding", value=args.draft_checkpoint_path is not None)
        temperature = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="Temperature", value=args.temperature)
        top_k = gr.Slider(minimum=1, maximum=200, step=1, label="Top K", value=1)
        speculate_k = gr.Slider(minimum=1, maximum=20, step=1, label="Number of draft tokens", value=args.speculate_k)
    
    enter_event = msg.submit(user, [msg, chatbot, gs, img_input], [msg, chatbot, gs], queue=False).then(
        bot, [chatbot, temperature, top_k, use_speculative, gs, speculate_k], [chatbot, speed_box, acceptance_box, tokens_box, gs], queue=True
    )
    
    clear_button.click(clear, [chatbot, gs], [chatbot, speed_box, acceptance_box, tokens_box, gs], queue=False)
    
    send_event = send_button.click(user, [msg, chatbot, gs, img_input], [msg, chatbot, gs], queue=False).then(
        bot, [chatbot, temperature, top_k, use_speculative, gs, speculate_k], [chatbot, speed_box, acceptance_box, tokens_box, gs], queue=True
    )
    
    regenerate_event = regenerate_button.click(regenerate, [chatbot, gs], [chatbot, msg, speed_box, acceptance_box, tokens_box, gs], queue=False).then(
        bot, [chatbot, temperature, top_k, use_speculative, gs, speculate_k], [chatbot, speed_box, acceptance_box, tokens_box, gs], queue=True
    )
    
    stop_button.click(fn=None, inputs=None, outputs=None, cancels=[send_event, regenerate_event, enter_event])

    # Only add image gallery update when multimodal
    if multimodal:
        img_input.upload(update_gallery, [img_input, gs], [image_gallery, gs])

demo.queue().launch(share=True)
