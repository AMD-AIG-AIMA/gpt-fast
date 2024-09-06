import json
import time
from pathlib import Path
from typing import Optional
import torch
from tqdm import tqdm
from generate import generate, encode_tokens, get_tokenizer, _load_model, model_forward, logits_to_probs, decode_one_token, prefill
from fastchat.model import get_conversation_template
from tp import maybe_init_dist, apply_tp

MODEL_TO_TEMPLATE = {
    "llama-2-7b-chat": "llama-2-chat",
    "llama-2-13b-chat": "llama-2-chat",
    "llama-2-70b-chat": "llama-2-chat",
    "mistral-7b-instruct": "mistral",
    "mixtral-8x7b-instruct": "mistral",
    "openchat-3.5": "openchat",
    # Add more mappings as needed
}

def load_benchmark_data(bench_name):
    data_path = Path(f"./data/{bench_name}/questions.jsonl")
    questions = []
    with open(data_path, 'r') as f:
        for line in f:
            questions.append(json.loads(line.strip()))
    return questions

def get_model_template(model_name):
    for key, template in MODEL_TO_TEMPLATE.items():
        if key in model_name.lower():
            return template
    return "llama-2-chat"  # Default to llama-2-chat if no match found

def main(
    bench_name: str,
    checkpoint_path: Path,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 200,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    draft_checkpoint_path: Optional[Path] = None,
    speculate_k: int = 5,
    compile: bool = False,
    compile_prefill: bool = False,
):
    # Initialize distributed setup
    rank = maybe_init_dist()
    use_tp = rank is not None
    if use_tp:
        if rank != 0:
            # only print on rank 0
            print = lambda *args, **kwargs: None

    print(f"Using device={device}")
    precision = torch.bfloat16
    is_speculative = draft_checkpoint_path is not None

    print("Loading model ...")
    t0 = time.time()
    model = _load_model(checkpoint_path, device, precision, use_tp)

    if is_speculative:
        print(f"Loading draft model from {draft_checkpoint_path}")
        draft_model = _load_model(draft_checkpoint_path, device, torch.bfloat16, use_tp=False)
    else:
        draft_model = None

    if use_tp:
        print("Applying tensor parallel to model ...")
        apply_tp(model)

    print(f"Time to load model: {time.time() - t0:.02f} seconds")

    tokenizer = get_tokenizer(checkpoint_path.parent / "tokenizer.model", checkpoint_path)

    if compile:
        if is_speculative:
            global model_forward, logits_to_probs
            model_forward = torch.compile(model_forward, mode="reduce-overhead", fullgraph=True)

        global decode_one_token, prefill
        decode_one_token = torch.compile(decode_one_token, mode="reduce-overhead", fullgraph=True)

        if compile_prefill:
            prefill = torch.compile(prefill, fullgraph=True, dynamic=True)

    questions = load_benchmark_data(bench_name)
    
    results = []
    speeds = []
    
    model_template = get_model_template(checkpoint_path.name)
    
    system_message = ("You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, "
                      "while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, "
                      "dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\n"
                      "If a question does not make any sense, or is not factually coherent, explain why instead of answering "
                      "something not correct. If you don't know the answer to a question, please don't share false information.")
    
    for question in tqdm(questions):
        torch.manual_seed(0)

        conv = get_conversation_template(model_template)
        conv.set_system_message(system_message)
        
        turns = []
        new_tokens = []
        wall_time = []

        for j in range(len(question["turns"])):
            qs = question["turns"][j]
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            
            encoded = encode_tokens(tokenizer, prompt, bos=True, device=device)
            
            start_time = time.time()
            output, metrics = generate(
                model,
                encoded,
                max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                interactive=False,
                batch_size=1,
                draft_model=draft_model,
                speculate_k=speculate_k,
                callback=lambda x: x,
            )
            end_time = time.time()
            
            output_ids = output[0][len(encoded):]
            if conv.stop_token_ids:
                stop_token_ids_index = [i for i, id in enumerate(output_ids) if id in conv.stop_token_ids]
                if stop_token_ids_index:
                    output_ids = output_ids[:stop_token_ids_index[0]]

            generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)
            
            if conv.stop_str:
                stop_str_index = generated_text.find(conv.stop_str)
                if stop_str_index != -1:
                    generated_text = generated_text[:stop_str_index]
            
            generated_text = generated_text.strip()
            
            walltime = end_time - start_time
            tokens_generated = len(output_ids)
            speed = tokens_generated / walltime
            
            turns.append(generated_text)
            new_tokens.append(tokens_generated)
            wall_time.append(walltime)
            speeds.append(speed)
            
            conv.messages[-1][-1] = generated_text

        results.append({
            'question_id': question.get('question_id', len(results)),
            'turns': turns,
            'tokens_generated': new_tokens,
            'walltime': wall_time,
            'speed': speeds[-len(turns):],
            'accept_counts': metrics['accept_counts'] if is_speculative else None,
        })
    
    if rank == 0 or rank is None:
        output_file = Path(f"./results_{bench_name}.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {output_file}")
        
        total_tokens = sum(sum(result['tokens_generated']) for result in results)
        total_time = sum(sum(result['walltime']) for result in results)
        overall_tokens_per_second = total_tokens / total_time
        all_speeds = [speed for result in results for speed in result['speed']]
        mean_speed = sum(all_speeds) / len(all_speeds)
        max_speed = max(all_speeds)
        
        print(f"Overall generation speed: {overall_tokens_per_second:.2f} tokens/second")
        print(f"Mean generation speed: {mean_speed:.2f} tokens/second")
        print(f"Max generation speed: {max_speed:.2f} tokens/second")

        if is_speculative:
            counts_aggregated = [sum(result['accept_counts']) for result in results]
            total_counts = sum(counts_aggregated)
            acceptance_probs = [count / total_counts for count in counts_aggregated]
            print(f"Acceptance probs: {acceptance_probs}")
            print(f"Mean Accepted: {sum([idx * count for idx, count in enumerate(counts_aggregated)]) / total_counts:.2f}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate model on benchmark datasets.')
    parser.add_argument('--bench-name', type=str, required=True, help='Name of the benchmark (e.g., mt_bench, human_eval)')
    parser.add_argument('--checkpoint-path', type=Path, required=True, help='Path to the model checkpoint')
    parser.add_argument('--max-new-tokens', type=int, default=100, help='Maximum number of new tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Temperature for sampling')
    parser.add_argument('--top-k', type=int, default=200, help='Top-k for sampling')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for computation')
    parser.add_argument('--draft-checkpoint-path', type=Path, default=None, help='Path to the draft model checkpoint for speculative decoding')
    parser.add_argument('--speculate-k', type=int, default=5, help='Speculative execution depth')
    parser.add_argument('--compile', action='store_true', help='Whether to compile the model')
    parser.add_argument('--compile-prefill', action='store_true', help='Whether to compile the prefill')
    
    args = parser.parse_args()
    main(args.bench_name, args.checkpoint_path, args.max_new_tokens, args.temperature, args.top_k, args.device,
         args.draft_checkpoint_path, args.speculate_k, args.compile, args.compile_prefill)