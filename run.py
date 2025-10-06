import os
import json
import argparse
from tqdm import tqdm
import torch
import torch.distributed as dist
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from generation_utils import (
    set_seed,
    get_math_symbols_ids,
    generate_cot,
    generate_swir,
)
from grader import answer_match


def main(args):
    set_seed(args.seed)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    
    if not dist.is_initialized():
        dist.init_process_group("nccl")

    model_name = args.model_name
    dataset_name = args.dataset_name
    batch_size = args.batch_size
    max_new_tokens = args.max_new_tokens
    n_samples = args.n_samples
    method = args.method
    alpha = args.alpha
    max_switch_count = args.max_switch_count

    gen_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "do_sample": args.do_sample,
        "max_new_tokens": args.max_new_tokens,
    }

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map={"": local_rank}
    )
    
    if dataset_name == "gsm8k":
        dataset = load_dataset("gsm8k", "main", split="test")
    elif dataset_name == "math500":
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    elif dataset_name == "aime_2024":
        dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
    elif dataset_name == "aime_2025":
        dataset = load_dataset("yentinglin/aime_2025", split="train")
    elif dataset_name == "gpqa_diamond":
        dataset = load_dataset("hendrydong/gpqa_diamond_mc", split="test")
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    if n_samples is not None:
        dataset = dataset.select(range(n_samples))
    total_len = len(dataset)
    chunk_size = (total_len + world_size - 1) // world_size
    start = local_rank * chunk_size
    end = min(start + chunk_size, total_len)
    dataset = dataset.select(range(start, end))
    
    correct = 0
    total = 0
    details = []
    total_token_lens = []
    correct_token_lens = []
    wrong_token_lens = []

    math_symbols_ids = get_math_symbols_ids(tokenizer)
    math_ids_tensor = torch.tensor(list(math_symbols_ids), device=model.device)
    
    for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating"):
        batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
        if args.dataset_name == "gsm8k":
            questions = batch["question"]
            golds = [str(a).split("####")[-1].strip() for a in batch["answer"]]
        elif args.dataset_name == "math500":
            questions = batch["problem"]
            golds = [str(a).strip() for a in batch["answer"]]
        elif args.dataset_name == "aime_2024":
            questions = batch["problem"]
            golds = [str(a).strip() for a in batch["answer"]]
        elif args.dataset_name == "aime_2025":
            questions = batch["problem"]
            golds = [str(a).strip() for a in batch["answer"]]
        elif args.dataset_name == "gpqa_diamond":
            questions = batch["problem"]
            golds = [str(a).strip() for a in batch["solution"]]
        prompts = [
            f"{q}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
            for q in questions
        ]
        messages_batch = [[{"role": "user", "content": prompt}] for prompt in prompts]
        texts = [
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            for messages in messages_batch
        ]
        model_inputs = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        ).to(model.device)
    
        with torch.no_grad():
            if method == "cot":
                # generated_ids = model.generate( 
                #     **model_inputs,
                #     **gen_kwargs,
                # )
                generated_ids = generate_cot( # better memory efficiency 
                    model,
                    tokenizer,
                    **model_inputs,   
                    **gen_kwargs,   
                )
            elif method == "cot_greedy":
                gen_kwargs["do_sample"] = False
                # generated_ids = model.generate( 
                #     **model_inputs,
                #     **gen_kwargs,
                # )
                generated_ids = generate_cot( # better memory efficiency
                    model,
                    tokenizer,
                    **model_inputs,   
                    **gen_kwargs,   
                )
            elif method == "swir":
                model_inputs["alpha_0"] = alpha
                model_inputs["max_switch_count"] = max_switch_count
                model_inputs["math_ids_tensor"] = math_ids_tensor
                model_inputs["convergence_words"] = "</think>" if "Qwen" in model_name else "\n\n</think>\n\n"
                generated_ids = generate_swir(
                    model,
                    tokenizer,
                    **model_inputs,   
                    **gen_kwargs,   
                )
        
        prompt_len = model_inputs["input_ids"].shape[1]
        preds = [
            tokenizer.decode(generated_ids[idx][prompt_len:], skip_special_tokens=True)
            for idx in range(len(questions))
        ]
    
        for idx in range(len(questions)):
            gold = golds[idx]
            question = questions[idx]
            pred = preds[idx]
            output_ids = generated_ids[idx][prompt_len:].tolist()
            try:
                eot_id = 128014 if "Llama" in model_name else 151668
                index = len(output_ids) - output_ids[::-1].index(eot_id)
            except ValueError:
                index = 0
            thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip()
            answer_content = pred[len(thinking_content):]
            is_correct, prediction = answer_match(dataset_name, answer_content, gold)
            correct += int(is_correct)
            total += 1
            details.append({
                "question": question,
                "gold": gold,
                "prediction": prediction,
                "correct": is_correct,
                "thinking": thinking_content,
                "answer_content": answer_content,
            })
            if total % 20 == 0:
                print(f"Processed {total} examples, Accuracy: {correct/total:.2%}")
                
            output_token_ids = tokenizer.encode(pred, add_special_tokens=False)
            total_token_len = len(output_token_ids)
            total_token_lens.append(total_token_len)
            if is_correct:
                correct_token_lens.append(total_token_len)
            else:
                wrong_token_lens.append(total_token_len)

    print(f"Total: {total}, Correct: {correct}, Accuracy: {correct/total:.2%}")
    
    avg = lambda l: float(sum(l)) / len(l) if l else 0.0
    length_stats = {
        "max_new_tokens": max_new_tokens,
        "avg_total_token_len": avg(total_token_lens),
        "correct_avg_total_token_len": avg(correct_token_lens),
        "wrong_avg_total_token_len": avg(wrong_token_lens),
    }
    
    result = {
        "accuracy": correct / total if total > 0 else 0.0,
        "total": total,
        "correct": correct,
        "length_stats": length_stats,
        "details": details
    }
    
    os.makedirs("logs", exist_ok=True)
    model_name = model_name.split("/")[-1]
    log_path = f"logs/{model_name}_{dataset_name}_{method}_{max_new_tokens}_rank{local_rank}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[Rank {local_rank}] log written: {log_path}")


if __name__ == "__main__":
    parser  = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen3-8B")
    parser.add_argument('--dataset_name', type=str, default="gsm8k")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_samples', type=int, default=None) 
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--do_sample", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--max_new_tokens', type=int, default=38912)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--method", type=str, default="swir", choices=["swir", "cot", "cot_greedy"])
    parser.add_argument('--alpha', type=float, default=1.0) # swir-specific
    parser.add_argument('--max_switch_count', type=int, default=None) # swir-specific
    args = parser.parse_args()
    main(args)
