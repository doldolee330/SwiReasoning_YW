export HF_TOKEN="YOUR_HUGGINGFACE_KEY"


# Chatting with different methods
python run_chat.py --model_name Qwen/Qwen3-8B --method swir

python run_chat.py --model_name Qwen/Qwen3-8B --method cot

python run_chat.py --model_name Qwen/Qwen3-8B --method cot_greedy


# Trying different reasoning models (not an exhaustive list)
python run_chat.py --model_name Qwen/Qwen3-1.7B --method swir 

python run_chat.py --model_name Qwen/Qwen3-0.6B --method swir 

python run_chat.py --model_name Qwen/Qwen3-4B --method swir 

python run_chat.py --model_name Qwen/Qwen3-14B --method swir 

python run_chat.py --model_name Qwen/Qwen3-32B --method swir 

python run_chat.py --model_name Qwen/Qwen3-30B-A3B --method swir 

python run_chat.py --model_name Qwen/Qwen3-30B-A3B-Thinking-2507 --method swir

python run_chat.py --model_name Qwen/Qwen3-30B-A3B-Thinking-2507-FP8 --method swir

python run_chat.py --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B --method swir 

python run_chat.py --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-70B --method swir 

python run_chat.py --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --method swir 

python run_chat.py --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --method swir 

python run_chat.py --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --method swir 

python run_chat.py --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --method swir 

python run_chat.py --model_name Qwen/QwQ-32B --method swir 

...

