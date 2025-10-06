export HF_TOKEN="YOUR_HUGGINGFACE_KEY"


# Evaulate with Qwen/Qwen3-1.7B
torchrun --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_port $((RANDOM + 20000)) run.py --model_name Qwen/Qwen3-1.7B --dataset_name gsm8k --batch_size 512 --max_new_tokens 32768 --method swir --alpha 0.6
python merge.py --model_name Qwen/Qwen3-1.7B --dataset_name gsm8k --max_new_tokens 32768 --method swir

torchrun --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_port $((RANDOM + 20000)) run.py --model_name Qwen/Qwen3-1.7B --dataset_name math500 --batch_size 256 --max_new_tokens 32768 --method swir --alpha 0.5
python merge.py --model_name Qwen/Qwen3-1.7B --dataset_name math500 --max_new_tokens 32768 --method swir
 
torchrun --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_port $((RANDOM + 20000)) run.py --model_name Qwen/Qwen3-1.7B --dataset_name gpqa_diamond --batch_size 64 --max_new_tokens 32768 --method swir --alpha 1.0
python merge.py --model_name Qwen/Qwen3-1.7B --dataset_name gpqa_diamond --max_new_tokens 32768 --method swir 

torchrun --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_port $((RANDOM + 20000)) run.py --model_name Qwen/Qwen3-1.7B --dataset_name aime_2024 --batch_size 30 --max_new_tokens 38912 --method swir --alpha 0.5
python merge.py --model_name Qwen/Qwen3-1.7B --dataset_name aime_2024 --max_new_tokens 38912 --method swir

torchrun --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_port $((RANDOM + 20000)) run.py --model_name Qwen/Qwen3-1.7B --dataset_name aime_2025 --batch_size 30 --max_new_tokens 38912 --method swir --alpha 0.3
python merge.py --model_name Qwen/Qwen3-1.7B --dataset_name aime_2025  --max_new_tokens 38912 --method swir 


# Evaulate with Qwen/Qwen3-8B
torchrun --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_port $((RANDOM + 20000)) run.py --model_name Qwen/Qwen3-8B --dataset_name gsm8k --batch_size 256 --max_new_tokens 32768 --method swir --alpha 0.5
python merge.py --model_name Qwen/Qwen3-8B --dataset_name gsm8k --max_new_tokens 32768 --method swir

torchrun --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_port $((RANDOM + 20000)) run.py --model_name Qwen/Qwen3-8B --dataset_name math500 --batch_size 128 --max_new_tokens 32768 --method swir --alpha 1.0
python merge.py --model_name Qwen/Qwen3-8B --dataset_name math500 --max_new_tokens 32768 --method swir 

torchrun --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_port $((RANDOM + 20000)) run.py --model_name Qwen/Qwen3-8B --dataset_name gpqa_diamond --batch_size 64 --max_new_tokens 32768 --method swir --alpha 1.0
python merge.py --model_name Qwen/Qwen3-8B --dataset_name gpqa_diamond --max_new_tokens 32768 --method swir

torchrun --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_port $((RANDOM + 20000)) run.py --model_name Qwen/Qwen3-8B --dataset_name aime_2024 --batch_size 30 --max_new_tokens 38912 --method swir --alpha 0.9
python merge.py --model_name Qwen/Qwen3-8B --dataset_name aime_2024 --max_new_tokens 38912 --method swir 

torchrun --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_port $((RANDOM + 20000)) run.py --model_name Qwen/Qwen3-8B --dataset_name aime_2025 --batch_size 30 --max_new_tokens 38912 --method swir --alpha 0.9
python merge.py --model_name Qwen/Qwen3-8B --dataset_name aime_2025 --max_new_tokens 38912 --method swir 


# Evaulate with deepseek-ai/DeepSeek-R1-Distill-Llama-8B
torchrun --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_port $((RANDOM + 20000)) run.py --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B --dataset_name gsm8k --batch_size 256 --max_new_tokens 32768 --method swir --alpha 0.1
python merge.py --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B --dataset_name gsm8k --max_new_tokens 32768 --method swir

torchrun --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_port $((RANDOM + 20000)) run.py --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B --dataset_name math500 --batch_size 128 --max_new_tokens 32768 --method swir --alpha 0.5
python merge.py --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B --dataset_name math500 --max_new_tokens 32768 --method swir

torchrun --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_port $((RANDOM + 20000)) run.py --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B --dataset_name gpqa_diamond --batch_size 64 --max_new_tokens 32768 --method swir --alpha 0.7
python merge.py --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B --dataset_name gpqa_diamond --max_new_tokens 32768 --method swir 

torchrun --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_port $((RANDOM + 20000)) run.py --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B --dataset_name aime_2024 --batch_size 30 --max_new_tokens 38912 --method swir --alpha 0.65
python merge.py --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B --dataset_name aime_2024 --max_new_tokens 38912 --method swir

torchrun --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_port $((RANDOM + 20000)) run.py --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B --dataset_name aime_2025 --batch_size 30 --max_new_tokens 38912 --method swir --alpha 0.7
python merge.py --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B --dataset_name aime_2025 --max_new_tokens 38912 --method swir


# Evaulate with switch count control, e.g., with Qwen3-8B and gsm8k
torchrun --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_port $((RANDOM + 20000)) run.py --model_name Qwen/Qwen3-8B --dataset_name gsm8k --batch_size 256 --max_new_tokens 32768 --method swir --alpha 0.5 --max_switch_count 1
python merge.py --model_name Qwen/Qwen3-8B --dataset_name gsm8k --max_new_tokens 32768 --method swir

torchrun --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_port $((RANDOM + 20000)) run.py --model_name Qwen/Qwen3-8B --dataset_name gsm8k --batch_size 256 --max_new_tokens 32768 --method swir --alpha 0.5 --max_switch_count 2
python merge.py --model_name Qwen/Qwen3-8B --dataset_name gsm8k --max_new_tokens 32768 --method swir

...

torchrun --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_port $((RANDOM + 20000)) run.py --model_name Qwen/Qwen3-8B --dataset_name gsm8k --batch_size 256 --max_new_tokens 32768 --method swir --alpha 0.5 --max_switch_count 6
python merge.py --model_name Qwen/Qwen3-8B --dataset_name gsm8k --max_new_tokens 32768 --method swir