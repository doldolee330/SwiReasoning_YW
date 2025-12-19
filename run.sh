export HF_TOKEN="YOUR_HUGGINGFACE_KEY"

############################################
# Evaluate with Qwen/Qwen3-1.7B
############################################

torchrun --nproc_per_node 4 --nnodes 1 --node_rank 0 --master_port $((RANDOM + 20000)) \
run.py --model_name Qwen/Qwen3-1.7B --dataset_name gsm8k \
--batch_size 64 --max_new_tokens 32768 --method c2f --alpha 0.6
python merge.py --model_name Qwen/Qwen3-1.7B --dataset_name gsm8k --max_new_tokens 32768 --method c2f


torchrun --nproc_per_node 4 --nnodes 1 --node_rank 0 --master_port $((RANDOM + 20000)) \
run.py --model_name Qwen/Qwen3-1.7B --dataset_name math500 \
--batch_size 64 --max_new_tokens 32768 --method c2f --alpha 0.5
python merge.py --model_name Qwen/Qwen3-1.7B --dataset_name math500 --max_new_tokens 32768 --method c2f


torchrun --nproc_per_node 4 --nnodes 1 --node_rank 0 --master_port $((RANDOM + 20000)) \
run.py --model_name Qwen/Qwen3-1.7B --dataset_name gpqa_diamond \
--batch_size 16 --max_new_tokens 32768 --method c2f --alpha 1.0
python merge.py --model_name Qwen/Qwen3-1.7B --dataset_name gpqa_diamond --max_new_tokens 32768 --method c2f


torchrun --nproc_per_node 4 --nnodes 1 --node_rank 0 --master_port $((RANDOM + 20000)) \
run.py --model_name Qwen/Qwen3-1.7B --dataset_name aime_2024 \
--batch_size 16 --max_new_tokens 38912 --method c2f --alpha 0.5
python merge.py --model_name Qwen/Qwen3-1.7B --dataset_name aime_2024 --max_new_tokens 38912 --method c2f


torchrun --nproc_per_node 4 --nnodes 1 --node_rank 0 --master_port $((RANDOM + 20000)) \
run.py --model_name Qwen/Qwen3-1.7B --dataset_name aime_2025 \
--batch_size 16 --max_new_tokens 38912 --method c2f --alpha 0.3
python merge.py --model_name Qwen/Qwen3-1.7B --dataset_name aime_2025 --max_new_tokens 38912 --method c2f


############################################
# Evaluate with Qwen/Qwen3-8B
############################################

torchrun --nproc_per_node 4 --nnodes 1 --node_rank 0 --master_port $((RANDOM + 20000)) \
run.py --model_name Qwen/Qwen3-8B --dataset_name gsm8k \
--batch_size 64 --max_new_tokens 32768 --method c2f --alpha 0.5
python merge.py --model_name Qwen/Qwen3-8B --dataset_name gsm8k --max_new_tokens 32768 --method c2f


torchrun --nproc_per_node 4 --nnodes 1 --node_rank 0 --master_port $((RANDOM + 20000)) \
run.py --model_name Qwen/Qwen3-8B --dataset_name math500 \
--batch_size 32 --max_new_tokens 32768 --method c2f --alpha 1.0
python merge.py --model_name Qwen/Qwen3-8B --dataset_name math500 --max_new_tokens 32768 --method c2f


torchrun --nproc_per_node 4 --nnodes 1 --node_rank 0 --master_port $((RANDOM + 20000)) \
run.py --model_name Qwen/Qwen3-8B --dataset_name gpqa_diamond \
--batch_size 16 --max_new_tokens 32768 --method c2f --alpha 1.0
python merge.py --model_name Qwen/Qwen3-8B --dataset_name gpqa_diamond --max_new_tokens 32768 --method c2f


torchrun --nproc_per_node 4 --nnodes 1 --node_rank 0 --master_port $((RANDOM + 20000)) \
run.py --model_name Qwen/Qwen3-8B --dataset_name aime_2024 \
--batch_size 16 --max_new_tokens 38912 --method c2f --alpha 0.9
python merge.py --model_name Qwen/Qwen3-8B --dataset_name aime_2024 --max_new_tokens 38912 --method c2f


torchrun --nproc_per_node 4 --nnodes 1 --node_rank 0 --master_port $((RANDOM + 20000)) \
run.py --model_name Qwen/Qwen3-8B --dataset_name aime_2025 \
--batch_size 16 --max_new_tokens 38912 --method c2f --alpha 0.9
python merge.py --model_name Qwen/Qwen3-8B --dataset_name aime_2025 --max_new_tokens 38912 --method c2f


############################################
# Evaluate with deepseek-ai/DeepSeek-R1-Distill-Llama-8B
############################################

torchrun --nproc_per_node 4 --nnodes 1 --node_rank 0 --master_port $((RANDOM + 20000)) \
run.py --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B --dataset_name gsm8k \
--batch_size 64 --max_new_tokens 32768 --method c2f --alpha 0.1
python merge.py --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B --dataset_name gsm8k --max_new_tokens 32768 --method c2f


torchrun --nproc_per_node 4 --nnodes 1 --node_rank 0 --master_port $((RANDOM + 20000)) \
run.py --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B --dataset_name math500 \
--batch_size 32 --max_new_tokens 32768 --method c2f --alpha 0.5
python merge.py --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B --dataset_name math500 --max_new_tokens 32768 --method c2f


torchrun --nproc_per_node 4 --nnodes 1 --node_rank 0 --master_port $((RANDOM + 20000)) \
run.py --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B --dataset_name gpqa_diamond \
--batch_size 16 --max_new_tokens 32768 --method c2f --alpha 0.7
python merge.py --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B --dataset_name gpqa_diamond --max_new_tokens 32768 --method c2f


torchrun --nproc_per_node 4 --nnodes 1 --node_rank 0 --master_port $((RANDOM + 20000)) \
run.py --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B --dataset_name aime_2024 \
--batch_size 16 --max_new_tokens 38912 --method c2f --alpha 0.65
python merge.py --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B --dataset_name aime_2024 --max_new_tokens 38912 --method c2f


torchrun --nproc_per_node 4 --nnodes 1 --node_rank 0 --master_port $((RANDOM + 20000)) \
run.py --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B --dataset_name aime_2025 \
--batch_size 16 --max_new_tokens 38912 --method c2f --alpha 0.7
python merge.py --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B --dataset_name aime_2025 --max_new_tokens 38912 --method c2f
