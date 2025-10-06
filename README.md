
<div align="center">
<h1>SwiReasoning: Switch-Thinking in Latent and Explicit for Pareto-Superior Reasoning LLMs</h1>
</div>

<p align="center">
    <a href="">
        <img alt="ArXiv" src="https://img.shields.io/badge/arXiv-2507.14204-B31B1B?logo=arxiv" />
    </a>
    <a href="">
        <img alt="Webiste" src="https://img.shields.io/badge/website-link-4285F4?logo=googleearth" />
    </a><br>
</p>


## 👀 TL;DR
SwiReasoning is a training-free method for pareto-superior reasoning LLMs that dynamically switches between explicit and latent thinking, with a switch count control mechanism to suppress overthinking.

![swir](assets/method.png)

https://github.com/user-attachments/assets/2c917cfe-8b10-4af4-91b2-1a9c45228e1c

## ⚙️ Getting Started

### Clone project
``` bash
git clone https://github.com/sdc17/SwiReasoning.git
cd SwiReasoning
```

### Environment setup
```bash
conda create -n swir python=3.12
conda activate swir
pip install -r requirements.txt
```

## 💻 Interactive Chat

```bash
python run_chat.py --model_name Qwen/Qwen3-8B --method swir --max_switch_count 2
```

* Increase `--max_switch_count` to enable more thinking rounds (default is 2).
* Modify `--model_name` to try different reasoning LLMs.

```bash
Commands:
  exit or q -> [Exit]
  switch <N|none> -> [Set] swir max_switch_count = N (integer >= 1) or None (disabled)
  method <swir|cot|cot_greedy> -> [Set] generation method
```
* Please check [run_chat.sh](./run_chat.sh) for more examples.

## 📈 Evaluation

```bash
# Evaluate without switch count control
torchrun --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_port $((RANDOM + 20000)) run.py --model_name Qwen/Qwen3-1.7B \
    --dataset_name gsm8k --batch_size 512 --max_new_tokens 32768 --method swir --alpha 0.6
python merge.py --model_name Qwen/Qwen3-1.7B --dataset_name gsm8k --max_new_tokens 32768 --method swir

# Evaluate with switch count control
torchrun --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_port $((RANDOM + 20000)) run.py --model_name Qwen/Qwen3-8B \
    --dataset_name gsm8k --batch_size 256 --max_new_tokens 32768 --method swir --alpha 0.5 --max_switch_count 2
python merge.py --model_name Qwen/Qwen3-8B --dataset_name gsm8k --max_new_tokens 32768 --method swir

```
* Increase ``--nproc_per_node`` to enable faster evaluation on multiple GPUs. 
* Modify ``--model_name`` and ``--dataset_name`` for evaluation with different models and datasets.
* Please check [run.sh](./run.sh) for more examples.


