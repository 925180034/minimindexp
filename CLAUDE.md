# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MiniMind is an educational project for training ultra-small language models (~26M–64M parameters) from scratch using native PyTorch. The goal is to demonstrate a complete LLM pipeline (pretrain → SFT → RL alignment) on a single consumer GPU (3090) for ~3 CNY.

**Key design principle**: Everything is implemented in vanilla PyTorch — no high-level abstractions. The codebase prioritizes readability and educational value over production optimization.

## Setup

```bash
pip install -r requirements.txt
# Python >= 3.10 recommended
```

No build step required. PyTorch must be installed separately (commented out in requirements.txt to allow version flexibility).

## Training Pipeline

Run from the `trainer/` directory unless noted:

```bash
# 1. Pretrain on raw text
python train_pretrain.py

# 2. Supervised Fine-Tuning
python train_full_sft.py

# 3. LoRA fine-tuning (parameter-efficient)
python train_lora.py

# 4. Reinforcement learning variants
python train_dpo.py      # Direct Preference Optimization
python train_ppo.py      # PPO
python train_grpo.py     # GRPO

# Resume from checkpoint (all trainers support this)
python train_pretrain.py --from_resume 1

# Distributed training (DDP)
torchrun --nproc_per_node=N train_pretrain.py
```

## Inference

```bash
# Interactive chat with pre-trained model
python eval_llm.py --load_from ./minimind-3

# With a specific weight checkpoint type
python eval_llm.py --weight full_sft

# With LoRA adapter
python eval_llm.py --weight full_sft --lora_weight lora_medical

# With chain-of-thought thinking
python eval_llm.py --open_thinking 1

# Multi-turn conversation (must be even number)
python eval_llm.py --historys 4
```

## Serving

```bash
# OpenAI-compatible API server
cd scripts && python serve_openai_api.py

# Streamlit web UI
cd scripts && python web_demo.py

# Tool-call evaluation
cd scripts && python eval_toolcall.py --weight full_sft

# Convert to HuggingFace safetensors format
cd scripts && python convert_model.py
```

## Architecture

### Model (`model/`)
- **`model_minimind.py`**: Core transformer implementation — `MiniMindConfig` (extends `PretrainedConfig`), `RMSNorm`, `Attention` (grouped-query + RoPE), `MiniMindMLP`, `MiniMindBlock`, and `MiniMindForCausalLM`. Supports both dense and MoE variants. YaRN RoPE scaling for long-context (up to 32K tokens).
- **`model_lora.py`**: LoRA adapter — `apply_lora()` injects into all square linear layers; `merge_lora()` folds weights back for deployment.
- **`tokenizer.json`**: Custom BPE tokenizer with 6,400-token vocabulary (Chinese/English).

### Training (`trainer/`)
- **`trainer_utils.py`**: Shared infrastructure — cosine LR scheduling (`get_lr`), DDP initialization, checkpoint save/load (`lm_checkpoint`), model init (`init_model`), `SkipBatchSampler` for resuming mid-epoch.
- **`rollout_engine.py`**: Generation engine used by PPO/GRPO for online rollouts.
- All training scripts follow the same pattern: argparse → dataset/dataloader → model init → training loop with gradient accumulation + AMP → checkpoint.

### Data (`dataset/`)
- **`lm_dataset.py`**: `PretrainDataset` (raw text), `SFTDataset` (chat conversations). Training data is JSONL with HuggingFace `datasets` format.
- Dataset format for SFT: `{"conversations": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`

### Checkpoints
- Saved to `./out/` during training (configurable via `--save_dir`)
- Intermediate checkpoints go to `./checkpoints/`
- All trainers support `--from_resume 1` to continue from the latest checkpoint

## Key Configuration

Model dimensions are set in `MiniMindConfig` (in `model/model_minimind.py`). The pre-trained `minimind-3` model uses:
- Hidden size: 768, Layers: 8, Attention heads: 8 (4 KV heads for GQA)
- Vocab: 6,400, Max context: 32,768 tokens

Training hyperparameters are passed via CLI args to each training script (`--batch_size`, `--lr`, `--epochs`, `--seq_len`, etc.).

## No Test Suite

There is no automated test suite. Validation is done by running `eval_llm.py` interactively or via `eval_toolcall.py` for tool-use capability.
