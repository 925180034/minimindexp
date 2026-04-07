# 文档一：整体架构总览

## 1. 项目全景图

```
输入文本
   │
   ▼
┌─────────────────────────────────────────────┐
│              Tokenizer (BPE)                │
│  词表大小 6400，中英文混合，自训练            │
└─────────────────────────────────────────────┘
   │  token ids: [1, 234, 56, 789, 2]
   ▼
┌─────────────────────────────────────────────┐
│         MiniMindForCausalLM (64M)           │
│                                             │
│  Embedding (6400 × 768)                     │
│       ↓                                     │
│  ┌──────────────────────────────────┐       │
│  │  MiniMindBlock × 8               │       │
│  │  ┌────────────────────────────┐  │       │
│  │  │  RMSNorm                   │  │       │
│  │  │  Attention (GQA + RoPE)    │  │       │
│  │  │  残差连接                   │  │       │
│  │  │  RMSNorm                   │  │       │
│  │  │  FeedForward (SwiGLU)      │  │       │
│  │  │  残差连接                   │  │       │
│  │  └────────────────────────────┘  │       │
│  └──────────────────────────────────┘       │
│       ↓                                     │
│  RMSNorm                                    │
│  lm_head (Linear 768 → 6400)               │
└─────────────────────────────────────────────┘
   │  logits: [batch, seq_len, 6400]
   ▼
next token prediction（自回归）
```

## 2. 关键超参数一览（面试背下来）

| 参数 | 值 | 含义 |
|---|---|---|
| hidden_size | 768 | 每个 token 的向量维度 |
| num_hidden_layers | 8 | Transformer 块的数量 |
| num_attention_heads | 8 | Q 头数 |
| num_key_value_heads | 4 | KV 头数（GQA，减半） |
| head_dim | 96 | 每个注意力头的维度（768/8） |
| intermediate_size | 约 2048 | FFN 中间层维度 |
| vocab_size | 6400 | 词表大小 |
| max_position_embeddings | 32768 | 最大上下文长度（YaRN 支持） |
| rope_theta | 1e6 | RoPE 基础频率 |
| 总参数量 | **63.91M** | 约 64M |

## 3. 训练全链路流水线

```
┌─────────────────────────────────────────────────────────────────┐
│                     训练全链路                                    │
├────────┬──────────┬────────┬───────────┬─────────┬─────────────┤
│ 预训练  │  全量SFT │  LoRA  │    DPO    │  GRPO   │   蒸馏SFT   │
│pretrain│ full_sft │  lora  │   align   │  rl     │  distill    │
├────────┼──────────┼────────┼───────────┼─────────┼─────────────┤
│自回归   │指令对齐   │轻量微调 │偏好对齐   │在线RL   │知识压缩      │
│预测下一 │让模型学会 │冻结主干 │区分好坏回答│group   │用大模型输出  │
│个 token │对话格式   │只训低秩 │无需RM     │relative │训小模型      │
└────────┴──────────┴────────┴───────────┴─────────┴─────────────┘
          ↓           ↓                             ↓
       pretrain_    full_sft_  lora_*_    dpo_    grpo_   distill_
       768.pth      768.pth    768.pth    768.pth  768.pth sft_768.pth
```

## 4. 文件结构速查

```
minimind/
├── model/
│   ├── model_minimind.py        ← 核心：Transformer 实现
│   ├── model_minimind_attnres.py← AttnRes 架构变体
│   ├── model_lora.py            ← LoRA 注入/保存/合并
│   └── tokenizer.json           ← BPE 词表（6400 tokens）
│
├── trainer/
│   ├── train_pretrain.py        ← 预训练
│   ├── train_full_sft.py        ← 全量 SFT
│   ├── train_lora.py            ← LoRA 微调
│   ├── train_dpo.py             ← DPO 偏好对齐
│   ├── train_grpo.py            ← GRPO 在线强化学习
│   ├── train_agent.py           ← Agentic RL（Tool Use）
│   ├── trainer_utils.py         ← 公共工具（LR调度/checkpoint/DDP）
│   └── rollout_engine.py        ← GRPO 生成引擎
│
├── dataset/
│   ├── lm_dataset.py            ← PretrainDataset / SFTDataset / DPODataset
│   └── *.jsonl                  ← 训练数据文件
│
├── experiments/                 ← 本项目自定义实验
│   ├── lora_schema_matching/    ← LoRA 微调实验
│   ├── attnres/                 ← AttnRes 对比实验
│   ├── distillation/            ← 知识蒸馏实验
│   └── interview_docs/          ← 本文档集
│
└── out/                         ← 训练好的权重
    ├── full_sft_768.pth
    ├── lora_schema_matching_768.pth
    ├── dpo_768.pth
    ├── grpo_768.pth
    └── distill_sft_768.pth
```

## 5. 面试必答：MiniMind 和标准 GPT 的区别

| 组件 | 标准 GPT-2 | MiniMind |
|---|---|---|
| 位置编码 | 绝对位置嵌入（可学习） | RoPE（旋转位置编码） |
| 归一化 | LayerNorm（Post-Norm） | RMSNorm（Pre-Norm） |
| 激活函数 | GELU | SiLU（SwiGLU 变体） |
| 注意力 | MHA（Q=K=V头数相同） | GQA（K/V头数减半） |
| 注意力QK归一化 | 无 | QK-Norm（RMSNorm on head_dim） |
| 上下文扩展 | 无 | YaRN RoPE Scaling |
| Flash Attention | 无 | 支持（PyTorch SDPA） |
| 权重共享 | 无 | embed_tokens 与 lm_head 共享权重 |
