# MiniMind 学习解读大纲

> 目标：完全吃透 MiniMind 代码框架及个人改造内容，应对技术面试。
> 整体顺序：**先吃透基座 → 再过一遍标准训练全链路 → 最后专攻个人改造与 PRISM**

---
## 简历内容
Tiny-Tabular-LLM：面向结构化数据的轻量级 LLM 全链路训练框架
项目背景： 在大规模多智能体图表检索与匹配场景中，受限于千亿级大模型的高昂调用成本与并发延迟，基于 MiniMind 开源框架，完整覆盖 Tokenizer 构建、自回归预训练到 RLHF 对齐的全技术栈。
1.底座构建与架构重构：自主实现含 RoPE、SwiGLU、GQA、KV-Cache 的 Transformer 骨架。针对深层 PreNorm 隐状态稀释问题，复现 Moonshot AI 提出的 Attention Residuals 架构，以可学习 Softmax 权重替代固定残差累加，同等算力下 Loss 降低 0.9%，各层输出变异系数（CV）从 0.368 降至 0.153，验证其缓解梯度稀释的底层优势。
2.结构化数据序列化与微调：针对 Schema Matching 任务，将 2D 表头结构化序列化为 JSON 指令格式。面对极端类别不平衡（99.7% 负样本）引发的 LoRA 模式坍塌，构建严格 1:1 平衡数据集并引入 SM+EM 多任务联合训练迁移稀少正例特征，Precision 从 0.50 大幅提升至 0.905，解决稀少正例场景下的特征解耦难题。
3.后训练与 Agentic RL 对齐：独立搭建偏好对齐全链路：DPO Loss 从理论起点 ln2≈0.693 稳健收敛至 0.421；GRPO（CISPO loss）阶段通过量化实验论证了 64M 极小模型的 RL 容量瓶颈（SFT 收益显著优于在线 RL），为轻量级模型对齐提供实证依据。跑通 Tool-Use 与自适应推理，8 项工具评测达 6/8 通过率，支持链式/并行调用。
4.黑盒知识蒸馏：以 Mixtral-8x7B 生成的 CoT 为监督信号进行黑盒蒸馏，将结构化推理能力压缩至 64M 底座。推理吞吐量达 174 tokens/s（同类 7B 模型的 4.5 倍），参数量仅为 1/109，作为极速判别节点接入多智能体系统，在保障 SOTA 精度的同时实现降本增效。

## 学习路线总览

```
阶段一：基础设施         tokenizer / trainer_utils / config
        ↓
阶段二：模型架构核心     model_minimind.py（RoPE / GQA / MoE / 前向传播）
        ↓
阶段三：标准训练全链路   pretrain → sft → lora → dpo → ppo/grpo → tool/agent_rl → 蒸馏
        ↓
阶段四：个人改造与实验   AttnRes / LoRA SM / GRPO 退化组 / 蒸馏失败案例
```

---

## 阶段一：基础设施（约 0.5 天）

不要直接冲模型架构，先搞清楚"数据怎么变成模型能吃的张量"。

- **Tokenizer 训练代码**：BPE + ByteLevel 怎么训练出来；`<tool_call>`、`<think>` 等特殊 token 怎么预留 buffer
- **`trainer/trainer_utils.py`**：
  - `get_lr`：cosine 衰减的学习率公式
  - `init_distributed_mode`：DDP 怎么启动、`LOCAL_RANK` 怎么用
  - `lm_checkpoint`：断点续训机制（resume 路径、权重命名规则）
  - `get_model_params`：怎么统计 MoE 的激活参数（Active）vs 总参数（Total）
- **`MiniMindConfig`**：逐个超参数过一遍，重点关注隐含的维度关系，例如 `head_dim = hidden_size // num_attention_heads`

**面试价值**：这部分容易被忽略但体现工程素养——断点续训、DDP 初始化等细节是常见追问点。

---

## 阶段二：模型架构核心（重点，约 1-2 天）

全部精力放在 `model/model_minimind.py`，建议按以下顺序读：

1. **RMSNorm**：为什么去掉均值中心化也不影响效果
2. **RoPE（`precompute_freqs_cis`）**：旋转矩阵怎么注入位置信息；YaRN 的 NTK-by-parts（`beta_fast`/`beta_slow`）怎么做长文本外推
3. **Attention（GQA）**：8 个 query head、4 个 kv head 怎么分组共享；矩阵维度如何变化
4. **FeedForward（SwiGLU）**：门控机制的设计动机
5. **MOEFeedForward**：路由怎么选 top-1 专家；`aux_loss`（负载均衡损失）怎么计算
6. **MiniMindBlock → MiniMindModel → MiniMindForCausalLM**：整个前向传播怎么串起来；`generate` 方法的 KV Cache 推理逻辑

### 自测标准
拿一张白纸，不看代码，画出一个 token 从 embedding 到 logits 的完整数据流，标出每一步的张量维度变化（如 `[batch, seq, hidden]` → `[batch, q_heads, seq, head_dim]`）。

---

## 阶段三：标准训练全链路（约 2-3 天）

严格按训练阶段顺序读对应的 `trainer/train_*.py`：

| 顺序 | 脚本 | 核心关注点 |
|---|---|---|
| 1 | `train_pretrain.py` | loss 怎么算：shift labels 做 cross entropy |
| 2 | `train_full_sft.py` | 与 pretrain 的区别：指令掩码，只在 response 部分算 loss |
| 3 | `train_lora.py` | 低秩矩阵插入哪些层；参数量怎么算 |
| 4 | `train_dpo.py` | 为什么不需要 reward model；partition function 怎么在 loss 推导中被消掉 |
| 5 | `train_ppo.py` / `train_grpo.py` | 对比看：GRPO 用组内归一化代替 value network 的动机；CISPO 的 `.detach()` 在哪一行起作用 |
| 6 | `train_agent.py` | 多轮工具调用的 rollout 怎么组织（Tool Use / Agentic RL） |
| 7 | 蒸馏脚本 | 黑盒蒸馏数据怎么构造（Mixtral-8x7B 生成的 CoT chains） |

**阅读技巧**：每个脚本先看 `argparse` 部分，超参数本身就是对算法设计的提示（如 GRPO 的 `num_generations`、PPO 的 `clip_epsilon`）。

---

## 阶段四：个人改造与实验（面试最该讲透的部分）

把这部分单独拎出来，对照阶段二 / 三的原生代码看差异：

- **`model_minimind_attnres.py`**：对照 `MiniMindModel.forward` 看 `prev_hiddens` 怎么传递；学习型 softmax 权重怎么替代固定残差累加；CV 从 0.368 降到 0.153 的实验验证
- **LoRA Schema Matching 实验**：1:1 平衡采样的代码逻辑；Precision 从 0.50 提升到 0.905 背后做了什么（多任务联合训练迁移稀少正例特征）
- **GRPO 在 64M 模型上的退化组现象**：对照 `train_grpo.py` 里组内归一化 `(R-μ)/σ` 那段代码，理解"模型容量不足导致 rollout 多样性塌陷"这一负面结果
- **知识蒸馏 F1=0 的失败案例**：黑盒蒸馏把数据不平衡问题也蒸馏进了模型——这是很好的面试素材，体现实验严谨性而非单纯堆数字

---

## 面试串讲自检清单

- [ ] 能否不看代码画出 `MiniMindBlock` 的前向流程（Pre-Norm + Attention + 残差 + Pre-Norm + MLP + 残差）
- [ ] 能否解释为什么 AttnRes 把残差流 CV 从 0.368 降到 0.153（Pre-Norm 导致的范数累积问题）
- [ ] 能否说清 DPO loss 初始值 ln2≈0.693 的数学来源，以及 GRPO 退化组失效的根本原因
- [ ] 能否独立讲出 LoRA SM 任务从 Precision 0.50 到 0.905 的关键设计决策（平衡重采样 vs focal loss）
- [ ] 能否解释知识蒸馏 F1=0 失败案例背后的根本原因（类别不平衡被蒸馏进模型）

---

## 关键参考文件清单

```
model/
├── model_minimind.py          模型架构核心（必读）
└── model_minimind_attnres.py  AttnRes 改造版本（对照阅读）

trainer/
├── trainer_utils.py           训练工具函数
├── train_pretrain.py
├── train_full_sft.py
├── train_lora.py
├── train_dpo.py
├── train_ppo.py
├── train_grpo.py
├── train_agent.py
└── 蒸馏相关脚本

experiments/
├── PLAN.md                    实验记录与简历数字来源
├── lora_schema_matching/
├── attnres/
├── dpo/
├── grpo/
├── tool_use/
└── distillation/
```
