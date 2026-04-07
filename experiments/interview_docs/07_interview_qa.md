# 文档七：面试高频问题速查手册

## 模块一：Transformer 架构

**Q1: RMSNorm 和 LayerNorm 的区别？**
> RMSNorm 去掉了均值中心化步骤，只做均方根归一化，也没有偏移量β。计算公式是 `x / sqrt(mean(x²) + ε) × γ`。优点：计算更快，无需求均值；论文证明效果持平。LLaMA、MiniMind 等现代 LLM 全部使用 RMSNorm。

**Q2: RoPE 的原理是什么？为什么比绝对位置编码好？**
> RoPE 把位置信息编码成旋转矩阵，作用在 Q/K 上。关键性质：Q_i · K_j 的结果只依赖于 (i-j) 的相对距离，而不是绝对位置。因此 RoPE 天然支持相对位置感知，且在训练长度之外（外推）表现更好。MiniMind 用 YaRN 在 RoPE 基础上进一步扩展到 32K 上下文。

**Q3: GQA 是什么？和 MHA 的区别？**
> GQA（Grouped Query Attention）将 KV 头数量减少为 Q 头的一半（MiniMind：8 Q头 + 4 KV头）。优势：KV-Cache 减半，推理时显存和带宽占用降低。每 2 个 Q 头共享 1 组 KV，通过 `repeat_kv` 将 4 个 KV 头复制为 8 个配合 Q 头计算。

**Q4: KV-Cache 是什么？为什么能加速推理？**
> 自回归生成时，每步只输入最新的 1 个 token。如果没有 KV-Cache，需要把整个历史序列重新计算 K/V，复杂度 O(n²)。KV-Cache 将历史 K/V 存储下来，每步只计算新 token 的 K/V 并拼接，复杂度降为 O(n)。代价是显存随上下文长度线性增长。

**Q5: Flash Attention 是什么？什么时候用？**
> Flash Attention 是一种 IO 感知的注意力计算方法，将 softmax(QK^T)V 分块计算，避免将完整的注意力矩阵写入 HBM（显存），大幅减少 IO 开销。MiniMind 在训练和长序列推理时使用（`F.scaled_dot_product_attention`），在有 KV-Cache 的推理阶段退回手动实现。

**Q6: embed_tokens 和 lm_head 为什么共享权重？**
> 两者都是在"token 向量空间"中操作：embed_tokens 将 token_id 映射为 768 维向量（查表），lm_head 将 768 维向量映射回 vocab_size 维（反查表）。共享权重（Tied Weights）让两者保持语义一致：语义相近的 token 在输入和输出空间的相似性一致。同时节省约 4.9M 参数。

---

## 模块二：LoRA 与微调

**Q7: LoRA 的原理？为什么用低秩矩阵？**
> 假设权重更新 ΔW 是低秩的（实验证明微调时确实如此），则 ΔW ≈ B×A（B∈R^{d×r}, A∈R^{r×d}）。只训练 A 和 B，参数量从 d² 降为 2dr，当 r<<d 时大幅节省。初始化：B=0（保证初始 ΔW=0，不破坏预训练分布），A 高斯初始化。

**Q8: 本项目 LoRA 注入了哪些层？为什么？**
> 只注入方阵线性层（weight.shape[0] == weight.shape[1]），即 q_proj 和 o_proj（768×768）。FFN 的三个矩阵（gate/up/down）不是方阵所以跳过。k_proj 和 v_proj 因为 GQA 是 768×384 也跳过。每层 2 个注入点，共 16 个 LoRA，总参数 1.57M（2.4%）。

**Q9: SM 任务为什么会模式坍塌？如何解决？**
> 99.7% 的样本是 No，模型发现"全预测 No"可以让 cross-entropy loss 最小（因为 No 的条件概率接近 1），不需要真正理解语义。解决方案：(1) 1:1 降采样让正负样本平衡；(2) 引入 EM 多任务迁移补充正例；(3) 增大 rank（64）提升 LoRA 表达能力；(4) 足够的训练步数（1390步）。

**Q10: LoRA 合并（merge）的原理？**
> 训练后 W_new = W_old + B@A。合并就是直接把 B@A 加到 W_old 上，此后不再需要独立的 LoRA 模块。合并后推理速度与原模型完全一致，存储只需一个权重文件。

---

## 模块三：DPO 与 GRPO

**Q11: DPO 为什么不需要 Reward Model？**
> DPO 利用了最优策略和 reward 之间的数学关系：在 KL 约束下，最优策略 π*(y|x) ∝ π_ref(y|x)·exp(r(x,y)/β)，因此 r 可以被表达为 β·log[π/π_ref] + const。把这个代入 Bradley-Terry 偏好模型，就得到了不含 r 的 DPO loss，直接用偏好对优化策略。

**Q12: ref_model 在 DPO 中的作用？**
> 提供 KL 惩罚，防止策略偏离预训练分布太远。如果没有 KL 约束，模型可能通过压低所有 token 的概率来操纵 log-ratio，导致语言能力灾难性遗忘。KL 约束让模型在"提升偏好"和"保持流畅性"之间取得平衡。β 越大，约束越强。

**Q13: GRPO 和 PPO 的核心区别？**
> PPO 需要 Value Network（Critic）估计每个状态的期望奖励作为 baseline。GRPO 去掉 Value Network，用同一 prompt 的 G 个回复的 reward 均值作为 baseline（组内相对评估）。优势：(1) 省去 Critic 的训练复杂度和显存；(2) 组内 baseline 自然对 prompt 进行归一化（不同难度的 prompt reward 量级不同）。

**Q14: 为什么 DPO 初始 loss ≈ 0.693？**
> 训练初始时 policy = ref_model（权重相同），所有 log-ratio = 0，DPO loss = -log σ(0) = -log(0.5) = ln(2) ≈ 0.693。这是 DPO 的理论起点，表示模型对 chosen 和 rejected 完全无偏好。训练过程中 loss 从 0.693 降到 0.421，说明偏好对齐生效。

**Q15: GRPO 对 64M 模型效果为什么有限？**
> (1) 模型容量瓶颈：64M 参数的语义理解能力有限，reward model 给出的信号对应的语义改进超出了模型能力范围；(2) reward 信号噪声：小模型回复质量变化范围窄，组内 reward 方差小，梯度信号弱；(3) 策略空间小：RL 的探索空间受模型容量限制，64M 很快穷尽。

---

## 模块四：知识蒸馏

**Q16: 白盒和黑盒蒸馏的区别？本项目为什么用黑盒？**
> 白盒蒸馏：用 Teacher 的 soft logits（每个 token 的概率分布）指导 Student，loss = KL散度。需要 Teacher 和 Student 词表对齐。黑盒蒸馏：直接用 Teacher 的输出文本作为监督，Student 做普通 SFT。本项目中 Jellyfish（vocab=32000）和 MiniMind（vocab=6400）词表不兼容，无法做白盒蒸馏。

**Q17: 蒸馏为什么有效（理论上）？**
> Teacher 模型的 soft labels（整个概率分布）包含比 hard labels（只有 0/1）更丰富的信息。比如"猫"和"狗"的概率可能都高，说明 Teacher 认为这两者相似。Student 通过学习 soft labels 可以获得比直接用 hard labels 更好的泛化能力（Hinton 2015 KD 论文核心结论）。

**Q18: 本项目蒸馏 F1=0，怎么解释这是有价值的实验？**
> 揭示了一个重要结论：在极端类别不平衡场景下，黑盒蒸馏会原样复制 Teacher 输出的偏差分布（84K数据中99.7%是No，Mixtral的推理链也大多结论是No）。这验证了数据质量比算法复杂度更重要——我们的 LoRA 方案通过数据工程（平衡采样+多任务）解决了同样的问题，而更"高级"的蒸馏算法却失败了。这是有意义的对照实验。

---

## 模块五：系统设计与工程

**Q19: DDP（分布式训练）是怎么工作的？**
> DDP 将 batch 平均分给每张 GPU，每张 GPU 维护完整的模型副本，独立做前向传播和计算梯度，然后通过 AllReduce 对所有 GPU 的梯度求均值，每张 GPU 用相同的梯度更新模型，保持一致性。本项目使用 `torchrun --nproc_per_node=3` 启动 3 卡 DDP 训练 GRPO。

**Q20: Checkpoint 断点续训是如何实现的？**
> `lm_checkpoint` 函数每 10 步保存一次，内容包括：模型权重（half 精度节省空间）、optimizer 状态（Adam 的 m/v）、当前 epoch/step、SwanLab run_id。恢复时用 `SkipBatchSampler` 跳过已训练的 batch（通过 `skip_n` 参数），从断点处继续，不重复训练数据。

**Q21: AMP（混合精度）是怎么回事？**
> 前向传播和梯度计算用 bfloat16（16位，减少显存和计算）；梯度累加和权重更新用 float32（保持精度）。通过 `GradScaler` 解决 fp16 下梯度下溢问题（bf16 不需要 scaler）。MiniMind 使用 `torch.amp.autocast('cuda', dtype=torch.bfloat16)`。

**Q22: 推理时的 top-p（核采样）是什么？**
> 将所有 token 按概率从高到低排序，从最高开始累加，直到累加概率超过 p（如 0.95），只从这个截断后的词表中采样。这保留了高概率的 token，同时避免了极低概率的"意外"token 被采样。相比 top-k（固定取 k 个），top-p 能自适应词表大小——概率分布尖锐时取少，平坦时取多。

---

## 数字速查表（背下来，面试用）

| 项目 | 数值 |
|---|---|
| 总参数量 | 63.91M（约 64M） |
| hidden_size | 768 |
| 层数 | 8 |
| Q 头数 | 8 |
| KV 头数 | 4（GQA） |
| head_dim | 96 |
| 词表大小 | 6400 |
| LoRA rank | 64 |
| LoRA 参数量 | 1.57M（2.4%） |
| SM 数据正例比 | 0.3%（212/84345） |
| LoRA Precision | 0.905（平衡集） |
| LoRA Accuracy | 91.7%（全量集） |
| AttnRes Loss 提升 | -0.9% |
| AttnRes norm CV 下降 | -58%（0.368→0.153）|
| DPO 初始 Loss | 0.693（=ln2） |
| DPO 最终 Loss | 0.421 |
| Tool Use 通过率 | 6/8（75%） |
| 推理速度 | 174 tokens/s |
| 速度 vs Jellyfish-7B | 约 4.5x |
| 参数量 vs Jellyfish-7B | 1/109 |
