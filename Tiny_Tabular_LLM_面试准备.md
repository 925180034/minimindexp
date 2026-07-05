# Tiny-Tabular-LLM 项目面试完全手册

> 适用岗位：算法工程师（LLM / NLP 方向）
> 项目周期：2026.01 – 2026.04

---

## 目录

1. [项目完整介绍](#一项目完整介绍)
2. [模块一：底座构建与架构重组](#二模块一底座构建与架构重组)
3. [模块二：结构化数据序列化与微调](#三模块二结构化数据序列化与微调)
4. [模块三：后训练与 Agentic RL 对齐](#四模块三后训练与-agentic-rl-对齐)
5. [模块四：黑盒知识蒸馏](#五模块四黑盒知识蒸馏)
6. [综合压轴题](#六综合压轴题)
7. [延伸基础题（高频通用考点）](#七延伸基础题高频通用考点)
8. [快速背诵卡片](#八快速背诵卡片)
9. [面试讲述模板](#九面试讲述模板)

---

**关于覆盖范围的说明**：第一至六部分是项目定制题，聚焦"你做了什么、为什么这么做"；第七部分是**从项目自然延伸出的通用高频题**——面试官问完项目细节后，大概率会往下追问的基础概念。这部分不是项目独有的，但每一题都尽量绑定回项目里的具体实现，方便你用自己的经历举例而不是背书。

---

## 一、项目完整介绍

### 1.1 项目背景与动机

**项目名称**：Tiny-Tabular-LLM：面向结构化数据的轻量级 LLM 全链路训练框架

**核心动机**：在多智能体图表检索与匹配场景中（对应我的 WWW 2026 论文），使用千亿级大模型作为 MatcherAgent 的骨干存在两个根本瓶颈：

1. **调用成本高**：API 费用随请求量线性增长，在大规模数据湖检索场景中不可持续
2. **并发延迟大**：外部 API 的网络 RTT 与 rate limit 导致多智能体流水线吞吐严重受限

因此，**核心问题是**：能否训练一个极小参数量（~64M）的专用模型，在结构化数据匹配任务上逼近 7B 级别大模型的精度，同时推理速度提升 4× 以上？

**技术路径**：基于 MiniMind 开源框架，完整覆盖 Tokenizer 构建 → 自回归预训练 → SFT → LoRA 微调 → DPO/GRPO 对齐 → 黑盒知识蒸馏的全技术栈，并在多个环节引入针对结构化数据场景的原创改进。

---

### 1.2 项目架构全貌

```
Tiny-Tabular-LLM 训练流水线
│
├── 底座构建
│   ├── Transformer 骨架：RoPE + SwiGLU + GQA + KV-Cache
│   ├── 架构改进：Attention Residuals（可学习 softmax 残差权重）
│   └── 自回归预训练（原始文本语料）
│
├── 任务适配（SFT + LoRA）
│   ├── 2D 表头结构序列化 → JSON 指令格式
│   ├── 极端类别不平衡处理（99.7% 负样本）
│   └── SM + EM 多任务联合训练（Precision 0.50 → 0.905）
│
├── 后训练对齐
│   ├── DPO（偏好对齐，Loss: ln2 → 0.421）
│   ├── GRPO + CISPO（小模型 RL 容量验证）
│   └── Agentic RL（Tool-Use，6/8 通过率）
│
└── 部署压缩
    ├── 黑盒知识蒸馏（Mixtral-8x7B CoT → MiniMind-64M）
    └── 推理吞吐 174 tokens/s（Jellyfish-7B 的 ~4.5×）
```

---

### 1.3 核心技术指标

| 指标 | 数值 | 对比基准 |
|------|------|----------|
| 模型参数量 | 64M | Jellyfish-7B = 7B（1/109） |
| Schema Matching Precision（平衡候选集） | 0.905 | 基线 full_sft = 0.50 |
| 推理吞吐量 | 174 tokens/s | Jellyfish-7B ≈ 30-50 tokens/s |
| Attention Residuals CV 改善 | 0.368 → 0.153（↓58%） | 标准 PreNorm |
| DPO Loss 收敛 | ln2≈0.693 → 0.421 | 初始健康检查点 → 偏好学习生效 |
| Tool-Use 通过率 | 6/8 | — |

**指标口径必须主动说明**：

- `Precision=0.905` 指的是**平衡候选集/验证集**（48 Yes + 48 No）上的正类 Precision，不是全量极不平衡候选空间上的 Precision。
- 全量测试集有 11,936 条候选对，其中 Yes 只有 48 条。LoRA SM+EM 在该分布上 Accuracy=0.917，但 Precision=0.019、Recall=0.396、F1=0.037。这个结果说明：64M 模型适合作为**高吞吐候选过滤/重排节点**，不应被描述成“直接穷举全量候选即可高精度替代 7B”。
- 线上方案应该搭配候选召回、阈值校准、Precision@K/PR-AUC 评估和大模型兜底路由。面试时先把口径说清楚，反而会显得实验严谨。

---

## 二、模块一：底座构建与架构重组

### Q1：请介绍项目的基础 Transformer 架构，为什么选择这几个组件？

**回答要点**：

项目底座包含四个关键组件，每一个都有明确的工程动机：

**① RoPE（旋转位置编码）**

相比绝对位置编码，RoPE 将位置信息编码为 query/key 向量的旋转变换，数学形式为：

$$q_m \cdot k_n = \text{Re}[(\mathbf{W}_q x_m) \odot e^{im\theta}] \cdot \text{Re}[(\mathbf{W}_k x_n) \odot e^{in\theta}]$$

内积只依赖相对位置 $(m-n)$，因此具备**长度外推能力**（配合 YaRN 可外推到 32K）。对于表格数据，行列关系本质上是相对位置关系，RoPE 在语义上也更合适。

**② SwiGLU**

$$\text{SwiGLU}(x, W, V, b, c) = \text{Swish}(xW + b) \odot (xV + c)$$

相比 ReLU/GELU，SwiGLU 引入了门控机制，可以自适应地抑制无关特征，在 PaLM 等工作中被证明在相同参数量下效果更好。对于结构化数据的稀疏激活特性，门控尤其有意义。

**③ GQA（Grouped Query Attention）**

MHA 每个 head 都有独立 KV，显存占用为 $2 \times n_{head} \times d \times L$。GQA 将多个 query head 共享一组 KV head，在不显著损失精度的前提下将 KV Cache 显存降低 $n_{head}/n_{kv}$ 倍，对长序列推理意义显著。

**④ KV-Cache**

自回归生成时，每步都会重新计算所有历史 token 的 K/V，时间复杂度为 $O(n^2)$。KV-Cache 将已计算的 K/V 存储下来，推理时只计算新 token，将复杂度降为 $O(n)$，是工程部署的基础。

---

### Q2：你提到实现了 Attention Residuals 架构改进，具体是什么？为什么能缓解梯度稀疏？

**回答要点**：

**问题背景**：标准 Transformer 的残差连接是固定等权叠加：

$$h^{(l+1)} = h^{(l)} + f^{(l)}(h^{(l)})$$

在深层网络中，靠近输入端的层梯度会因反向传播路径过长而衰减（梯度稀疏/消失），导致底层参数更新缓慢，即 **deep PreNorm 隐状态稀疏问题**。

**Attention Residuals 的改进**（来自 Moonshot AI，arXiv:2603.15031）：

更准确地说，我在 MiniMind 里改的是 **attention 子层的 residual branch**：标准 PreNorm block 里 attention 输出会加到上一层 hidden state 上，而 AttnRes 把这个“上一层 hidden state”替换成对所有历史 hidden state 的 softmax 加权聚合：

$$r^{(l)} = \sum_{k=0}^{l-1} \alpha_k^{(l)} \cdot h^{(k)}, \quad
\alpha^{(l)} = \text{softmax}(\text{score}(h^{(0)},...,h^{(l-1)}))$$

随后再计算：

$$\tilde{h}^{(l)} = r^{(l)} + \text{Attn}(\text{Norm}(h^{(l-1)}))$$

代码实现中，用每个历史 hidden state 的 sequence mean 与一个可学习 query 做打分，再沿 depth 维度 softmax，得到每层对历史层的动态选择权重。因此不要把它说成“整层输出完全等于历史层加权平均”，而应说成“attention 残差分支由固定上一层输入变成了历史层的可学习加权聚合”。

**为什么能缓解梯度稀疏**：

- 标准残差中，底层的梯度信号必须穿过所有上层才能传回，路径长、衰减大
- Attention Residuals 建立了每层到所有历史层的**直接连接**，梯度可以绕过中间层直接回传，相当于在计算图中添加了大量"捷径"
- 可学习权重使模型能动态决定每层应该"参考多远的历史"

**实验验证**：

在 MiniMind-64M 上对比实验：

| 指标 | 标准 PreNorm | Attention Residuals |
|------|-------------|---------------------|
| 各层输出 hidden state norm 的 CV | 0.368 | 0.153（↓58%） |
| Loss（等算力） | 基准 | 降低 ~0.9% |

CV（变异系数）从 0.368 降到 0.153，说明各层隐状态的尺度分布更均匀，即梯度传播更顺畅，底层参数得到了更充分的更新。

---

### Q3：PreNorm 和 PostNorm 的区别是什么？你们用的是哪种？

**回答要点**：

**PostNorm**（原始 Transformer）：

$$h^{(l+1)} = \text{LayerNorm}(h^{(l)} + f^{(l)}(h^{(l)}))$$

归一化作用在残差加和之后，梯度分析表明训练初期不稳定，需要 warm-up，但最终精度往往略高。

**PreNorm**（现代大模型主流）：

$$h^{(l+1)} = h^{(l)} + f^{(l)}(\text{LayerNorm}(h^{(l)}))$$

归一化作用在子层输入，梯度直接通过残差路径回传，训练更稳定，可以省去 warm-up 或使用较大学习率，工程友好。

**本项目**：使用 PreNorm（与 Llama/Qwen 等现代架构一致），并在 PreNorm 的基础上叠加 Attention Residuals 改进。引入 Attention Residuals 正是针对 PreNorm 在深层网络中隐状态稀疏问题的定向解决方案。

---

### Q4：RoPE 位置编码是怎么实现的？

**回答要点**：

RoPE 的核心思想是将位置 $m$ 编码为旋转矩阵 $\mathbf{R}_m$，使得：

$$q_m^T k_n = (R_m q)^T (R_n k) = q^T R_{n-m} k$$

只依赖相对位置差 $(n-m)$，而非绝对位置。

**实现细节**：

```python
# 预计算旋转角度
theta = 1.0 / (10000 ** (2i/d)) for i in [0, d/2)

# 对 query/key 的相邻维度两两配对，施加旋转
# [x1, x2] → [x1*cos(mθ) - x2*sin(mθ), x1*sin(mθ) + x2*cos(mθ)]
```

频率 $\theta_i = 10000^{-2i/d}$ 使得低维度对应高频（短距离关系），高维度对应低频（长距离关系），形成天然的多尺度位置表示。

---

## 三、模块二：结构化数据序列化与微调

### Q5：你是怎么把二维表头结构序列化为 JSON 指令格式的？

**回答要点**：

**问题**：LLM 的输入是一维 token 序列，而 Schema Matching 任务的输入是两个表的列信息（列名、数据类型、统计特征、采样值），需要将这些结构化信息无损地转换为文本。

**序列化方案**（参考 Jellyfish 论文的 prompt 设计）：

```json
{
  "instruction": "判断以下两个属性是否语义等价，输出 Yes 或 No",
  "source_column": {
    "name": "employee_id",
    "type": "INTEGER",
    "sample_values": [1001, 1002, 1003],
    "description": "员工唯一标识符"
  },
  "target_column": {
    "name": "staff_no",
    "type": "VARCHAR",
    "sample_values": ["E1001", "E1002"],
    "description": "职工编号"
  }
}
```

**关键设计决策**：

- **保留数据类型和统计信息**：纯列名匹配对 `employee_id` vs `staff_no` 这类命名差异无能为力，需要类型和采样值辅助判断
- **限定输出格式**：`output_format: Choose from [Yes, No]`，减少解码空间，提高稳定性
- **JSON 结构而非自然语言**：保留了属性边界信息，避免不同列的内容混淆

对于三种匹配场景（SMD/SSD/SLD），prompt 中的信息丰富度递增：
- **SMD（仅元数据）**：列名 + 类型
- **SSD（有采样数据）**：+ 采样值
- **SLD（大量实例）**：+ 统计信息（均值、方差、分布）

---

### Q6：99.7% 负样本导致 LoRA 崩塌，具体现象是什么？你怎么诊断的？

**回答要点**：

**具体现象**：

训练初期 loss 会下降，但验证指标不动，预测分布塌缩成单一类别。这里要把两个实验现象分开讲：

- **base/full_sft 在平衡测试集上常见现象**：几乎全预测 Yes，因此 Precision=0.50、Recall=1.0、F1=0.667。这个 F1 看起来不低，但它是平衡测试集上的“全 Yes 假高分”，不是模型真的学会了 schema matching。
- **不平衡训练或黑盒蒸馏时常见现象**：模型更容易全预测 No，因为训练数据里 99.7% 都是 No。此时在平衡集上 Precision/Recall/F1 会接近 0；在全量极不平衡集上 Accuracy 可能很高，但同样没有正类识别能力。

**为什么发生**：

交叉熵对每个样本等权处理。面对 1:300 的正负比例，模型只要学会输出多数类，就能获得很低的 loss；面对平衡测试集时，全 Yes 或全 No 都会暴露出同一个问题：**模型没有学到“两个字段是否语义等价”的判别边界，只是在复现某个标签分布或解码偏置**。

**诊断过程**：

1. 对比训练 loss 和 val F1：train loss 下降，val F1 不动 → 过拟合标签分布
2. 查看预测分布：`Counter(predictions)` 发现几乎全是单一类别
3. 确认 root cause：训练集正负样本比约 1:300，模型选择"懒惰策略"

---

### Q7：你如何解决类别不平衡问题？为什么选择重采样而不是 focal loss？

**回答要点**：

**最终方案**：严格 1:1 平衡采样 + SM+EM 多任务联合训练。

**为什么不用 Focal Loss**：

Focal Loss 的核心思想是对 hard sample 加大权重：

$$\mathcal{L}_{focal} = -(1-p_t)^\gamma \log(p_t)$$

问题在于：在 LoRA 微调场景下，模型是在一个已有强先验的预训练权重上继续训练的，**loss 权重调整对 LoRA 低秩适配器影响有限**——LoRA 的更新量本来就很小（秩 64，参数量约占全量的 1%），在极端不平衡下 focal loss 的梯度信号仍然主要来自多数类，效果不如直接平衡数据。

**重采样的优势**：

- 直接从数据层面解决问题，不依赖 loss 函数的超参数调整
- 保证每个 batch 的梯度信号均衡
- 实现简单、效果确定

**多任务联合训练（SM + EM）的作用**：

Entity Matching（产品匹配）和 Schema Matching 共享"判断两个描述是否语义等价"的底层能力。EM 数据集正负样本相对平衡，通过多任务训练，模型在 SM 任务上获得了来自 EM 的**泛化先验**，有效缓解了正例特征学习不充分的问题（稀少正例特征解耦）。

**结果**：在平衡候选集上，Precision 从 0.50（等同于全 Yes 基线） → 0.905，说明模型真正学到了语义等价性的判断逻辑。全量候选集上还需要结合候选召回、阈值校准和大模型兜底，不能只报一个 0.905。

---

### Q8：LoRA 的原理是什么？rank 怎么选？

**回答要点**：

**LoRA 原理**：

预训练模型权重 $W_0 \in \mathbb{R}^{d \times k}$ 冻结，用一个低秩分解矩阵表示增量：

$$W = W_0 + \Delta W = W_0 + BA, \quad B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}, r \ll \min(d,k)$$

训练时只更新 $A, B$，前向传播变为：

$$h = W_0 x + \frac{\alpha}{r} BAx$$

参数量从 $d \times k$ 降为 $r(d+k)$，以 $r=64, d=k=768$ 为例：参数量降低约 6×。

**设计动机**：假设预训练模型的"知识调整"本质上是低秩的——任务适配不需要改变所有方向，只需要改变少数几个方向即可。这个假设在实践中被广泛验证。

**Rank 的选择**：

本项目选 `rank=64`（较大），原因：
- SM 任务是从通用语言理解到专业结构化匹配的跨度较大，需要更多自由度
- 参数量仍然远小于全量微调
- 通过 ablation 实验（rank=8/32/64）确认 64 在精度和效率上的平衡点

---

## 四、模块三：后训练与 Agentic RL 对齐

### Q9：DPO 是什么？它和 PPO 的根本区别在哪？

**回答要点**：

**DPO 的推导逻辑**：

RLHF 的目标是在 KL 约束下最大化期望奖励：

$$\max_\pi \mathbb{E}_{x,y\sim\pi}[r(x,y)] - \beta \mathbb{KL}[\pi || \pi_{ref}]$$

这个问题有闭合解：

$$\pi^*(y|x) = \frac{\pi_{ref}(y|x)}{Z(x)} \exp\left(\frac{1}{\beta}r(x,y)\right)$$

反解出奖励：$r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$

代入 Bradley-Terry 偏好模型，$Z(x)$ 约掉，得到 DPO 目标：

$$\mathcal{L}_{DPO} = -\mathbb{E}_{(x,y_w,y_l)}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$

**核心区别**：

| 维度 | PPO | DPO |
|------|-----|-----|
| 需要 Reward Model | 是（独立训练） | 否（隐式） |
| 在线/离线 | 在线（需要采样） | 离线（直接用偏好对） |
| 训练复杂度 | 高（4 个模型） | 低（2 个模型） |
| 稳定性 | 不稳定（需要精细调参） | 稳定 |
| 适合场景 | 大模型、复杂奖励 | 小模型、偏好数据充足 |

**DPO Loss 初始值为 ln2 的原因**：

训练初期 $\pi_\theta = \pi_{ref}$（模型没有任何偏好），两个 log-ratio 均为 0，$\sigma(0) = 0.5$，所以 $-\log(0.5) = \ln 2 \approx 0.693$。这是一个可以自验证的**初始健康检查点**，不是理论下界；Loss 从 0.693 降到 0.421，说明 policy 相对 reference 已经更偏向 chosen 回答，偏好学习生效。

---

### Q10：GRPO 和 PPO 的核心区别？为什么 GRPO 更适合小模型？

**回答要点**：

**PPO 的问题**：需要同时维护 Policy、Reference、Value（Critic）、Reward 四个模型，内存压力巨大，Value Function 的估计在小模型上容易不准确。

**GRPO 的改进**（Group Relative Policy Optimization）：

GRPO 用**组内相对奖励**替代 Value Function 的优势估计：

$$A_t = \frac{R_i - \mu_{group}}{\sigma_{group}}$$

对同一个 prompt 采样 $N$ 个回答，用组内奖励的均值和方差归一化，不需要额外的 Critic 网络。

$$\mathcal{L}_{GRPO} = -\mathbb{E}\left[\min\left(r_t A_t, \text{clip}(r_t, 1-\varepsilon, 1+\varepsilon)A_t\right) - \beta\text{KL}_t\right]$$

其中 $r_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}$ 是重要性权重。

**为什么更适合小模型**：

1. 只需要 1 个训练模型（+ 1 个参考模型），节省 50% 显存
2. 组内基线比 Value Function 更稳定——小模型的 Critic 估计噪声大，而组内统计量只依赖奖励本身
3. GRPO 的 Degenerate Groups 问题（组内奖励相同导致梯度为 0）在 64M 模型上确实存在，但通过限制任务难度范围可以缓解

---

### Q11：CISPO 相比 GRPO 改进了什么？

**回答要点**：

**PPO/GRPO 的 gradient 截断问题**：

标准 PPO/GRPO 的 clip 操作：

$$\min(r_t A_t, \text{clip}(r_t, 1-\varepsilon, 1+\varepsilon) A_t)$$

当 $r_t$ 超出 $[1-\varepsilon, 1+\varepsilon]$ 区间被截断时，梯度变为：

$$\nabla_\theta = \frac{\partial \text{clip}(r_t,...)}{\partial \theta} \cdot A_t = 0$$

**梯度路径被完全截断**，即使样本很重要（大 $|A_t|$），只要 ratio 被 clip，就没有梯度。

**CISPO 的修正**：

$$\mathcal{L}_{CISPO} = -\mathbb{E}\left[\min(r_t, \varepsilon_{max}) \cdot A_t \cdot \log\pi_\theta(a_t|s) - \beta \cdot \text{KL}_t\right]$$

关键变化：把 $r_t$ 只作为**截断权重**，梯度通过 $\log\pi_\theta$ 传播，而不是通过 $r_t$ 传播。

即使 $r_t$ 被截断为常数 $\varepsilon_{max}$，$\log\pi_\theta$ 仍然对参数可微，梯度仍然存在。

**实现上**：CISPO 是 GRPO 的 loss 变体，在 `train_grpo.py` 中设置 `loss_type=cispo` 即可，其余组内采样、奖励计算、优势构造逻辑完全复用 GRPO。

---

### Q12：你实验论证了"64M 极小模型的 RL 容量瓶颈，SFT 收益显著优于在线 RL"，怎么量化的？

**回答要点**：

**实验设计**：在同一个 Schema Matching 任务上，对比三条技术路径的最终 F1：

1. **SFT + LoRA（平衡采样 + 多任务）**：Precision 0.905，这是最终方案
2. **GRPO 在线 RL**：以 SM 判断准确性为奖励信号，在线采样更新策略
3. **SFT + GRPO**：先 SFT 再 RL 对齐

**观察到的现象**：

- GRPO 单独在 SM 任务上的收益远低于 SFT 方案，reward 提升但 F1 几乎不变
- 分析原因：64M 模型的**探索空间太小**，GRPO 需要同一 prompt 采样 $N$ 个不同回答并计算优势，但小模型的输出多样性很差（Degenerate Groups 问题），大量组的 $\sigma_{group} \approx 0$，梯度近乎为零

**结论的 Implication**：

对于极小模型（<100M），**数据质量和任务难度匹配比对齐算法的选择更重要**。RL 在大模型（>1B）上效果更显著，因为大模型有足够的参数容量去探索新的解空间。这一结论为后续多智能体系统的模型选型提供了实证依据——在资源受限场景下，应优先用更好的 SFT 数据而非更复杂的 RL 算法。

---

### Q13：Tool-Use 评测 6/8 通过，哪 2 项没过？为什么？

**回答要点**：

**6/8 通过的能力**：

| 测试用例 | 调用类型 | 结果 |
|----------|----------|------|
| 查询天气 | 单工具 | ✅ 正确 |
| 单位换算 | 单工具 | ✅ 正确 |
| 美元兑人民币汇率 | 单工具 | ✅ 正确 |
| 翻译"你好世界" | 单工具 | ✅ 正确 |
| Tokyo 天气 + 温度换算（并行） | 并行调用 | ✅ 成功 |
| 某链式推理任务 | 链式调用 | ✅ 成功 |

**2 项未通过**：

- **重复调用同一工具**：链式调用场景下，模型对已获得结果的工具发起了第二次调用。根因是 64M 模型的上下文追踪能力有限，无法稳定维护"哪些工具已经调用并返回结果"的状态。
- **Mock 函数 bug**（工程问题）：测试脚本中某工具的 mock 实现有 bug，返回了异常格式，模型未能正确处理 error 响应。

**对这两个问题的思考**：

第一个是小模型的根本限制，需要更多 Agentic RL 训练数据（特别是包含重复调用负例的惩罚样本）。第二个是工程问题，在实际部署中已修复。整体而言，6/8 验证了 Agentic RL 训练赋予小模型工具调用能力的可行性，单工具和并行调用已经满足多智能体系统的基本需求。

---

### Q14：Agentic RL 的奖励函数是怎么设计的？

**回答要点**：

Agentic RL 的核心挑战是**多轮延迟奖励**——中间步骤没有单独的标签，只有轨迹末尾才知道是否成功。

**奖励函数设计**：

$$R(\tau) = R_{answer} + R_{tool} + R_{format} + R_{rm} - R_{unfinished}$$

- $R_{answer}$：最终答案与 ground truth 的匹配度（最主要信号）
- $R_{tool}$：工具调用合法性（JSON 格式正确、参数类型匹配）
- $R_{format}$：输出格式规范性（`<tool_call>`/`</tool_call>` 标签闭合）
- $R_{rm}$：Reward Model 对整体回答质量的打分
- $R_{unfinished}$：轨迹未完成的惩罚项（防止模型"摆烂"不调用工具）

这种设计将稀疏的终末奖励拆解为多个密集的中间奖励信号，缓解了信用分配（credit assignment）问题，有助于 64M 小模型快速收敛。

---

## 五、模块四：黑盒知识蒸馏

### Q15：黑盒蒸馏和白盒蒸馏的区别？你为什么选择黑盒？

**回答要点**：

| 对比维度 | 黑盒蒸馏 | 白盒蒸馏 |
|----------|---------|---------|
| 监督信号 | Teacher 的输出文本（hard label / CoT） | Teacher 的 logit 分布（soft label） |
| 信息丰富度 | 低（只有最终答案或推理链） | 高（包含所有候选 token 的概率） |
| 实现难度 | 简单（就是 SFT） | 复杂（需要 teacher/student 词表对齐） |
| 是否需要 Teacher 源码 | 否（只需要 API） | 是（需要访问 logit） |

**本项目选择黑盒的原因**：

Jellyfish-7B 的 base model 是 Llama，词表大小 32,000；而 MiniMind 的 tokenizer 词表只有 6,400，**词表完全不兼容**，无法直接对齐 logit 分布做白盒蒸馏。因此使用 Mixtral-8x7B 生成的含推理链数据（84K 条 `sm_gen_m8x7b.jsonl`）作为 teacher 输出，直接 SFT 即黑盒蒸馏。

**蒸馏过程**：

```
Mixtral-8x7B（Teacher）
    ↓ 生成 SM 任务的 CoT 推理链
84K 条 "是/否，因为…" 格式的监督数据
    ↓ SFT 训练
MiniMind-64M（Student）
```

---

### Q16：蒸馏实验结果中 F1=0 是什么情况？你怎么解释？

**回答要点**：

**现象**：蒸馏后的模型在 SM 评测上出现 **F1=0、几乎全部预测 No**。这里要特别注意评估口径：

- 如果在平衡测试集上看，全预测 No 的 Accuracy 应约为 0.5，而不是 1.0。
- 如果在极端不平衡的全量候选集或全负样本切片上看，Accuracy 可能非常高，甚至接近 1.0，但这是多数类带来的假象。

所以面试里不要说“平衡测试集上 F1=0 且 Accuracy=1.0”。更严谨的说法是：**蒸馏模型没有正类识别能力，F1=0；Accuracy 在不平衡分布下会虚高，因此不能作为主指标**。

**根因分析**：

黑盒蒸馏使用了全量 84K SM 数据，其中 **99.7% 的标签为 No**。Mixtral-8x7B 生成的推理链大多数结论也是 "No"（因为绝大多数列对确实不匹配）。

MiniMind-64M 在这个数据上 SFT，本质上是**把数据集的标签分布蒸馏进了模型**——模型学会了"生成 No 推理"，而不是"判断语义等价性"。这与 LoRA 崩塌的表现形式不同，但根因相同：**极端类别不平衡是 SM 任务的根本难点，与模型大小或推理能力无关**。

**这个结果的价值**：

这个负面结果恰好形成了对 LoRA 方案的反向验证。LoRA 之所以成功（平衡候选集 Precision 0.905），核心不是"引入了 LoRA 这个技术"，而是：
1. 构造了 1:1 平衡数据集
2. 引入了多任务迁移（SM+EM）

黑盒蒸馏加了推理链、用了更大的 teacher，但没有解决数据不平衡，所以完全失败。这证明**数据分布才是决定因素**，对后续数据收集策略有直接指导意义。

---

### Q17：推理速度 174 tokens/s 是在什么条件下测的？

**回答要点**：

**测试配置**：

- 硬件：AutoDL RTX 4090 单卡
- 模型：MiniMind-64M（hidden_size=768, layers=8）
- 测试方法：50 次重复测量，每次生成 max_new_tokens=100
- 结果：full_sft 178.9 ± 1.4 tokens/s，distill_sft 173.6 ± 5.1 tokens/s

**对比基准**：

Jellyfish-7B 在 A100 80G 上约 30-50 tokens/s（引用论文数据）。因此 MiniMind-64M 的吞吐约为 Jellyfish-7B 的 3.5-6×（取均值约 4.5×），而参数量仅为 1/109。

**重要说明**：速度提升主要来自**参数量差异**（64M vs 7B），而非量化或推理引擎优化。若进一步对 MiniMind 做 INT8/INT4 量化，或使用 vLLM 连续批处理，吞吐还可提升 2-3×。这一点在面试中主动说出来，体现工程深度。

---

## 六、综合压轴题

### Q18：整个项目最难 debug 的问题是什么？

**回答要点（结合实际踩坑）**：

有几个印象深刻的 bug：

**① Tokenizer 相对路径问题（最隐蔽）**

在迁移到新机器时，transformers 的新版本不再支持相对路径加载 tokenizer，`AutoTokenizer.from_pretrained('./tokenizer')` 会 silently 失败——不报错，但返回的 tokenizer 会用错误的词表，导致训练 token 分布完全错误，loss 不正常下降。定位过程耗时半天，最终通过对比 tokenizer 的 `vocab_size` 属性发现问题，改为绝对路径后解决。

**② GRPO 的 SGLang 502 错误**

多卡 GRPO 训练时，SGLang 作为 rollout engine 偶发 502 Bad Gateway。根因是 SGLang 默认使用了与训练代码不兼容的 rollout 引擎，需要显式指定 `--rollout_engine torch`。这个问题的特殊之处在于它是环境依赖问题，不是代码本身的 bug，需要看 SGLang 的 issue tracker 才能找到解决方案。

**③ 工作目录问题（最低级但最消耗时间）**

MiniMind 的所有训练脚本必须从 `/root/minimind` 目录运行，不能从 `trainer/` 子目录运行，否则相对路径全部失效。但这个限制没有明显的报错，只是数据加载找不到文件而 silent 退出。

---

### Q19：如果把 FFN 替换成 MoE，你预期会遇到哪些问题？

**回答要点**：

**MoE 的核心思想**：

用 $N$ 个专家（FFN）+ 路由器替代单个 FFN，每个 token 只激活 top-K 个专家（稀疏激活）。在参数量扩大 $N$ 倍的同时，计算量仅增加 $K/N$ 倍。

**预期挑战**：

**① Expert Collapse（专家坍塌）**：路由器趋向于把所有 token 都发给同一个专家（因为某个专家初始略占优），其他专家得不到训练，最终退化为 Dense 模型。解决方案：在 loss 中加入 Auxiliary Load Balancing Loss：

$$\mathcal{L}_{aux} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i$$

其中 $f_i$ 是专家 $i$ 被分配的 token 比例，$P_i$ 是路由器对该专家的平均概率。

**② 通信开销（多卡场景）**：在 DDP 训练中，MoE 的 all-to-all 通信（将 token 发到对应专家所在的 GPU）是主要瓶颈，尤其在 AutoDL 这种 PCIe 互联环境下会更明显。

**③ 容量因子（capacity factor）**：每个专家在一个 batch 内能处理的 token 数量有上限（capacity = batch_size × top_k / num_experts × capacity_factor）。超出容量的 token 会被 drop，需要合理设置 capacity_factor 平衡精度和显存。

**④ 对 64M 模型的适用性**：MoE 的优势在参数量大时才显著（DeepSeek-MoE 是 ~16B 参数、~2.5B 激活）。对 64M 模型，MoE 的工程复杂度增量远大于精度收益，这是一个 interesting 的实验方向，但在当前场景下性价比存疑。

---

### Q20：这个 64M 模型和你论文里用的大模型在 Schema Matching 上差距主要在哪里？

**回答要点**：

**论文（IJCNN 2025）里的方案**：使用 Llama3.1-8B 作为 MatcherAgent 的骨干，在 SMD/SSD/SLD 三种场景下达到 SOTA。

**64M 模型的能力边界**：

通过实验观察，主要差距在三个维度：

| 维度 | Llama3.1-8B | MiniMind-64M |
|------|-------------|--------------|
| 跨语言匹配 | 强（英中/英法列名对） | 弱（词表仅 6400，非中英双语） |
| 长文本 schema 理解 | 好（支持 8K context） | 有限（768dim 模型有效范围约 512 token） |
| 零样本泛化 | 好（预训练知识丰富） | 差（需要特定领域的 SFT 覆盖） |
| 复杂类型推断 | 强（能推断 `VARCHAR` 里隐含的数值含义） | 弱（主要依赖列名相似度） |

**实践意义**：

这个差距说明"小模型专用化 + 大模型通用化"的两轨策略更合理：在多智能体系统中，可以用 64M 模型处理高频、简单的 SMD 场景（纯列名匹配），用大模型处理低频、复杂的 SLD 场景（需要实例推断），实现计算成本和精度的自适应平衡。这也是论文系统中 dual-model routing 设计的动机。

---

### Q21：RL 算法统一框架角度，DPO/PPO/GRPO/CISPO 怎么理解？

**回答要点**：

所有 PO 算法可以分解为三个核心组件：

$$\mathcal{L} = -\mathbb{E}[f(r_t) \cdot g(A_t) - h(\text{KL}_t)]$$

| 算法 | 策略项 $f(r_t)$ | 优势项 $g(A_t)$ | 正则项 | 训练模型数 |
|------|-----------------|----------------|--------|-----------|
| DPO | $\log r_w - \log r_l$ | 无显式优势项 | 隐含在 $\beta$ 中 | 1（2 个参与前向） |
| PPO | $\min(r, \text{clip}(r))$ | $R - V(s)$（Critic） | $\beta \cdot \mathbb{E}[\text{KL}]$ | 2 |
| GRPO | $\min(r, \text{clip}(r))$ | $\frac{R-\mu}{\sigma}$（组内） | $\beta \cdot \text{KL}_t$ | 1 |
| CISPO | $\text{clip}(r, 0, \varepsilon_{max}) \cdot \log\pi_\theta$ | $\frac{R-\mu}{\sigma}$（组内） | $\beta \cdot \text{KL}_t$ | 1 |

**演化逻辑**：

- PPO → GRPO：去掉 Critic，用组内统计量估计优势，更轻量
- GRPO → CISPO：修复 clip 后梯度截断，将 ratio 从梯度路径移出，梯度更稳定
- DPO：将 Reward Model 的学习隐式嵌入策略网络，一步完成 RM 训练和策略优化

---

## 七、延伸基础题（高频通用考点）

> 这部分不是 Tiny-Tabular-LLM 代码里直接出现的题，但都是从项目自然引出的追问点——面试官问完项目细节，大概率会往这几个方向探底。每题都尽量绑定回项目里的实际组件或数字，而不是背书式回答。

### 7.1 Attention 与 Transformer 基础

#### Q22：Self-Attention 里为什么要除以 $\sqrt{d_k}$？

**回答要点**：

Attention 计算为 $\text{softmax}(QK^T/\sqrt{d_k})V$。若不做缩放，当 $d_k$ 较大时，$QK^T$ 的方差会随 $d_k$ 线性增长（假设 Q、K 各分量独立同分布，均值 0 方差 1，则点积的方差为 $d_k$）。方差过大会让 softmax 的输入落入饱和区，梯度趋近于 0，训练不稳定。除以 $\sqrt{d_k}$ 把方差重新归一化到 1，这是一个纯粹的数值稳定性设计，不是语义上的改动。

在项目里，GQA 的每个 KV head 被多个 Q head 共享，$d_k$（每个 head 的维度）在缩放公式中依然按单 head 维度计算，与是否分组无关——这个点容易被追问混淆，需要说清楚 GQA 只改变 KV 的组织方式，不改变单头内的 attention 计算逻辑。

#### Q23：Self-Attention 的时间和空间复杂度是多少？为什么长序列会成为瓶颈？

**回答要点**：

- 时间复杂度：$O(n^2 d)$（$n$ 为序列长度，$d$ 为隐藏维度），主要来自 $QK^T$ 的计算
- 空间复杂度：$O(n^2)$，需要存储完整的 attention 矩阵

这也是为什么长文本外推是个独立课题——朴素 Attention 在 $n$ 增大时显存和计算量都是平方级增长。项目里用 KV-Cache 只解决了"重复计算"问题（把 $O(n^2)$ 的增量计算降为 $O(n)$），并没有解决 attention 矩阵本身的平方复杂度；真正缓解这个问题需要 FlashAttention（用分块计算避免显式存储 $n\times n$ 矩阵）或者稀疏 Attention。

#### Q24：Multi-Head Attention 为什么比单头更好？

**回答要点**：

单头 Attention 只能学到一种"关注模式"，多头相当于在多个子空间里并行计算不同的相似度度量，每个头可以关注不同类型的依赖关系（比如句法依赖 vs 语义依赖）。数学上，多头是把 $d_{model}$ 切分成 $h$ 个 $d_{model}/h$ 维子空间分别计算 attention，再拼接：

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1,...,\text{head}_h)W^O$$

值得注意的是，**多头不是免费的**——GQA 的设计动机正是发现"KV 不需要和 Q 一样多头"，多个 Q head 完全可以共享同一组 K/V，因为不同 Q head 关注的是不同的查询模式，但底层被检索的信息（K/V 编码的内容）冗余度很高。

#### Q25：LayerNorm 和 BatchNorm 的区别？为什么 NLP 模型普遍用 LayerNorm？

**回答要点**：

- **BatchNorm**：在 batch 维度上归一化，对每个特征通道计算整个 batch 的均值方差，依赖 batch size，且训练/推理行为不一致（推理用滑动平均统计量）
- **LayerNorm**：在特征维度上归一化，对单个样本的所有特征计算均值方差，不依赖 batch size，训练推理行为一致

NLP 任务里序列长度变化、常用小 batch（尤其是长文本训练时受显存限制），BatchNorm 的统计量会不稳定；而 LayerNorm 对每个 token 独立归一化，天然适配变长序列。项目里用的是 **RMSNorm**（LayerNorm 的简化版，去掉了均值中心化，只做方差归一化）：

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum x_i^2 + \epsilon}} \cdot \gamma$$

省去均值计算，速度更快，在 Llama 系列中被证明精度损失可忽略——这也是为什么现代小模型基本都用 RMSNorm 而非原始 LayerNorm。

---

### 7.2 训练工程基础

#### Q26：AdamW 和 Adam 的区别？为什么大模型训练几乎都用 AdamW？

**回答要点**：

Adam 的权重衰减是加在梯度里一起做动量更新的：

$$g_t = \nabla L(\theta) + \lambda \theta_{t-1}$$

这导致权重衰减和自适应学习率耦合——衰减力度会被 Adam 的二阶矩估计放大或缩小，不同参数实际衰减量不一致。

AdamW 把权重衰减从梯度更新中解耦出来，直接作用在参数上：

$$\theta_t = \theta_{t-1} - \eta\left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon} + \lambda\theta_{t-1}\right)$$

解耦后正则化效果更符合预期，是目前几乎所有 Transformer 训练的标配。项目里所有训练脚本（pretrain/SFT/DPO/PPO/GRPO/Agent）统一用 `optim.AdamW(model.parameters(), lr=args.learning_rate)`。

#### Q27：为什么要用 Cosine Annealing 学习率调度？Warmup 的作用是什么？

**回答要点**：

**Warmup**（本项目训练脚本中通过调度器隐式处理）：训练初期模型参数是随机初始化的，梯度方向不可靠，如果直接用较大学习率，容易在早期就把参数带偏，导致后续难以收敛。Warmup 让学习率从很小的值线性增长到目标值，给模型一个"热身"过程。

**Cosine Annealing**：

$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max}-\eta_{min})\left(1+\cos\left(\frac{t}{T_{max}}\pi\right)\right)$$

项目里 `CosineAnnealingLR(optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate/10)`——学习率从初始值余弦衰减到 1/10。相比阶梯衰减，余弦调度是平滑的，避免了学习率突变带来的 loss 震荡；相比线性衰减，余弦在训练中期下降较慢、后期下降较快，能让模型在充分学习的基础上再做精细收敛。

#### Q28：混合精度训练是怎么回事？FP16 和 BF16 的区别？

**回答要点**：

混合精度训练用 FP16/BF16 做大部分计算（矩阵乘法等），关键部分（如梯度累加、参数更新）保留 FP32，以节省显存、提升算力利用率（Tensor Core 对半精度有专门加速）。

**FP16 vs BF16**：

| | FP16 | BF16 |
|---|------|------|
| 位分配 | 1 符号 + 5 指数 + 10 尾数 | 1 符号 + 8 指数 + 7 尾数 |
| 数值范围 | 较小，容易溢出（需要 loss scaling） | 与 FP32 相同范围，不易溢出 |
| 精度 | 尾数位多，精度略高 | 尾数位少，精度略低 |

项目训练代码里可以看到两种都支持：`dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16`，并且只有用 FP16 时才需要 `GradScaler`（`scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype=='float16'))`）——因为 FP16 数值范围小，梯度容易下溢为 0，需要先放大 loss 再反向传播，更新前再缩小回来；BF16 因为指数位和 FP32 一致，不存在这个问题，不需要 GradScaler。3090/4090 这类消费级显卡通常用 FP16（BF16 支持在 Ampere 架构才完整），实际选择要看硬件。

#### Q29：为什么要做梯度裁剪？梯度累积又是解决什么问题？

**回答要点**：

**梯度裁剪**：防止梯度爆炸——当某个 batch 的 loss 异常大（比如遇到脏数据）时，反向传播的梯度范数可能骤增，一步更新就把参数带到很差的区域。常用做法是按范数裁剪：

$$g \leftarrow g \cdot \min\left(1, \frac{\text{clip\_norm}}{\|g\|}\right)$$

**梯度累积**：解决显存不够但需要大 batch size 的问题。真实 batch 拆分成多个 micro-batch 依次前向反向，梯度累加但不立即更新参数，累积到目标 batch size 才做一次 optimizer step。项目里的 `accumulation_steps` 参数就是这个作用，配合 `total_optimizer_steps = math.ceil(iters/accumulation_steps)*epochs` 计算真实的调度器步数——这是个容易在面试中被问到的实现细节：**调度器的 `T_max` 应该用 optimizer step 数而不是 dataloader 的 batch 数**，用错了会导致学习率衰减节奏和实际训练进度不匹配。

---

### 7.3 分词与数据基础

#### Q30：BPE（Byte Pair Encoding）分词算法的原理是什么？

**回答要点**：

BPE 是一种自底向上的子词切分算法：

1. 初始词表 = 所有单字符
2. 统计训练语料中相邻符号对的共现频率
3. 把频率最高的符号对合并成一个新符号，加入词表
4. 重复步骤 2-3，直到词表达到目标大小

结果是**高频词整体保留为一个 token，低频词/生僻词被切分成子词组合**，兼顾了词表大小控制和未登录词（OOV）处理能力——理论上不存在真正的 OOV，任何字符串都能分解到字符级。

#### Q31：项目里 tokenizer 词表只有 6,400，会不会有 OOV 或效率问题？

**回答要点**：

**不会有严格意义的 OOV**：BPE 分词的底层保底粒度是字符，任何未见过的词都能退化到字符级编码，只是编码效率（每个字符占一个 token）会变差。

**权衡是编码效率而非能力缺失**：

- 大词表（如 Qwen2 的 151,643）：常见词可以整体编码为 1 个 token，序列更短，但 embedding 层参数量大幅增加
- 小词表（MiniMind 的 6,400）：同样的文本会被切成更多 token，序列更长，推理时计算量增加，但 embedding + 输出层的参数量显著降低

对于 64M 参数量级的模型，embedding 层如果用大词表会占掉过大比例的总参数（词表 151K × 隐藏维度 768 ≈ 1.16 亿参数，直接超过整个模型的目标规模）。项目选择小词表本质上是**参数预算的取舍**：用推理时序列变长（吞吐略降）换取模型主体计算能力占比更高。这也是文档里提到"评估跨语言/长文本理解能力时 MiniMind-64M 弱于大模型"的部分原因——上下文有效容量被更长的序列占用。

---

### 7.4 推理与部署基础

#### Q32：YaRN 长文本外推具体是怎么做的？

**回答要点**：

RoPE 的旋转角度 $\theta_i = 10000^{-2i/d}$ 是在训练序列长度内标定的。如果推理时序列长度超过训练长度，高频维度的旋转周期会被"用完"，导致位置编码失去分辨能力（外推失效）。

YaRN（Yet another RoPE extensioN）的核心思路是**对不同频率的维度做不同程度的插值**：

- 高频维度（短距离依赖）：保持外推能力较强，改动较小
- 低频维度（长距离依赖）：做插值压缩，把原本训练时"看过"的旋转范围重新映射到更长的实际长度

同时会引入温度系数调整 attention 的 softmax 分布，补偿插值带来的注意力熵变化。项目里说的"支持长度外推到 2048 及以上，无需额外训练"，指的就是推理时动态启用 YaRN，不需要用长序列重新训练模型权重。

#### Q33：vLLM 的核心加速原理是什么？（PagedAttention）

**回答要点**：

传统推理框架为每个请求的 KV Cache 预分配连续显存（按最大序列长度），导致大量显存碎片和浪费——大多数请求实际长度远小于最大长度。

**PagedAttention** 借鉴操作系统虚拟内存分页的思想，把 KV Cache 切分成固定大小的 block，通过一张 block table 做逻辑地址到物理显存地址的映射：

- 显存按需分配，不再需要为最大长度预留
- 多个请求可以共享相同的 block（比如相同的 system prompt 前缀，即 prefix caching）
- 支持 **Continuous Batching**：新请求可以随时插入正在进行的 batch，不需要等一整个 batch 全部生成完毕

这是为什么项目里用 vLLM 部署 Qwen3-9B-Instruct 做推理服务的原因——相比朴素的 HuggingFace `generate()`，vLLM 在多并发场景下吞吐提升通常是数倍到十倍以上。

#### Q34：模型量化（INT8/INT4）的原理？会带来什么精度损失？

**回答要点**：

量化把 FP16/FP32 的权重映射到低比特整数表示，核心公式（以对称量化为例）：

$$q = \text{round}\left(\frac{x}{s}\right), \quad s = \frac{\max(|x|)}{2^{b-1}-1}$$

$s$ 是缩放因子，$b$ 是目标比特数。推理时用 $x \approx q \cdot s$ 近似还原。

**INT8**：精度损失通常很小（<1% 指标下降），显存减半，是目前最常用的部署量化方案。

**INT4**：显存降至 1/4，但激活值的量化误差会明显放大，通常需要更精细的方法（如 AWQ——Activation-aware Weight Quantization，根据激活值分布调整哪些权重通道需要保留更高精度）。我在推理侧用的 Qwen3-9B-Instruct AWQ 4-bit 就是这种方案——通过激活感知的方式挑出对输出影响大的权重通道做保护，其余通道正常量化，在 4-bit 下把精度损失控制在可接受范围。

**如果被追问"MiniMind-64M 要不要量化"**：64M 本身已经很小，量化的显存收益边际不大（几百 MB 的差异），但如果追求单位时间内更高的并发吞吐，INT8 量化仍然值得做——这是文档里"进一步量化可提升吞吐"这句话背后的具体依据。

---

### 7.5 评估指标基础（项目里的活案例）

#### Q35：Precision、Recall、F1 的定义和适用场景？

**回答要点**：

$$\text{Precision} = \frac{TP}{TP+FP}, \quad \text{Recall} = \frac{TP}{TP+FN}, \quad F1 = \frac{2 \cdot P \cdot R}{P+R}$$

- **Precision 优先**场景：误报代价高（比如 Schema Matching 中，把不匹配的列误判为匹配，会导致数据集成时错误合并两个不同语义的字段，下游影响大）
- **Recall 优先**场景：漏报代价高（比如异常检测，漏掉一个真正的异常比多报几个误报更严重）
- **F1** 是两者的调和平均，用于没有明显偏向、需要整体平衡的场景

我们项目里 LoRA 方案最终 Precision 0.905 但 Recall 只有 0.396（F1 仅 0.551），这个取舍本身是可以被追问的——**这不是最优 F1 的方案，而是刻意偏向高 Precision 的方案**，因为 Schema Matching 场景下误合并的代价远高于漏检（漏检最多是少发现一个匹配，人工可以补充；误合并会污染下游数据）。

#### Q36：为什么类别不平衡场景下 Accuracy 会失真？你的项目里怎么体现的？

**回答要点**：

Accuracy = $(TP+TN)/\text{总数}$。当负样本占 99.7% 时，一个**什么都不学、全部预测负类**的模型 Accuracy 能高达 99.7%，看起来"很准"，但对正类的识别能力是 0——这正是项目里黑盒蒸馏实验暴露的问题：**distill_sft 几乎全预测 No，F1=0；如果在极度不平衡或全负切片上看 Accuracy 会虚高，但在平衡测试集上 Accuracy 不能说成 1.0**。

这组数字组合本身就是一个很好的面试素材：单独报 Accuracy 具有极强的误导性，评估不平衡分类任务必须同时看 Precision/Recall/F1，最好再看 PR-AUC（对不平衡数据比 ROC-AUC 更敏感，因为 ROC-AUC 的 FPR 计算会被大量真负例稀释，容易显得"好看"）。

---

## 八、快速背诵卡片

### 核心数字

```
参数量:      64M（Jellyfish-7B 的 1/109）
SM Precision: 0.905（平衡候选集，基线 0.50）
推理吞吐:    174 tokens/s（Jellyfish-7B 的 ~4.5×）
CV 改善:     0.368 → 0.153（↓58%）
DPO Loss:    0.693（ln2 初始点） → 0.421
Tool-Use:    6/8 通过
词表大小:    6,400
训练数据:    SM+EM 1:1 平衡 + 84K Mixtral CoT
```

### 关键因果链（面试时主动说出来）

1. **为什么做这个项目** → 多智能体系统中 7B 模型 API 成本和延迟瓶颈
2. **为什么不用白盒蒸馏** → 词表不兼容（6400 vs 32000）
3. **为什么 LoRA 成功** → 根本不是 LoRA 本身，是平衡采样 + 多任务迁移
4. **为什么 GRPO 收益有限** → 64M 模型探索空间不足，Degenerate Groups 问题
5. **为什么黑盒蒸馏 F1=0** → 把数据不平衡蒸馏进了模型，反向验证了 LoRA 方案

### 一句话总结项目贡献

> 面向结构化数据匹配场景，系统性地验证了在 64M 极小模型上完整 LLM 训练流水线的可行性，核心发现是：**数据分布比算法选择更重要**（平衡采样 + 多任务迁移 > 黑盒蒸馏，SFT > 在线 RL），为资源受限的多智能体部署提供了实证依据。

### 延伸题速查表（Q22-Q36 一句话版）

| 问题 | 一句话答案 |
|------|-----------|
| 为什么除以 $\sqrt{d_k}$ | 防止点积方差过大，softmax 进入饱和区梯度消失 |
| Attention 复杂度 | 时间 $O(n^2d)$，空间 $O(n^2)$，长序列是平方级瓶颈 |
| LayerNorm vs BatchNorm | LN 按样本归一化不依赖 batch size，NLP 变长序列更适配 |
| AdamW vs Adam | 解耦权重衰减与自适应学习率，正则化更符合预期 |
| Warmup 作用 | 避免初期大学习率把随机初始化参数带偏 |
| FP16 vs BF16 | FP16 精度高但范围小需 GradScaler；BF16 范围大不易溢出 |
| 梯度累积 | 显存不够时模拟大 batch，注意调度器 T_max 用 step 数 |
| BPE 原理 | 高频符号对反复合并，无真 OOV，只是编码效率差异 |
| YaRN 原理 | 对不同频率维度做不同程度插值，实现免训练长度外推 |
| PagedAttention | 分页管理 KV Cache，减少碎片，支持前缀共享和连续批处理 |
| INT8/INT4 量化 | 权重映射低比特整数，AWQ 用激活感知保护关键通道 |
| 不平衡下 Accuracy 失真 | 全预测多数类也能刷高 Accuracy，必须看 F1/PR-AUC |

---

## 九、面试讲述模板

这一部分不是新的技术点，而是面试现场的表达稿。核心原则：**先讲业务问题，再讲技术选择，最后主动说明指标口径和线上边界**。不要一上来堆 RoPE、DPO、GRPO，否则面试官很难判断项目价值。

### 9.1 90 秒项目总述

> 我这个项目做的是 Tiny-Tabular-LLM，目标是在结构化数据匹配场景里训练一个 64M 级别的小模型，作为多智能体系统里的高吞吐 MatcherAgent。背景是我之前做图表检索和 schema matching 时，如果每个候选字段对都调用 7B 或更大的模型，API 成本、并发延迟和 rate limit 都会成为瓶颈，所以我想验证：能不能用一个极小但专用的模型处理高频匹配判断。
>
> 技术上，我基于 MiniMind 复现了一条完整 LLM 训练链路，包括 tokenizer、Transformer 预训练、SFT、LoRA、DPO/GRPO/CISPO、Tool-Use 和黑盒蒸馏。项目里比较核心的改动有两类：第一是底座侧，我实现了 AttnRes，把 attention 子层的固定残差分支改成对历史 hidden state 的 softmax 加权聚合，使各层 hidden norm 的 CV 从 0.368 降到 0.153；第二是任务侧，我把二维 schema 信息序列化成 JSON 指令，并针对 99.7% 负样本的不平衡问题做了 1:1 平衡采样和 SM+EM 多任务迁移，使平衡候选集上的 Precision 从 0.50 提升到 0.905。
>
> 最重要的结论不是“某个算法最强”，而是对于 64M 这种极小模型，数据分布和任务适配比复杂 RL 更关键。黑盒蒸馏和在线 RL 都暴露了容量与分布瓶颈，而 LoRA + 平衡数据 + 多任务迁移最有效。部署上，这个模型在 4090 上约 174 tokens/s，适合作为高频候选过滤或重排节点，再把低置信度样本交给大模型兜底。

**90 秒版收尾句**：

> 所以这个项目的价值是：我系统性验证了在资源受限场景下，小模型不是直接替代大模型，而是可以通过专用化训练变成一个便宜、快、可控的中间决策节点。

### 9.2 5 分钟技术深挖版

**0:00-0:40 背景与目标**

这个项目来自结构化数据匹配场景，核心任务是判断两个字段、实体或表头是否语义等价。原系统如果直接用 7B/8B 大模型做 MatcherAgent，效果不错，但在大规模候选对上成本和延迟不可接受。因此目标不是训练一个通用聊天模型，而是训练一个 64M 专用判别模型，处理高频、简单或中等难度的 schema matching 请求。

**0:40-1:40 底座构建与架构改动**

底座基于 MiniMind，主体是标准 decoder-only Transformer：RMSNorm、RoPE、SwiGLU、GQA 和 KV-Cache。这里我重点做了 AttnRes 复现。标准 PreNorm 的残差路径是固定叠加，深层时容易出现 hidden state 范数随层数累积、底层贡献被稀释的问题。我在 attention 子层残差分支上引入历史 hidden state 的 softmax 加权聚合，代码里用历史 hidden 的 sequence mean 和可学习 query 打分，再沿 depth 维度归一化。实验上，同等算力下 loss 降低约 0.9%，各层 hidden norm 的 CV 从 0.368 降到 0.153，说明层间尺度更均匀。

**1:40-3:00 结构化数据序列化与 LoRA 适配**

Schema Matching 的难点是输入本质上是二维结构，而 LLM 输入是一维 token 序列。我把字段名、类型、采样值、统计信息封装成 JSON 指令格式，让模型明确区分 source column 和 target column。第一次训练遇到的主要问题是类别极端不平衡：原始候选对里 99.7% 是 No，模型很容易学成全 Yes 或全 No 这种单一类别策略。我的诊断方式是看 train loss、val F1 和 `Counter(predictions)`，确认模型没有学到判别边界。最终方案是严格 1:1 平衡采样，并引入 Entity Matching 作为多任务迁移，因为 EM 和 SM 共享“判断两个描述是否语义等价”的底层能力。结果是在平衡候选集上 Precision 从 0.50 提升到 0.905。

**3:00-4:10 后训练、RL 与蒸馏的负结论**

后训练部分我跑了 DPO、GRPO/CISPO 和 Agentic RL。DPO loss 从 ln2 附近的 0.693 降到 0.421，说明偏好对齐链路是通的。GRPO/CISPO 的价值更多是验证小模型 RL 的容量边界：64M 模型采样多样性不足，容易出现 group 内 reward 方差很小的问题，所以在线 RL 收益不如高质量 SFT。黑盒蒸馏也给了一个重要负结论：Mixtral-8x7B 生成 84K CoT 后，student 仍然容易全预测 No，F1=0，原因是 teacher 输出也继承了 99.7% No 的数据分布。这反向证明了：不解决数据分布，蒸馏更多是在复制行为偏差。

**4:10-5:00 结果、边界与反思**

最终结果要分口径讲：平衡候选集 Precision=0.905，说明模型在经过筛选的候选对上有较高可信度；但全量候选空间极度不平衡，Precision 会明显下降，所以线上不能直接穷举全量候选并让小模型单独拍板。正确用法是把它放在候选生成之后，作为高吞吐过滤/重排节点，并把灰区样本交给大模型。这个项目给我的最大经验是：在小模型上，数据构造、评估口径和系统路由比堆复杂算法更重要。

### 9.3 “为什么这个方案线上可用”的系统设计版

**一句话定位**：

> Tiny-Tabular-LLM 不是线上唯一决策者，而是一个低成本、高吞吐、可校准的 schema matching 中间节点，用来减少大模型调用量，并把复杂样本路由给大模型或人工兜底。

**线上链路设计**：

```text
离线数据画像
  ↓
字段标准化 / 类型推断 / 采样值统计
  ↓
候选召回（名称相似度、类型约束、embedding top-k、规则过滤）
  ↓
Tiny-Tabular-LLM 打分或 Yes/No 判别
  ↓
三段式路由：高置信接受 / 低置信拒绝 / 灰区交给 7B 大模型
  ↓
结果缓存 + 人工抽检 + 线上反馈回流
```

**为什么可用**：

1. **成本可控**：64M 模型本地推理，不依赖外部 API；4090 上约 174 tokens/s，适合处理大量候选对。
2. **角色清晰**：它不负责所有 schema matching，只负责候选召回之后的高频判别和排序，把大模型调用集中到灰区样本。
3. **风险可控**：Schema Matching 里误合并代价高，所以线上阈值应偏 Precision；高置信才自动接受，低置信拒绝，模糊样本进入大模型或人工审核。
4. **可监控**：线上不只看 Accuracy，而看 Precision@K、PR-AUC、人工抽检误合并率、字段级覆盖率、P50/P95 延迟和大模型调用节省比例。
5. **可迭代**：把大模型兜底结果、人工修正结果回流到训练集，定期做平衡采样和难例挖掘，缓解领域漂移。

**面试官追问“全量 Precision 只有 0.019，为什么还能线上用？”时这样答**：

> 这个 0.019 是在全量极不平衡候选空间上直接评估的结果，不能代表线上最终链路。真实系统不会让模型从所有字段两两组合里直接做最终决策，而是先用规则、类型约束和 embedding 召回把候选空间缩小，再让小模型做重排或高 Precision 过滤。也就是说，小模型优化的是候选集内的判别质量和吞吐，而不是替代候选生成本身。对高风险样本，我会用阈值和大模型兜底控制误合并率。

**主要风险与缓解**：

| 风险 | 表现 | 缓解方式 |
|------|------|----------|
| 候选召回漏掉真匹配 | 后续模型无论多强都找不回来 | 提高召回 top-k，保留类型兼容但名称弱相似的候选 |
| 小模型误合并 | 下游字段被错误合并，污染数据 | 阈值偏保守，只自动接受高置信结果 |
| 领域漂移 | 新业务表命名方式变化，Precision 下降 | 线上抽检 + 难例回流 + 周期性 LoRA refresh |
| 类别不平衡复发 | 模型重新塌缩到多数类 | 训练集持续保持 1:1 或 hard negative 控制比例 |
| 大模型兜底成本失控 | 灰区过大导致节省不明显 | 校准阈值，按业务优先级和置信区间分层路由 |

**系统设计版收尾句**：

> 所以线上可用的关键不是声称 64M 小模型全面替代 7B，而是把它放在正确的位置：用便宜快速的小模型吃掉大部分简单候选，用大模型处理少量复杂灰区，整体上实现成本、延迟和精度的折中。