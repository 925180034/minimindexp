# 文档五：DPO 与 GRPO 对齐算法

## 1. RLHF 全景图

```
传统 RLHF 三步流程：
┌─────────────────────────────────────────────────────────┐
│ Step 1: SFT                                             │
│   pretrain_model → 指令数据微调 → sft_model            │
├─────────────────────────────────────────────────────────┤
│ Step 2: 训练 Reward Model                               │
│   (x, y_w, y_l) 偏好对 → reward_model                  │
│   reward_model(x, y_w) > reward_model(x, y_l)          │
├─────────────────────────────────────────────────────────┤
│ Step 3: PPO 强化学习                                    │
│   policy_model 生成回复 → reward_model 打分 → PPO 更新  │
│   同时需要：Value Network（Critic）估计状态价值          │
└─────────────────────────────────────────────────────────┘

DPO：跳过 Step 2 和 Step 3，直接从偏好对优化策略
GRPO：简化 PPO，去掉 Value Network
```

---

## 2. DPO：Direct Preference Optimization

### 核心公式推导

**最优策略下 reward 和策略的关系（数学推导结论）：**

```
在 KL 约束的 RL 中，最优策略满足：
  π*(y|x) ∝ π_ref(y|x) × exp(r*(x,y) / β)

整理得 reward 可以被策略表达：
  r*(x,y) = β × log[π*(y|x) / π_ref(y|x)] + β × log Z(x)
             ↑这就是 reward 的隐式表达

把这个代入 Bradley-Terry 偏好模型：
  P(y_w > y_l | x) = σ(r(x,y_w) - r(x,y_l))
                   = σ(β × log[π(y_w|x)/π_ref(y_w|x)]
                       - β × log[π(y_l|x)/π_ref(y_l|x)])

最大化这个概率（最小化负对数）得到 DPO loss：
  L_DPO = -E[log σ(β × (log π_θ(y_w|x) - log π_ref(y_w|x))
                  - β × (log π_θ(y_l|x) - log π_ref(y_l|x)))]
```

### 直觉理解

```
DPO 优化的是：
  π_θ(y_w|x) 比 π_ref(y_w|x) 提升了多少（chosen 比参考模型好多少）
  π_θ(y_l|x) 比 π_ref(y_l|x) 下降了多少（rejected 比参考模型差多少）

  → 扩大 chosen 和 rejected 之间的 log-ratio 差距
```

### 代码实现

**代码位置：`trainer/train_dpo.py:24`**

```python
def logits_to_log_probs(logits, labels):
    # logits: [B, S, V]，labels: [B, S]
    log_probs = F.log_softmax(logits, dim=2)
    # gather: 取每个位置实际 token 的概率
    return torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    # 输出: [B, S]，每个位置的 log P(token)

def dpo_loss(ref_log_probs, policy_log_probs, mask, beta=0.1):
    # 对序列求和（整个回复的联合概率对数）
    ref_log_probs   = (ref_log_probs   * mask).sum(dim=1)   # [B]
    policy_log_probs = (policy_log_probs * mask).sum(dim=1)  # [B]
    
    # 前半 batch 是 chosen，后半是 rejected
    batch_size = ref_log_probs.shape[0]
    chosen_ref    = ref_log_probs[:batch_size//2]
    rejected_ref  = ref_log_probs[batch_size//2:]
    chosen_policy = policy_log_probs[:batch_size//2]
    rejected_policy = policy_log_probs[batch_size//2:]
    
    # log-ratio 差值
    pi_logratios  = chosen_policy  - rejected_policy    # π_θ 偏好差
    ref_logratios = chosen_ref     - rejected_ref       # π_ref 偏好差
    logits = pi_logratios - ref_logratios               # 相对于参考模型的改进
    
    loss = -F.logsigmoid(beta * logits)   # β 控制 KL 惩罚强度
    return loss.mean()
```

### 为什么初始 loss ≈ 0.693

```
训练开始时：policy_model = ref_model（都是 full_sft 权重）
→ 所有 log-ratio = 0
→ logits = 0
→ loss = -log σ(0) = -log(0.5) = log(2) ≈ 0.693

这是 DPO loss 的理论下界（无偏好时的 loss）
能从 0.693 下降说明模型在学习区分偏好
```

### 实验结果

```
数据：17,166 条偏好对（dpo.jsonl）
配置：batch_size=4，lr=4e-8（极小，防止破坏 SFT），1 epoch，4,292 步

步数    DPO Loss    含义
100     0.6958      接近理论起点 ln2=0.693
1000    0.4897      开始建立偏好
4292    0.4210      偏好对齐生效
```

### ref_model 的作用（面试重点）

```
如果没有 ref_model（纯 RL 目标）：
  模型会不择手段地提高 π_θ(y_w|x) / π_θ(y_l|x) 的比值
  → 可能通过降低对所有 token 的概率来实现（退化）
  → 语言能力灾难性遗忘

ref_model 提供 KL 惩罚：
  β × KL(π_θ || π_ref) = β × Σ log[π_θ(y|x) / π_ref(y|x)]
  
  这项惩罚让策略不能偏离参考模型太远
  β 越大 = KL 惩罚越强 = 越保守（本项目 β=0.1）
```

---

## 3. GRPO：Group Relative Policy Optimization

### PPO vs GRPO 核心区别

```
PPO 需要：
  1. Policy Model（策略模型，被优化）
  2. Reference Model（参考模型，提供 KL 约束）
  3. Reward Model（奖励模型，评分）
  4. Value Network / Critic（估计状态价值，作为基线）← GRPO 去掉这个

Value Network 问题：
  - 需要额外训练一个和策略模型等大的网络
  - 训练不稳定（两个网络相互影响）
  - 显存消耗翻倍

GRPO 思路：不用 Value Network，用同组样本的均值作为 baseline
```

### GRPO 算法流程

```
对每个 prompt x：

Step 1：采样 G 个回复（num_generations=4）
  y_1, y_2, y_3, y_4 ← 用策略模型采样

Step 2：计算每个回复的 reward
  r_i = RewardModel(x, y_i)    i=1,2,3,4

Step 3：组内标准化（Group Relative）
  mean_r = mean(r_1, r_2, r_3, r_4)
  std_r  = std(r_1, r_2, r_3, r_4)
  Â_i    = (r_i - mean_r) / std_r    ← 归一化优势值

  高于组均值的回复 → Â_i > 0（鼓励）
  低于组均值的回复 → Â_i < 0（抑制）

Step 4：PPO-clip 更新
  ratio_i = π_θ(y_i|x) / π_ref(y_i|x)   ← 概率比值
  L_i = min(ratio_i × Â_i,
            clip(ratio_i, 1-ε, 1+ε) × Â_i)
  L = -mean(L_i) + β × KL(π_θ || π_ref)
```

### 关键超参数

```
num_generations = 4    每个 prompt 采样 4 个回复
beta            = 0.04 KL 惩罚系数
epsilon         = 0.2  PPO clip 范围（ratio 限制在 [0.8, 1.2]）
max_gen_len     = 512  最大生成长度
```

### 为什么 GRPO 对 64M 模型效果有限

```
1. 模型容量瓶颈：
   64M 参数的语言理解能力本身有限
   Reward Model（InternLM2-1.8B）打分对应的语义理解
   超出了 64M 模型的表达范围

2. Reward 信号噪声：
   rlaif.jsonl 是通用偏好数据，不针对推理
   小模型的回复质量变化范围有限
   → reward 差异小 → 梯度信号弱

3. 理论视角：
   RL 的本质是在策略空间探索
   模型容量决定"策略空间"的大小
   64M 的策略空间太小，很快就穷尽了
   
结论：小模型更适合 SFT（监督直接），大模型才能充分发挥 RL 优势
      DeepSeek-R1 用 671B 参数跑 GRPO 才有显著效果
```

### Reward 曲线解读

```
step 451 ：Reward ≈ -3.0（初始，模型回复质量差）
step 1000：Reward ≈ -1.5（有所提升）
step 3510：Reward ≈ -1.0（波动较大，收益有限）

KL_ref 接近 0：说明策略没有大幅偏离参考模型（KL 惩罚生效）
Adv Std ≈ 1.0：组内归一化工作正常（标准化后方差为 1）
```

---

## 4. 面试常见追问

**Q: DPO 和 RLHF 哪个更好？**
A: 取决于场景。DPO 更简单稳定，不需要训练 RM 和 Value Network，适合数据充足的情况。传统 RLHF（PPO）更灵活，可以结合在线 rollout，适合复杂推理任务（如 DeepSeek-R1）。

**Q: GRPO 和 REINFORCE 有什么区别？**
A: REINFORCE 直接用 reward 作为梯度权重；GRPO 用组内相对 reward（减去均值除以标准差）作为优势函数，这相当于给 REINFORCE 加了一个 group-level baseline，减少方差，训练更稳定。

**Q: 为什么用 clip？**
A: clip 限制 ratio = π_θ/π_ref 的范围，防止策略更新步幅过大导致训练不稳定。如果回复 y 的概率在策略下大幅高于参考模型，梯度可能非常大，clip 截断这个异常值。

**Q: β 参数如何理解？**
A: β 是 KL 惩罚的系数。β→0 时，策略可以任意偏离参考模型（只追求 reward）；β→∞ 时，策略完全被锁定在参考模型附近（reward 无法优化）。合适的 β 让策略在"提升 reward"和"保持语言能力"之间取得平衡。
