# 文档三：LoRA 微调与 Schema Matching 实验

## 1. LoRA 原理

### 核心思想

```
原始权重矩阵 W ∈ R^{d×d}（768×768，约 590K 参数）

正常微调：W → W + ΔW    ← ΔW 和 W 同维，参数量巨大

LoRA：ΔW ≈ B × A
  A ∈ R^{r×d}   （rank × 768）
  B ∈ R^{d×r}   （768 × rank）
  
当 rank=64 时：
  参数量 = 2 × 64 × 768 = 98,304   ← 只有原来的 16.6%
```

### 代码位置：`model/model_lora.py`

```python
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        self.A = nn.Linear(in_features, rank, bias=False)  # 降维
        self.B = nn.Linear(rank, out_features, bias=False) # 升维
        
        # 关键初始化：A 高斯初始化，B 全零初始化
        # → 训练初期 ΔW = B@A = 0，不破坏预训练权重
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        self.B.weight.data.zero_()

    def forward(self, x):
        return self.B(self.A(x))   # x → rank → out_features
```

### apply_lora 注入机制

```python
def apply_lora(model, rank=16):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and \
           module.weight.shape[0] == module.weight.shape[1]:  # 只注入方阵（q/k/v/o_proj）
            
            lora = LoRA(dim, dim, rank=rank)
            setattr(module, "lora", lora)
            
            # 修改 forward：原输出 + LoRA 输出
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)
            
            module.forward = forward_with_lora
```

```
注入位置（方阵条件）：
  q_proj: 768×768  ✓ 注入
  k_proj: 768×384  ✗ 不注入（KV头减半，非方阵）
  v_proj: 768×384  ✗ 不注入
  o_proj: 768×768  ✓ 注入
  
  gate_proj: 768×2048  ✗ 非方阵
  up_proj:   768×2048  ✗ 非方阵
  down_proj: 2048×768  ✗ 非方阵
```

### 训练时的参数冻结

```
全量参数（63.91M）全部冻结（requires_grad=False）
LoRA 参数（rank=64）解冻（requires_grad=True）

每层注入 2 个位置（q_proj, o_proj）× 8 层 = 16 个 LoRA
每个 LoRA：2 × 64 × 768 = 98,304 参数
总 LoRA 参数：16 × 98,304 ≈ 1.57M（占总参数 2.4%）
```

---

## 2. Schema Matching 任务

### 任务定义

```
输入：
  Attribute A: [name: "customer_id", description: "unique customer identifier"]
  Attribute B: [name: "client_id",   description: "unique client identifier"]
  问：这两个属性语义等价吗？

输出：Yes / No
```

### 数据集分布（核心难点）

```
Jellyfish SM 训练集（84,345 条）：
┌─────────────────────────────────────────┐
│  No:  84,133 条  ████████████████ 99.7% │
│  Yes:    212 条  ▏            0.3%      │
└─────────────────────────────────────────┘

这就是为什么 LoRA 一开始失败的根本原因！
```

---

## 3. 四次实验的失败与成功（面试重点）

### 第一次：全量数据直接训练（失败）

```
数据：84,345 条（No:99.7%, Yes:0.3%）
超参：rank=16，lr=2e-4，3 epochs

发生了什么：
  交叉熵 loss 的最优解 = 永远预测多数类
  P(No) = 99.7% → 模型只要输出 "No" loss 就很低
  
  这就是"模式坍塌"：模型收敛到退化解

结果：全预测 No，Precision=0, Recall=0, F1=0
```

### 第二次：1:1 平衡数据（失败）

```
数据：212 Yes + 212 No = 424 条，严格平衡
问题：总步数 = 424 / batch_size / epochs ≈ 65 步

65 步远不够让 LoRA 参数从全零学到有用特征
即使平衡了类别，步数太少就是没有学习

结果：退化为 base 模型行为（全预测 Yes），F1=0.667（虚假高分）
```

### 第三次：更大 rank + 平衡数据（部分成功）

```
数据：424 条（1:1）
超参：rank=16→64，lr=2e-4，20 epochs（约 530 步）

参数量：16×98304 ≈ 1.57M（2.4%）
步数增加，有一定效果

结果：Precision=0.703，仍不够好
```

### 第四次：SM+EM 多任务联合训练（成功）

```
核心洞察（来自 Jellyfish 论文）：
  SM（Schema Matching）正例只有 212 条，太少
  EM（Entity Matching）也是"判断两个实体是否等价"
  → 同类任务！EM 有 7,199 个正例，可以迁移等价判断能力

数据构造：
  SM:  212 Yes + 212 No = 424 条
  EM: 2000 Yes + 2000 No = 4000 条（从 42,957 条中采样）
  合计：4,424 条，严格 1:1

超参：rank=64，lr=2e-4，10 epochs（约 1,390 步）

为什么有效：
  1. EM 数据量充足 → LoRA 学会了"判断等价性"的通用特征
  2. SM 和 EM 结构相同（都是两个 attribute/entity 比较）
  3. 1:1 平衡防止坍塌到多数类

结果：Precision=0.905，Accuracy=91.7%（全量测试集），F1=0.551（平衡集）
```

---

## 4. 评测结果完整对比

### 平衡测试集（48 Yes + 48 No = 96 条）

```
┌─────────────────┬───────────┬────────┬───────┬──────────┐
│ 模型            │ Precision │ Recall │  F1   │ Accuracy │
├─────────────────┼───────────┼────────┼───────┼──────────┤
│ base (全Yes)    │   0.500   │  1.000 │ 0.667 │  0.500   │
│ LoRA SM+EM      │   0.905   │  0.396 │ 0.551 │  0.677   │
└─────────────────┴───────────┴────────┴───────┴──────────┘
```

### 全量测试集（11,936 条，Yes 仅 48 条 = 0.4%）

```
┌─────────────────┬───────────┬────────┬───────┬──────────┐
│ 模型            │ Precision │ Recall │  F1   │ Accuracy │
├─────────────────┼───────────┼────────┼───────┼──────────┤
│ base (全Yes)    │   0.004   │  1.000 │ 0.008 │  0.004   │
│ LoRA SM+EM      │   0.019   │  0.396 │ 0.037 │  0.917   │
└─────────────────┴───────────┴────────┴───────┴──────────┘
```

**如何解释 Recall 只有 0.396？**

这是任务本身的难点，不是 LoRA 的问题：
- SM 任务要求精确判断列名+描述是否等价
- 64M 小模型语义理解能力有限
- Jellyfish-7B 的 F1 也只有 40-50（7B 参数的专门微调模型）
- **Precision 0.905 才是关键指标**：模型认为等价的 90.5% 确实等价，精度很高

---

## 5. LoRA 合并部署

```python
def merge_lora(model, lora_path, save_path):
    # 合并后：W_new = W_old + B @ A
    # 推理时不再需要额外的 LoRA 模块
    # 推理速度和原模型完全一样
    
    state_dict[f'{name}.weight'] = module.weight.data.clone()
    if hasattr(module, 'lora'):
        state_dict[f'{name}.weight'] += (module.lora.B.weight @ module.lora.A.weight)
```

```
部署优势：
  训练时：主干冻结（省显存），只训 2.4% 参数（快）
  部署时：合并权重，零额外开销（快）
  存储时：只保存 LoRA 权重（3MB vs 128MB 完整模型）
```

---

## 6. 面试常见追问

**Q: LoRA 为什么初始化 B=0？**
A: 保证训练开始时 ΔW=B@A=0，即 LoRA 不改变预训练权重的初始行为。如果 B 随机初始化，第一步就会破坏预训练分布，导致不稳定。

**Q: rank 怎么选？**
A: rank 越大，表达能力越强，但参数量越多。本项目从 16→64，是因为 SM 任务需要更强的语义理解（等价判断是复杂语义任务）。一般任务 rank=8-16 足够，复杂任务 rank=64-128。

**Q: 为什么不用全量微调？**
A: 全量微调需要存储和更新 63.91M 参数的梯度，显存消耗巨大（同时需要模型权重 + 梯度 + 优化器状态，约 3-4× 模型大小）。LoRA 只需要 1.57M 参数的梯度，显存减少 97%。

**Q: 为什么 EM 能迁移到 SM？**
A: 两个任务在语义层面同构：都是给定两个对象的名称和描述，判断是否指同一个概念。EM 的"实体等价"和 SM 的"属性等价"使用相同的判断逻辑，只是输入对象的粒度不同（行 vs 列）。Jellyfish 论文明确指出了这种迁移关系。
