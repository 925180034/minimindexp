# 文档四：AttnRes 架构改造

## 1. 问题背景：PreNorm Hidden State Dilution

### 标准残差的累积问题

把 8 层 Transformer 的残差连接展开：

```
h0 = embedding(input)
h1 = h0 + f1(Norm(h0))
h2 = h1 + f2(Norm(h1))
h3 = h2 + f3(Norm(h2))
...
h8 = h0 + f1(·) + f2(·) + f3(·) + ... + f8(·)
         ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑
         每项权重都是固定的 1
```

后果：
```
层数越深，h_L 里累积的项越多
→ ||h_L|| 随深度线性增长（O(L)）

Norm(h_L) 归一化后，f_L(Norm(h_L)) 的输出相对越来越小
→ 深层每层的实际贡献占比越来越低
→ 这就是 Hidden State Dilution（隐状态稀释）

实验数据（本项目测量）：
  标准残差各层 norm：12 → 17 → 21 → 26 → 32 → 40（单调递增）
  最后一层是第一层的 3.2 倍
```

---

## 2. AttnRes 解决方案（Moonshot AI, 2026）

### 核心改动：固定权重 → 可学习 Softmax 权重

```
标准残差：
  h_l = h_{l-1} + f_l(Norm(h_{l-1}))
        ↑固定权重 1

AttnRes：
  h_l = [Σ α_{i→l} · h_i] + f_l(Norm(h_{l-1}))
         ↑softmax 加权聚合前面所有层的输出
  
  α_{i→l} = softmax(w_l · keys_i)    for i = 0, 1, ..., l-1
  
  w_l：第 l 层独有的可学习伪查询向量（d维）
  keys_i = mean(h_i, dim=seq)：第 i 层输出对 seq 维取均值
```

### Softmax 的关键约束

```
Σ α_{i→l} = 1    （softmax 归一化约束）

无论有多少层：
  残差项的总权重永远等于 1
  → 不会随深度线性增长
  → 模型可以自主选择"参考哪几层"
```

---

## 3. 代码实现

### 代码位置：`model/model_minimind_attnres.py`

#### Block 初始化（新增一行）

```python
class MiniMindBlock(nn.Module):
    def __init__(self, layer_id, config):
        super().__init__()
        # ... 原有代码不变 ...
        
        # 新增：可学习伪查询向量，零初始化
        # 零初始化 → 第一步等价于标准残差，训练稳定
        self.attn_res_query = nn.Parameter(torch.zeros(config.hidden_size))
```

#### Block 前向传播（修改残差计算）

```python
def forward(self, x, pos_cis, past_kv=None, use_cache=False, prev_hiddens=None):
    # 自注意力（不变）
    h_attn, present_kv = self.self_attn(self.input_layernorm(x), ...)
    
    # ====== 残差计算（核心改动）======
    if prev_hiddens and len(prev_hiddens) > 0:
        # 堆叠前面所有层的输出
        stacked = torch.stack(prev_hiddens, dim=0)   # [L, B, S, H]
        
        # keys: 对序列维度取均值，得到每层的"代表向量"
        keys = stacked.mean(dim=2)                   # [L, B, H]
        
        # 用伪查询向量计算注意力分数
        scores = torch.einsum('h,lbh->lb', self.attn_res_query, keys)  # [L, B]
        
        # softmax 归一化
        weights = torch.softmax(scores, dim=0)       # [L, B]，对层维度
        
        # 加权聚合：用 weights 加权求和前面所有层的输出
        residual = torch.einsum('lb,lbsh->bsh', weights, stacked)      # [B, S, H]
    else:
        residual = x    # 第一层退化为标准残差
    
    hidden_states = residual + self.drop(h_attn)
    hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
    return hidden_states, present_kv
```

#### Model 前向传播（维护历史列表）

```python
def forward(self, input_ids, ...):
    hidden_states = self.embed_tokens(input_ids)
    
    prev_hiddens = []   # 存储前面所有层的输出
    
    for layer in self.layers:
        hidden_states, present = layer(
            hidden_states,
            pos_cis,
            prev_hiddens=prev_hiddens   # 传入历史
        )
        prev_hiddens.append(hidden_states.detach())   # detach！避免内存爆炸
    
    return self.norm(hidden_states), ...
```

**为什么要 detach()？**
```
如果不 detach，计算图会包含前面所有层的梯度路径
8 层的话，第 8 层的梯度要沿着 1→2→3→4→5→6→7→8 计算
内存消耗 O(L²)

detach() 后：
  梯度只通过当前层传播
  内存消耗 O(L)
  这是实现上的权衡（牺牲少量梯度精度换内存）
```

---

## 4. 实验结果

### 训练配置

```
hidden_size = 512（比完整模型小，加速实验）
num_layers  = 6
数据量      = 10,000 条 SFT 数据
Epochs      = 5
从零开始训练（不加载预训练权重）
```

### Loss 对比

```
┌─────────────┬─────────────┬──────────┐
│ 模型        │ 最终 Loss   │ 提升     │
├─────────────┼─────────────┼──────────┤
│ 标准残差    │   1.2204    │   —      │
│ AttnRes     │   1.2095    │  +0.9%   │
└─────────────┴─────────────┴──────────┘
```

### 各层 Hidden State Norm（核心指标）

```
标准残差（单调递增，稀释严重）：
  Layer  0   1   2   3   4   5
  Norm  12  17  21  26  32  40
        ↑______________________↑
        第5层是第0层的 3.3 倍

AttnRes（趋于稳定，第4-5层反而下降）：
  Layer  0   1   2   3   4   5
  Norm  16  20  21  26  26  22
                          ↗↘
                      后期层自适应回退
```

### 变异系数（CV = 标准差/均值，衡量均匀性）

```
标准残差 CV = 0.368   （高变异，不均匀）
AttnRes  CV = 0.153   （低变异，均匀）
                         ↓58%

CV 越低 = 各层贡献越均匀 = 深层没有被稀释
```

---

## 5. 面试常见追问

**Q: loss 只提升 0.9%，这个实验有意义吗？**
A: 有。论文在 48B 模型/1.4T tokens 上验证，小模型（64M）效果受限是预期的。AttnRes 的主要价值是：(1) 训练稳定性——各层 norm 均匀，梯度分布更好；(2) 大规模效果放大——小模型的 0.9% loss 差异在大模型/长训练上会放大。CV 下降 58% 是定量复现了论文核心结论。

**Q: 为什么 scores 对"层"维度做 softmax 而不是对序列维度？**
A: 这里的 softmax 是在决定"参考哪几层"（层间注意力），不是在计算 token 间的注意力。每一层对应一个权重，所有层的权重之和为 1。序列维度已经在 keys = mean(h_i, dim=seq) 这步被聚合掉了。

**Q: AttnRes 和 Highway Network、Dense Connection 有什么区别？**
A: 
- Highway Network：可学习的门控，但只连接相邻层
- DenseNet（Dense Connection）：连接所有前面的层，但权重固定为 1
- AttnRes：连接所有前面的层，权重可学习且归一化（softmax），是三者的进化版

**Q: 伪查询向量为什么初始化为 0？**
A: 零初始化时 scores=0，softmax(0...0)=均匀分布，所以初始状态等价于对前面所有层均匀平均——这接近于标准残差的行为，确保训练初期稳定。
