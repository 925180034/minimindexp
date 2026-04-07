# 文档二：Transformer 核心组件逐行解析

## 1. RMSNorm（替代 LayerNorm）

### 代码位置：`model/model_minimind.py:49`

```python
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        self.weight = nn.Parameter(torch.ones(dim))   # 可学习缩放因子 γ

    def norm(self, x):
        # x.pow(2).mean(-1) = 对最后一维求均方
        # rsqrt = 1/sqrt
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return (self.weight * self.norm(x.float())).type_as(x)
```

### LayerNorm vs RMSNorm 对比

```
LayerNorm:  x_norm = (x - μ) / sqrt(σ² + ε)  × γ + β
                      ↑减均值（中心化）
                                              ↑有偏移量β

RMSNorm:    x_norm = x / sqrt(mean(x²) + ε)  × γ
                      ↑不减均值，只除均方根
                                              ↑无偏移量β（少一半参数）
```

**为什么用 RMSNorm？**
- 计算更快（不需要计算均值）
- 论文证明效果相当
- LLaMA/Mistral 等现代 LLM 全部采用

---

## 2. RoPE 旋转位置编码

### 代码位置：`model/model_minimind.py:61`

### 核心思想

```
传统绝对位置编码：给每个位置加一个固定向量
  Q[pos] = Q + PE[pos]   ← 无法捕获相对位置关系

RoPE：将位置信息编码进旋转矩阵
  QK^T 的结果只依赖于 (pos_q - pos_k) 的相对距离
```

### 计算流程

```
1. 预计算频率矩阵（一次性，存为 buffer）：
   dim = head_dim = 96
   freqs[i] = 1 / (rope_theta ^ (2i/dim))   i=0,1,...,47
   
2. 对位置 t 的 token：
   cos[t, 2i]   = cos(t × freqs[i])
   sin[t, 2i]   = sin(t × freqs[i])

3. 旋转 Q/K：
   q_rotated = q × cos + rotate_half(q) × sin
   
   rotate_half(x)：将 x 后半部分移到前面，取负
   x = [x0, x1, ..., x47, x48, ..., x95]
   rotate_half(x) = [-x48, ..., -x95, x0, ..., x47]
```

### 为什么 RoPE 支持长上下文

```
标准 RoPE：rope_theta=10000，最大位置 2048
MiniMind：rope_theta=1e6（更大），最大位置 32768

YaRN 扩展（代码中实现）：
  对低频分量做线性插值，高频分量保持不变
  → 4x 上下文扩展（2048→32768 = 16x，用 factor=16）
```

---

## 3. GQA（分组查询注意力）

### 代码位置：`model/model_minimind.py:90`

### MHA vs GQA vs MQA 对比

```
MHA（多头注意力）：每个头都有独立的 Q/K/V
  8 heads Q：[B, 8, S, 96]
  8 heads K：[B, 8, S, 96]
  8 heads V：[B, 8, S, 96]
  KV-Cache 大小：2 × 8 × S × 96

GQA（分组查询，MiniMind 使用）：Q 头多，KV 头少
  8 heads Q：[B, 8, S, 96]
  4 heads K：[B, 4, S, 96]  ← KV 减半
  4 heads V：[B, 4, S, 96]
  KV-Cache 大小：2 × 4 × S × 96  → 减半！
  每 2 个 Q 头共享 1 个 KV 头

MQA（多查询）：所有 Q 头共享 1 个 KV 头
```

### repeat_kv 的作用

```python
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    # x shape: [B, S, 4, 96]   (4个KV头)
    # n_rep = 8/4 = 2
    # 输出:  [B, S, 8, 96]   (复制成8个，配合8个Q头做矩阵乘法)
```

### KV-Cache 的工作原理

```
推理时（逐 token 生成）：

第1步：输入 "今天天气"（4个token）
  → 计算 K1, V1（形状 [1,4,4,96]）
  → 缓存 K1, V1
  → 输出 "怎"

第2步：输入 "怎"（只有1个新token）
  → 新 K2, V2（形状 [1,1,4,96]）
  → 拼接：K = cat([K1, K2])（形状 [1,5,4,96]）
  → 不用重算前面的K/V，节省大量计算
```

代码实现：
```python
if past_key_value is not None:
    xk = torch.cat([past_key_value[0], xk], dim=1)  # 拼接历史K
    xv = torch.cat([past_key_value[1], xv], dim=1)  # 拼接历史V
```

---

## 4. Attention 完整前向传播

### 代码位置：`model/model_minimind.py:109`

```
输入 x: [B, S, 768]
    ↓
Q = x × W_Q  →  [B, S, 8×96] = [B, S, 768]
K = x × W_K  →  [B, S, 4×96] = [B, S, 384]
V = x × W_V  →  [B, S, 4×96] = [B, S, 384]
    ↓
reshape:
Q: [B, S, 8, 96]
K: [B, S, 4, 96]
V: [B, S, 4, 96]
    ↓
QK-Norm (RMSNorm on head_dim=96)  ← 防止注意力分数爆炸
    ↓
RoPE 旋转（注入位置信息）
    ↓
repeat_kv(K, 2): [B, S, 8, 96]
repeat_kv(V, 2): [B, S, 8, 96]
    ↓
transpose: Q [B,8,S,96] K [B,8,S,96] V [B,8,S,96]
    ↓
Flash Attention（训练时）：
  scores = Q @ K.T / sqrt(96)   因果掩码
  output = softmax(scores) @ V  [B, 8, S, 96]

或手动实现（推理时有 KV-Cache）：
  scores = Q @ K.T / sqrt(96)
  scores += causal_mask（上三角 -inf）
  output = softmax(scores) @ V
    ↓
reshape: [B, S, 8×96] = [B, S, 768]
    ↓
output = output × W_O  →  [B, S, 768]
```

---

## 5. SwiGLU FeedForward

### 代码位置：`model/model_minimind.py:134`

```python
class FeedForward(nn.Module):
    def forward(self, x):
        # gate_proj: x → [B, S, 2048]
        # up_proj:   x → [B, S, 2048]
        # act_fn = SiLU（sigmoid linear unit）
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

### GLU（门控线性单元）原理

```
标准 FFN：
  out = W2 × ReLU(W1 × x)

SwiGLU：
  gate = SiLU(W_gate × x)    ← 门控信号，值域在 [0, +∞)
  up   = W_up × x            ← 内容信号
  out  = W_down × (gate ⊙ up) ← 逐元素乘法，门控过滤信息

SiLU(x) = x × sigmoid(x)    ← 比 GELU 更平滑
```

**为什么有三个矩阵（gate/up/down）？**
门控机制需要两路信号相乘，因此 FFN 要用三个线性层：gate 和 up 维度相同（2048），经过门控后用 down 投回 hidden_size（768）。

---

## 6. MiniMindBlock（一个 Transformer 层）

```python
def forward(self, hidden_states, position_embeddings, ...):
    residual = hidden_states                           # 保存残差

    # === 自注意力部分（Pre-Norm）===
    hidden_states = self.input_layernorm(hidden_states) # RMSNorm 在前
    hidden_states, present_kv = self.self_attn(hidden_states, ...)
    hidden_states = residual + hidden_states            # 残差相加

    # === FFN 部分（Pre-Norm）===
    hidden_states = hidden_states + self.mlp(
        self.post_attention_layernorm(hidden_states)    # RMSNorm 在前
    )
    return hidden_states, present_kv
```

### Pre-Norm vs Post-Norm

```
Post-Norm（原始 Transformer）：
  x → Sublayer → + → LayerNorm → 输出

Pre-Norm（现代 LLM）：
  x → LayerNorm → Sublayer → + → 输出
      ↑先归一化再处理

Pre-Norm 优点：
  - 梯度更稳定（归一化在前，梯度传播路径更短）
  - 不需要 warmup 就能训练

Pre-Norm 缺点（本项目 AttnRes 要解决的问题）：
  - 各层 hidden state norm 随深度线性增长
  - 深层贡献被稀释（Hidden State Dilution）
```

---

## 7. 权重共享：embed_tokens 与 lm_head

```python
# MiniMindForCausalLM.__init__
self.model.embed_tokens.weight = self.lm_head.weight
```

```
输入端：token_id → embedding → [768维向量]
         使用 embed_tokens.weight 做查表

输出端：hidden_state [768维] → logits [6400维]
         使用 lm_head.weight 做线性变换

两者权重共享（Tied Weights）：
  - 减少参数量（节省 6400×768 = 4.9M 参数）
  - 语义一致性：相似 token 在输入空间相似，输出空间预测概率也相近
```

---

## 8. 自回归生成流程

```
generate() 核心循环（model/model_minimind.py:249）：

for _ in range(max_new_tokens):
    1. 前向传播（只传新 token，KV-Cache 存历史）
       outputs = model(input_ids[:, past_len:], past_key_values=past_kv)
    
    2. 取最后一个位置的 logits
       logits = outputs.logits[:, -1, :]   # [B, 6400]
    
    3. 温度缩放
       logits = logits / temperature
    
    4. top-k 过滤（保留概率最高的 k 个 token）
    
    5. top-p 核采样（保留累积概率 > p 之前的 token）
    
    6. 采样
       next_token = multinomial(softmax(logits))
    
    7. 拼接到序列末尾
       input_ids = cat([input_ids, next_token])
    
    8. 遇到 EOS token 停止
```
