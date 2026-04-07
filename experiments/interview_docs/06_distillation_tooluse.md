# 文档六：知识蒸馏与 Tool Use

## Part 1：知识蒸馏

### 1. 白盒 vs 黑盒蒸馏

```
白盒蒸馏（Logit-based KD）：
┌──────────────────────────────────────────────────────┐
│  Teacher 模型（如 Jellyfish-7B，vocab=32000）         │
│       ↓ 前向传播                                      │
│  soft logits：[B, S, 32000]（每个 token 的概率分布）  │
│       ↓ KL 散度损失                                   │
│  Student 模型（MiniMind，vocab=6400）                 │
│  loss = KL(teacher_probs || student_probs)           │
└──────────────────────────────────────────────────────┘

问题：Teacher vocab=32000，Student vocab=6400
      词表不兼容，无法对齐 logits
      → 本项目不可用白盒蒸馏

黑盒蒸馏（Output-based KD）：
┌──────────────────────────────────────────────────────┐
│  Teacher 模型已经生成了输出文本（Mixtral-8x7B CoT）   │
│  → 直接用这些输出文本作为监督信号                    │
│  → 对 Student 做普通 SFT                             │
│  loss = CrossEntropy(student_output, teacher_text)   │
└──────────────────────────────────────────────────────┘
```

### 2. 蒸馏数据：sm_gen_m8x7b.jsonl

```
数据来源：Jellyfish 论文使用 Mixtral-8x7B 为 SM 训练集生成推理链

格式示例：
{
  "instruction": "Your task is to determine if the two attributes are semantically equivalent...",
  "input": "Attribute A is [name: 'customer_id', description: '...'].\nAttribute B is ...",
  "output": "To determine if Attribute A and Attribute B are semantically equivalent,
             let's analyze their names and descriptions.
             
             Attribute A: 'customer_id' refers to a unique identifier for customers...
             Attribute B: 'client_id' refers to a unique identifier for clients...
             
             Both attributes serve the same purpose... Therefore, they are semantically
             equivalent. Answer: Yes"
}

output 包含：
  1. 逐步推理过程（Mixtral 的思维链）
  2. 最终答案（Yes/No）
```

### 3. 数据转换与训练

```python
# experiments/distillation/prepare_distill_data.py
def convert_with_reasoning(input_path, output_path):
    for line in fin:
        item = json.loads(line)
        user_content = item["instruction"] + "\n" + item["input"]
        out = {
            "conversations": [
                {"role": "user",      "content": user_content},
                {"role": "assistant", "content": item["output"]}  # 含推理链
            ]
        }
```

```
普通 SFT（full_sft）：output = "Yes" 或 "No"（1-2个token）
蒸馏 SFT（distill_sft）：output = "推理链...Answer: Yes/No"（100-300个token）

区别：蒸馏数据教模型"怎么推理"，不只是"答案是什么"
```

### 4. 训练结果

```
配置：84,345 条，2 epochs，batch_size=32，lr=1e-5

Loss 曲线：
  Epoch 1 step 100：0.701  ← 开始学习推理格式
  Epoch 1 结束：    0.345  ← 推理链结构已学会
  Epoch 2 结束：    0.287  ← 进一步拟合 Mixtral 推理风格
```

### 5. 速度测试结果

```
设备：单卡 4090，max_new_tokens=100，50 次平均

┌────────────────────┬──────────────────────┐
│ 模型               │ 推理速度             │
├────────────────────┼──────────────────────┤
│ MiniMind 64M       │ 174 ± 5 tokens/s    │
│ Jellyfish-7B       │ ~30-50 tokens/s     │
│ （参考，A100）     │                      │
└────────────────────┴──────────────────────┘

MiniMind 速度约为 Jellyfish-7B 的 3.5x~6x
参数量：64M / 7000M ≈ 1/109
```

### 6. SM F1 的负结论（重要）

```
蒸馏模型 SM F1 = 0（全预测 No）

原因分析：
┌─────────────────────────────────────────────────┐
│  84K SM 数据：99.7% No                          │
│      ↓                                          │
│  Mixtral 推理链：99.7% 的推理链结论是 No        │
│      ↓                                          │
│  蒸馏：把"No 推理风格"学进了 Student 模型       │
│      ↓                                          │
│  结果：模型会写推理，但推理链结论永远是 No       │
└─────────────────────────────────────────────────┘

这揭示了黑盒蒸馏的局限性：
  蒸馏只是复制 Teacher 的输出分布
  如果 Teacher 的训练数据是偏斜的，Student 学到的也是偏斜的
  
  → 蒸馏≠知识提炼，更准确说是"行为模仿"

对比 LoRA 方案的成功：
  LoRA 通过数据工程（1:1平衡+多任务）解决了不平衡问题
  蒸馏方案把数据不平衡原样传递给了 Student
  
  结论：数据质量 > 模型结构/算法选择
```

---

## Part 2：Tool Use（工具调用）

### 1. 工具调用的数据格式

```
数据文件：dataset/agent_rl.jsonl

对话格式示例：
{
  "conversations": [
    {"role": "system",    "content": "你是一个可以调用工具的助手。可用工具：[{...}]"},
    {"role": "user",      "content": "帮我算 256 × 37"},
    {"role": "assistant", "content": "<tool_call>\n{\"name\": \"calculate_math\", \"arguments\": {\"expression\": \"256 * 37\"}}\n</tool_call>"},
    {"role": "tool",      "content": "{\"result\": \"9472\"}"},
    {"role": "assistant", "content": "256 乘以 37 等于 9472。"}
  ]
}
```

### 2. 工具调用的执行流程

```
┌─────────────────────────────────────────────────────────┐
│                  Agentic 执行循环                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  用户输入："帮我算 256 × 37"                            │
│      ↓                                                  │
│  模型生成（含工具定义的 system prompt）                  │
│      ↓                                                  │
│  检测到 <tool_call> 标签                                │
│      ↓                                                  │
│  解析 JSON：{"name": "calculate_math",                  │
│              "arguments": {"expression": "256 * 37"}}  │
│      ↓                                                  │
│  执行工具函数：eval("256 * 37") = 9472                  │
│      ↓                                                  │
│  将结果注入对话：{"result": "9472"}                     │
│      ↓                                                  │
│  模型继续生成："256 乘以 37 等于 9472。"                │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 3. 三种调用模式

#### 单工具调用
```
用户：帮我把 100 公里换算成英里
模型：<tool_call>{"name": "unit_converter", "arguments": {...}}</tool_call>
工具：{"result": 62.14}
模型：100 公里等于约 62.14 英里。
```

#### 链式调用（前一个工具的结果作为后一个工具的输入）
```
用户：生成一个 1-1000 的随机数，然后计算它的平方
模型：<tool_call>{"name": "random_number", "arguments": {"min":1,"max":1000}}</tool_call>
工具：{"result": 991}
模型：<tool_call>{"name": "calculate_math", "arguments": {"expression": "991**2"}}</tool_call>
工具：{"result": 982081}
模型：生成的随机数是 991，它的平方是 982081。
```

#### 并行调用（一次生成多个 tool_call 标签）
```
用户：查一下东京的天气，同时把 30 摄氏度转成华氏度
模型：<tool_call>{"name": "get_current_weather", "arguments": {"location": "Tokyo"}}</tool_call>
      <tool_call>{"name": "unit_converter", "arguments": {"value": 30, ...}}</tool_call>
（两个工具同时执行）
```

### 4. 测试结果（8个用例）

```
┌───┬──────────────────────────┬─────────────────────────────┬────┐
│ # │ 问题                     │ 工具调用                    │结果│
├───┼──────────────────────────┼─────────────────────────────┼────┤
│ 1 │ 256×37 计算              │ calculate_math              │ ✅ │
│ 2 │ 现在几点                 │ get_current_time            │ ✅ │
│ 3 │ 100km→英里               │ unit_converter              │ ✅ │
│ 4 │ 随机数+平方（链式）       │ random_number→calculate    │ ✅ │
│ 5 │ 北京天气                 │ get_current_weather         │ ⚠️ │
│   │                          │ （重复调用了两次）           │    │
│ 6 │ 美元兑人民币              │ get_exchange_rate           │ ✅ │
│ 7 │ 翻译"你好世界"           │ translate_text              │ ✅ │
│ 8 │ 东京天气+温度换算（并行） │ weather + unit_converter    │ ✅ │
└───┴──────────────────────────┴─────────────────────────────┴────┘

通过率：6/8（75%）
```

### 5. 面试常见追问

**Q: Tool Use 是怎么训练出来的？**
A: 通过 Agentic RL（`train_agent.py`），使用包含 `<tool_call>` 标签的多轮对话数据做 SFT+RL。模型学会了在合适的时机输出特定格式的 JSON，外部代码检测到标签后执行对应函数，将结果注入对话继续生成。

**Q: 如果模型输出的 tool_call JSON 格式不对怎么办？**
A: 实际系统需要加 JSON 解析错误处理，输出格式校验，以及 fallback 机制（如让模型重试）。本项目是 demo 级别，测试时 6/8 用例的 JSON 格式都是正确的。

**Q: 并行工具调用是怎么实现的？**
A: 模型在一次生成中输出多个 `<tool_call>` 标签，代码用正则匹配提取所有标签，顺序执行（本项目是顺序，真实系统可以并发），然后将所有结果拼接注入对话。
