"""
Schema Matching 评测脚本
对比 full_sft 基础模型 vs full_sft + lora_schema_matching 的 F1
"""
import sys
import json
import random
import torch
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/root/minimind')
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import apply_lora, load_lora

# ── 配置 ──────────────────────────────────────────────────────────────
TEST_PATH   = '/root/autodl-tmp/jellyfish/sm_test_minimind.jsonl'
BASE_WEIGHT = '/root/minimind/out/full_sft_768.pth'
LORA_WEIGHT = '/root/minimind/out/lora_schema_matching_768.pth'
MODEL_DIR   = '/root/minimind/model'
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
N_SAMPLES   = 200   # 抽取测试样本数（Yes/No 各半，实际受限于 Yes 数量）
MAX_NEW_TOKENS = 16
SEED        = 42
LORA_RANK   = 64    # 必须和训练时的 --lora_rank 一致
# ──────────────────────────────────────────────────────────────────────


def load_model(use_lora=False):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = MiniMindForCausalLM(MiniMindConfig(hidden_size=768, num_hidden_layers=8))
    model.load_state_dict(torch.load(BASE_WEIGHT, map_location=DEVICE), strict=True)
    if use_lora:
        apply_lora(model, rank=LORA_RANK)
        load_lora(model, LORA_WEIGHT)
    return model.eval().to(DEVICE), tokenizer


def sample_balanced(test_path, n, seed):
    """从测试集中抽取 Yes/No 各半的样本，以 Yes 数量为上限做 1:1 平衡"""
    yes_items, no_items = [], []
    with open(test_path) as f:
        for line in f:
            item = json.loads(line)
            ans = item['conversations'][1]['content'].strip()
            if ans == 'Yes':
                yes_items.append(item)
            else:
                no_items.append(item)
    rng = random.Random(seed)
    # 以 Yes 数量为上限，严格 1:1
    half = min(n // 2, len(yes_items))
    sampled = rng.sample(yes_items, half) + rng.sample(no_items, half)
    rng.shuffle(sampled)
    return sampled


def predict(model, tokenizer, user_content):
    """返回模型预测的 Yes 或 No"""
    messages = [{"role": "user", "content": user_content}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id
        )
    generated = tokenizer.decode(
        output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True
    ).strip()
    # 提取首个 Yes/No
    for token in generated.split():
        if token.lower().startswith('yes'):
            return 'Yes'
        if token.lower().startswith('no'):
            return 'No'
    return generated[:10]  # 兜底返回原始输出前10字符


def compute_f1(preds, labels):
    tp = sum(p == 'Yes' and l == 'Yes' for p, l in zip(preds, labels))
    fp = sum(p == 'Yes' and l == 'No'  for p, l in zip(preds, labels))
    fn = sum(p == 'No'  and l == 'Yes' for p, l in zip(preds, labels))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    acc = sum(p == l for p, l in zip(preds, labels)) / len(labels)
    return {'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': acc}


def evaluate(model, tokenizer, samples):
    preds, labels = [], []
    for i, item in enumerate(samples):
        user_content = item['conversations'][0]['content']
        label        = item['conversations'][1]['content'].strip()
        pred = predict(model, tokenizer, user_content)
        preds.append(pred)
        labels.append(label)
        if (i + 1) % 200 == 0:
            print(f"  进度: {i+1}/{len(samples)}")
    return preds, labels


def load_all(test_path):
    """加载全量测试集"""
    samples = []
    with open(test_path) as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def main():
    print(f"加载全量测试集...")
    samples = load_all(TEST_PATH)
    yes_count = sum(s['conversations'][1]['content'].strip() == 'Yes' for s in samples)
    no_count  = len(samples) - yes_count
    print(f"  总计: {len(samples)} 条  Yes: {yes_count}  No: {no_count}\n")

    results = {}

    # ── 基础模型评测 ────────────────────────────────────────────────
    print("评测基础模型（full_sft）...")
    model, tokenizer = load_model(use_lora=False)
    preds_base, labels = evaluate(model, tokenizer, samples)
    results['base'] = compute_f1(preds_base, labels)
    del model
    torch.cuda.empty_cache()

    # ── LoRA 模型评测 ────────────────────────────────────────────────
    print("\n评测 LoRA 模型（full_sft + lora_schema_matching）...")
    model, tokenizer = load_model(use_lora=True)
    preds_lora, _ = evaluate(model, tokenizer, samples)
    results['lora'] = compute_f1(preds_lora, labels)
    del model
    torch.cuda.empty_cache()

    # ── 输出结果 ────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("Schema Matching 评测结果")
    print("=" * 50)
    for name, m in results.items():
        tag = "full_sft (base)" if name == 'base' else "full_sft + LoRA"
        print(f"\n[{tag}]")
        print(f"  Precision : {m['precision']:.4f}")
        print(f"  Recall    : {m['recall']:.4f}")
        print(f"  F1        : {m['f1']:.4f}")
        print(f"  Accuracy  : {m['accuracy']:.4f}")

    print("\n" + "=" * 50)
    f1_base = results['base']['f1']
    f1_lora = results['lora']['f1']
    print(f"F1 提升: {f1_base:.4f} → {f1_lora:.4f}  "
          f"(+{f1_lora - f1_base:.4f})")

    # 保存结果
    out_path = '/root/minimind/experiments/lora_schema_matching/f1_results.txt'
    with open(out_path, 'w') as f:
        f.write(f"测试集: {TEST_PATH}\n样本数: {N_SAMPLES}\n\n")
        for name, m in results.items():
            tag = "full_sft (base)" if name == 'base' else "full_sft + LoRA"
            f.write(f"[{tag}]\n")
            for k, v in m.items():
                f.write(f"  {k}: {v:.4f}\n")
            f.write("\n")
        f.write(f"F1 提升: {f1_base:.4f} → {f1_lora:.4f} (+{f1_lora-f1_base:.4f})\n")

    # 保存若干典型样本对比
    example_path = '/root/minimind/experiments/lora_schema_matching/example_outputs.txt'
    with open(example_path, 'w') as f:
        f.write("典型样本对比（前10条）\n" + "=" * 60 + "\n\n")
        for i in range(min(10, len(samples))):
            label = labels[i]
            f.write(f"[样本 {i+1}]\n")
            f.write(f"标签:      {label}\n")
            f.write(f"base 预测: {preds_base[i]}\n")
            f.write(f"lora 预测: {preds_lora[i]}\n\n")

    print(f"\n结果已保存到:\n  {out_path}\n  {example_path}")


if __name__ == '__main__':
    main()
