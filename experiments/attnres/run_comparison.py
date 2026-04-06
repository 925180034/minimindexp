"""
AttnRes vs 标准残差对比实验
从头训练两个小模型（相同配置、相同数据、相同随机种子），记录：
  - 每步 loss
  - 每 epoch 结束时各层 hidden state L2 norm（均匀性指标）
输出到 experiments/attnres/metrics.jsonl
"""
import sys, os, json, time, argparse
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from contextlib import nullcontext

sys.path.insert(0, '/root/minimind')
from model.model_minimind        import MiniMindForCausalLM, MiniMindConfig
from model.model_minimind_attnres import MiniMindForCausalLM as AttnResModel
from dataset.lm_dataset import SFTDataset
from transformers import AutoTokenizer

# ── 配置 ──────────────────────────────────────────────────────────────────────
DATA_PATH   = '/root/autodl-tmp/dataset/sft_compare_10k.jsonl'  # 1万条对比用子集
TOKENIZER   = '/root/minimind/model'
SAVE_DIR    = '/root/minimind/experiments/attnres'
HIDDEN_SIZE = 512      # 用更小的配置，跑得快
NUM_LAYERS  = 6
BATCH_SIZE  = 16
MAX_SEQ_LEN = 256
LR          = 3e-4
EPOCHS      = 5
LOG_STEPS   = 20       # 每隔多少步记录一次 loss
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED        = 42
# ──────────────────────────────────────────────────────────────────────────────


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_layer_norms(model, x):
    """收集各层 hidden state 的 L2 norm 均值（用于检验输出均匀性）"""
    norms = []
    model.eval()
    with torch.no_grad():
        hidden = model.model.dropout(model.model.embed_tokens(x))
        pos    = (model.model.freqs_cos[:x.shape[1]], model.model.freqs_sin[:x.shape[1]])
        prev_hiddens = []
        for layer in model.model.layers:
            if hasattr(layer, 'attn_res_query'):
                prev_hiddens.append(hidden)
                hidden, _ = layer(hidden, pos, prev_hiddens=prev_hiddens)
            else:
                hidden, _ = layer(hidden, pos)
            norms.append(hidden.norm(dim=-1).mean().item())
    model.train()
    return norms


def train_one_model(model_class, label, tokenizer, dataset, metrics_out):
    set_seed(SEED)
    cfg   = MiniMindConfig(hidden_size=HIDDEN_SIZE, num_hidden_layers=NUM_LAYERS)
    model = model_class(cfg).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    loader    = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    autocast  = torch.amp.autocast('cuda', dtype=torch.bfloat16) if DEVICE == 'cuda' else nullcontext()
    scaler    = torch.amp.GradScaler('cuda', enabled=False)

    # 用于记录 norm 的探针样本（固定不变）
    probe_x = next(iter(loader))[0][:4].to(DEVICE)

    step = 0
    for epoch in range(EPOCHS):
        for input_ids, labels in loader:
            input_ids, labels = input_ids.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with autocast:
                out  = model(input_ids, labels=labels)
                loss = out.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            step += 1

            if step % LOG_STEPS == 0:
                rec = {'label': label, 'epoch': epoch + 1, 'step': step, 'loss': loss.item()}
                metrics_out.append(rec)
                print(f"  [{label}] epoch={epoch+1} step={step} loss={loss.item():.4f}")

        # epoch 结束：记录各层 norm
        norms = get_layer_norms(model, probe_x)
        metrics_out.append({
            'label': label, 'epoch': epoch + 1, 'step': step,
            'layer_norms': norms
        })
        print(f"  [{label}] epoch={epoch+1} 各层 norm: {[f'{n:.2f}' for n in norms]}")

    return model


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    dataset   = SFTDataset(DATA_PATH, tokenizer, max_length=MAX_SEQ_LEN)
    print(f"数据集: {len(dataset)} 条\n")

    metrics = []

    print("=" * 50)
    print("训练标准残差模型...")
    train_one_model(MiniMindForCausalLM, 'standard', tokenizer, dataset, metrics)

    print("\n" + "=" * 50)
    print("训练 AttnRes 模型...")
    train_one_model(AttnResModel, 'attnres', tokenizer, dataset, metrics)

    # 保存 metrics
    out_path = os.path.join(SAVE_DIR, 'metrics.jsonl')
    with open(out_path, 'w') as f:
        for rec in metrics:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(f"\n指标已保存到: {out_path}")


if __name__ == '__main__':
    main()
