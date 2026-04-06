"""
读取 metrics.jsonl，生成：
  1. loss_comparison.png   — 标准残差 vs AttnRes 训练 loss 曲线
  2. layer_norm_comparison.png — 各层 hidden state norm（最后一个 epoch）
"""
import json, os
import matplotlib.pyplot as plt
import numpy as np

SAVE_DIR = '/root/minimind/experiments/attnres'
METRICS  = os.path.join(SAVE_DIR, 'metrics.jsonl')


def load_metrics(path):
    std_loss, atr_loss = [], []
    std_norms, atr_norms = None, None
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if 'loss' in rec:
                if rec['label'] == 'standard':
                    std_loss.append((rec['step'], rec['loss']))
                else:
                    atr_loss.append((rec['step'], rec['loss']))
            if 'layer_norms' in rec:
                # 只保留最后一个 epoch 的 norm
                if rec['label'] == 'standard':
                    std_norms = rec['layer_norms']
                else:
                    atr_norms = rec['layer_norms']
    return std_loss, atr_loss, std_norms, atr_norms


def plot_loss(std_loss, atr_loss, out_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    sx, sy = zip(*std_loss)
    ax_, ay = zip(*atr_loss)
    ax.plot(sx, sy, label='Standard Residual', color='steelblue',  linewidth=1.5)
    ax.plot(ax_, ay, label='AttnRes',           color='darkorange', linewidth=1.5)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Cross-Entropy Loss')
    ax.set_title('Training Loss: Standard Residual vs AttnRes')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"保存: {out_path}")


def plot_norms(std_norms, atr_norms, out_path):
    n_layers = len(std_norms)
    x = np.arange(n_layers)
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width/2, std_norms, width, label='Standard Residual', color='steelblue',  alpha=0.8)
    ax.bar(x + width/2, atr_norms, width, label='AttnRes',           color='darkorange', alpha=0.8)

    # 标注标准差（均匀性指标）
    std_cv = np.std(std_norms) / np.mean(std_norms)
    atr_cv = np.std(atr_norms) / np.mean(atr_norms)
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Hidden State L2 Norm (mean over batch/seq)')
    ax.set_title(f'Per-Layer Hidden State Norms (last epoch)\n'
                 f'CV: Standard={std_cv:.3f}  AttnRes={atr_cv:.3f}  '
                 f'(lower CV = more uniform)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'L{i}' for i in range(n_layers)])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"保存: {out_path}")


def main():
    std_loss, atr_loss, std_norms, atr_norms = load_metrics(METRICS)

    # 最终 loss 对比
    print(f"标准残差 最终loss: {std_loss[-1][1]:.4f}")
    print(f"AttnRes  最终loss: {atr_loss[-1][1]:.4f}")
    print(f"AttnRes 相对提升: {(std_loss[-1][1] - atr_loss[-1][1]) / std_loss[-1][1] * 100:.1f}%")
    print()

    # 各层 norm 均匀性
    if std_norms and atr_norms:
        print(f"标准残差 各层norm: {[f'{n:.2f}' for n in std_norms]}")
        print(f"AttnRes  各层norm: {[f'{n:.2f}' for n in atr_norms]}")
        print(f"标准残差 norm CV: {np.std(std_norms)/np.mean(std_norms):.3f}")
        print(f"AttnRes  norm CV: {np.std(atr_norms)/np.mean(atr_norms):.3f}")

    plot_loss(std_loss, atr_loss,
              os.path.join(SAVE_DIR, 'loss_comparison.png'))
    if std_norms and atr_norms:
        plot_norms(std_norms, atr_norms,
                   os.path.join(SAVE_DIR, 'layer_norm_comparison.png'))


if __name__ == '__main__':
    main()
