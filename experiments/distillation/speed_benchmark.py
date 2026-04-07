import os
import sys
import time
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

def benchmark(model, tokenizer, device, prompt, n_runs=50, max_new_tokens=100):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    # 预热
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    speeds = []
    for _ in range(n_runs):
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        t1 = time.time()
        n_tokens = out.shape[1] - input_len
        speeds.append(n_tokens / (t1 - t0))

    mean_speed = sum(speeds) / len(speeds)
    std_speed = (sum((s - mean_speed) ** 2 for s in speeds) / len(speeds)) ** 0.5
    return mean_speed, std_speed

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("/root/minimind/model")

    prompt = "Your task is to determine if the two attributes (columns) are semantically equivalent.\nAttribute A is [name: \"customer_id\", description: \"unique identifier for customer\"].\nAttribute B is [name: \"client_id\", description: \"unique identifier for client\"].\nAre they semantically equivalent? Choose from: [Yes, No]"

    results = {}
    weights = {
        "full_sft":   "full_sft",
        "distill_sft": "distill_sft",
    }

    for label, weight in weights.items():
        ckp = f"/root/minimind/out/{weight}_768.pth"
        if not os.path.exists(ckp):
            print(f"跳过 {label}：{ckp} 不存在")
            continue
        model = MiniMindForCausalLM(MiniMindConfig(hidden_size=768, num_hidden_layers=8))
        model.load_state_dict(torch.load(ckp, map_location=device), strict=True)
        model = model.eval().to(device)
        mean, std = benchmark(model, tokenizer, device, prompt)
        results[label] = (mean, std)
        print(f"{label}: {mean:.1f} ± {std:.1f} tokens/s")
        del model
        torch.cuda.empty_cache()

    # 保存结果
    with open("/root/minimind/experiments/distillation/speed_results.txt", "w") as f:
        f.write("推理速度基准测试（50次平均，max_new_tokens=100）\n")
        f.write(f"设备: {device}\n\n")
        for label, (mean, std) in results.items():
            f.write(f"{label}: {mean:.1f} ± {std:.1f} tokens/s\n")
        if "full_sft" in results and "distill_sft" in results:
            ratio = results["distill_sft"][0] / results["full_sft"][0]
            f.write(f"\ndistill_sft / full_sft 速度比: {ratio:.2f}x\n")
        # Jellyfish-7B 参考值（论文/AutoDL 实测）
        f.write("\n参考：Jellyfish-7B 约 30-50 tokens/s（A100 80G）\n")
    print("\n结果已保存到 speed_results.txt")

if __name__ == "__main__":
    main()
