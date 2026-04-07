import json

def convert_with_reasoning(input_path, output_path):
    """将含 Mixtral 推理链的 Jellyfish 数据转换为 MiniMind SFT 格式（黑盒蒸馏）"""
    count = 0
    with open(input_path) as fin, open(output_path, 'w') as fout:
        for line in fin:
            item = json.loads(line)
            user_content = item["instruction"] + "\n" + item["input"]
            out = {
                "conversations": [
                    {"role": "user",      "content": user_content},
                    {"role": "assistant", "content": item["output"]}  # 含 Mixtral 推理链
                ]
            }
            fout.write(json.dumps(out, ensure_ascii=False) + '\n')
            count += 1
    print(f"转换完成：{count} 条 → {output_path}")

if __name__ == "__main__":
    convert_with_reasoning(
        "/root/autodl-tmp/jellyfish/train/with_generated_reasoning/sm_gen_m8x7b.jsonl",
        "/root/autodl-tmp/jellyfish/sm_distill_minimind.jsonl"
    )
