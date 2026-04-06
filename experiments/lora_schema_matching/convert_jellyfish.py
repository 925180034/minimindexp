import json


def convert(input_path, output_path, max_samples=None):
    """将 Jellyfish 格式转为 MiniMind SFT 格式"""
    count = 0
    with open(input_path) as fin, open(output_path, 'w') as fout:
        for i, line in enumerate(fin):
            if max_samples and i >= max_samples:
                break
            item = json.loads(line)
            user_content = item["instruction"] + "\n" + item["input"]
            out = {
                "conversations": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": item["output"]}
                ]
            }
            fout.write(json.dumps(out, ensure_ascii=False) + '\n')
            count += 1
    print(f"转换完成: {input_path} -> {output_path}  ({count} 条)")


if __name__ == "__main__":
    convert(
        "/root/autodl-tmp/jellyfish/train/direct_answer_only/schema_matching.jsonl",
        "/root/autodl-tmp/jellyfish/sm_train_minimind.jsonl"
    )
    convert(
        "/root/autodl-tmp/jellyfish/test/seen_tasks/schema_matching.jsonl",
        "/root/autodl-tmp/jellyfish/sm_test_minimind.jsonl"
    )
