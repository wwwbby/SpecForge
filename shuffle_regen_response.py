import json
import random
import os
from tqdm import tqdm


def shuffle_jsonl(input_path, output_path, seed=42):
    # 1. 设置随机种子以保证可重复性
    random.seed(seed)

    if not os.path.exists(input_path):
        print(f"❌ 错误：找不到输入文件 {input_path}")
        return

    print(f"正在读取文件: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        # 使用 readlines 一次性读入内存，10w 行数据通常仅占用几百 MB
        lines = f.readlines()

    total_lines = len(lines)
    print(f"共读取 {total_lines} 条数据。正在进行打乱 (Seed: {seed})...")

    # 2. 原地打乱
    random.shuffle(lines)

    print(f"正在保存至: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in tqdm(lines, desc="写入进度"):
            f.write(line)

    print(f"✅ 完成！已生成打乱后的文件。")


if __name__ == "__main__":
    INPUT_FILE = ""
    OUTPUT_FILE = ""

    shuffle_jsonl(INPUT_FILE, OUTPUT_FILE)
