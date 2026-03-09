import json
import os
import math
from transformers import AutoTokenizer
from tqdm import tqdm


def get_percentile(sorted_data, percentile):
    """计算分位数"""
    if not sorted_data:
        return 0
    index = (len(sorted_data) - 1) * percentile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return sorted_data[int(index)]
    # 线性插值
    return sorted_data[lower] * (upper - index) + sorted_data[upper] * (index - lower)


def count_jsonl_tokens(file_path, model_path):
    # 1. 加载 Tokenizer
    print(f"正在从 {model_path} 加载 Tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=True
        )
    except Exception as e:
        print(f"加载 Tokenizer 失败: {e}")
        return

    # 统计变量
    all_lengths = []
    total_tokens = 0
    count_gt_2048 = 0
    count_gt_4096 = 0
    count_gt_8192 = 0

    if not os.path.exists(file_path):
        print(f"错误：找不到文件 {file_path}")
        return

    # 预扫描行数
    print("正在预扫描文件行数...")
    with open(file_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)

    # 2. 遍历处理
    print(f"开始处理数据（共 {total_lines} 条）:")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total_lines, desc="Processing", unit="lines"):
            if not line.strip():
                continue

            try:
                data = json.loads(line)
                conversations = data.get("conversations", [])

                # 拼接内容
                full_text = ""
                for turn in conversations:
                    if turn.get("role") in ["user", "assistant"]:
                        full_text += turn.get("content", "")

                # 计算长度
                token_count = len(tokenizer.encode(full_text))

                # 记录数据
                all_lengths.append(token_count)
                total_tokens += token_count

                # 阈值计数
                if token_count > 2048: count_gt_2048 += 1
                if token_count > 4096: count_gt_4096 += 1
                if token_count > 8192: count_gt_8192 += 1

            except Exception:
                continue

    # 3. 计算汇总信息
    record_count = len(all_lengths)
    if record_count > 0:
        all_lengths.sort()  # 排序用于计算分位数

        avg_tokens = total_tokens / record_count
        p90 = get_percentile(all_lengths, 0.90)
        p95 = get_percentile(all_lengths, 0.95)
        p99 = get_percentile(all_lengths, 0.99)

        pct_2048 = (count_gt_2048 / record_count) * 100
        pct_4096 = (count_gt_4096 / record_count) * 100
        pct_8192 = (count_gt_8192 / record_count) * 100

        # 4. 打印报告
        print("\n" + "=" * 45)
        print(f"{'统计项目':<18} | {'数值':<20}")
        print("-" * 42)
        print(f"{'总记录数':<18} | {record_count}")
        print(f"{'总 Token 数':<18} | {total_tokens}")
        print(f"{'平均 Token 数':<18} | {avg_tokens:.2f}")
        print("-" * 42)
        print(f"{'90 分位数 (P90)':<18} | {p90:.2f}")
        print(f"{'95 分位数 (P95)':<18} | {p95:.2f}")
        print(f"{'99 分位数 (P99)':<18} | {p99:.2f}")
        print("-" * 42)
        print(f"{'> 2048 Token':<18} | {count_gt_2048} ({pct_2048:.2f}%)")
        print(f"{'> 4096 Token':<18} | {count_gt_4096} ({pct_4096:.2f}%)")
        print(f"{'> 8192 Token':<18} | {count_gt_8192} ({pct_8192:.2f}%)")
        print("=" * 45)
    else:
        print("\n未发现有效数据。")


if __name__ == "__main__":
    MODEL_DIR = ""
    DATA_FILE = ""

    count_jsonl_tokens(DATA_FILE, MODEL_DIR)
