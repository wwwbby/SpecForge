import json
import sys
import os
import torch
import re
from itertools import groupby
from transformers import AutoTokenizer


def find_ckpt_file(base_dir, target_index):
    """在子目录中快速定位 data_{idx}.ckpt 或 ckpt_{idx}.ckpt"""
    patterns = [f"data_{target_index}.ckpt", f"ckpt_{target_index}.ckpt"]
    for root, _, files in os.walk(base_dir):
        for p in patterns:
            if p in files:
                return os.path.join(root, p)
    return None


def get_mask_runs(mask):
    """
    将 [0, 0, 1, 1, 1, 0] 转换为 "0(2), 1(3), 0(1)" 的格式
    """
    runs = []
    for value, group in groupby(mask):
        count = len(list(group))
        runs.append(f"{int(value)}({count})")
    return " -> ".join(runs)


def count_and_verify_mask(file_path, target_index, model_path, cache_dir, max_len=2048):
    # 1. 初始化
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # 2. 读取 JSONL
    data = None
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == target_index:
                data = json.loads(line)
                break
    if not data:
        print(f"❌ 错误：JSONL 中找不到序号 {target_index}")
        return

    # 3. 读取 CKPT
    ckpt_path = find_ckpt_file(cache_dir, target_index)
    if not ckpt_path:
        print(f"❌ 错误：在 {cache_dir} 下找不到序号 {target_index} 的 ckpt 文件")
        return

    ckpt_content = torch.load(ckpt_path, map_location="cpu")
    actual_mask = ckpt_content.get("loss_mask", [])
    if isinstance(actual_mask, torch.Tensor):
        actual_mask = actual_mask.tolist()

    # 确保实际 mask 也遵循 2048 长度
    actual_mask = actual_mask[:max_len]

    # 4. 模拟模板拼接与 Mask 计算
    IM_START, IM_END = "<|im_start|>", "<|im_end|>\n"
    SYS_PROMPT = "You are a helpful assistant."

    expected_mask = []

    # --- System 部分 ---
    sys_text = f"{IM_START}system\n{SYS_PROMPT}{IM_END}"
    sys_tokens = tokenizer.encode(sys_text, add_special_tokens=False)
    expected_mask.extend([0] * len(sys_tokens))

    # --- 对话部分 ---
    for turn in data["conversations"]:
        role = turn["role"]
        content = turn["content"]

        header_tokens = tokenizer.encode(f"{IM_START}{role}\n", add_special_tokens=False)
        content_tokens = tokenizer.encode(content, add_special_tokens=False)
        footer_tokens = tokenizer.encode(IM_END, add_special_tokens=False)

        # 遵循你的逻辑：Assistant 的 Header + Content + Footer 均为 1
        if role == "assistant":
            turn_mask = [1] * (len(header_tokens) + len(content_tokens) + len(footer_tokens))
        else:
            turn_mask = [0] * (len(header_tokens) + len(content_tokens) + len(footer_tokens))

        expected_mask.extend(turn_mask)

    # 截断处理
    expected_mask = expected_mask[:max_len]

    # 5. 输出对比报告
    print("\n" + "=" * 80)
    print(f"🔍 校验数据序号: {target_index} | 目标文件: {os.path.basename(ckpt_path)}")
    print(f"📏 长度确认: 预期 {len(expected_mask)} tokens | 实际 {len(actual_mask)} tokens")
    print("=" * 80)

    # 描述 0/1 分布
    print("\n[Loss Mask 连续性描述]")
    print(f"💡 预期 (Logic): {get_mask_runs(expected_mask)}")
    print(f"📁 实际 (Ckpt):  {get_mask_runs(actual_mask)}")
    print("-" * 80)

    # 结果判定
    if expected_mask == actual_mask:
        print("✅ 结果：完美匹配！该 ckpt 确实对应 JSONL 中的这一条数据。")
    else:
        print("❌ 结果：不匹配！")
        # 寻找第一个不匹配的起始位置
        diff_pos = -1
        min_len = min(len(expected_mask), len(actual_mask))
        for i in range(min_len):
            if expected_mask[i] != actual_mask[i]:
                diff_pos = i
                break

        if diff_pos != -1:
            print(f"   ⚠️  首次冲突位置：Token {diff_pos}")
            # 打印冲突点周围的 token 以便调试
            # 注意：这里可能需要 input_ids 才能显示具体词，当前仅显示 mask 值的变化
            print(f"   预期值: {expected_mask[diff_pos]} | 实际值: {actual_mask[diff_pos]}")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    # 配置
    JSONL_FILE = ""
    MODEL_PATH = ""
    CACHE_DIR = ""

    # idx = int(sys.argv[1])
    for idx in range(8):
        print(f"TEST idx={idx}")
        count_and_verify_mask(JSONL_FILE, idx, MODEL_PATH, CACHE_DIR)
        print("\n")
