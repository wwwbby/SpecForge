import torch
import os
import shutil
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# 配置路径
SRC_ROOT = Path("")
DST_ROOT = Path("")
MAX_S = 2048


def process_single_file(file_path):
    """单个文件处理逻辑，供进程池调用"""
    try:
        # 计算输出路径
        relative_path = file_path.relative_to(SRC_ROOT)
        output_path = DST_ROOT / relative_path

        # 加载数据 (weights_only=False 兼容性更好)
        data = torch.load(file_path, weights_only=False)
        current_s = data['input_ids'].shape[0]

        # 1. 截断处理
        if current_s > MAX_S:
            data['input_ids'] = data['input_ids'][:MAX_S]
            data['loss_mask'] = data['loss_mask'][:MAX_S]
            data['hidden_state'] = data['hidden_state'][:, :MAX_S, :]
            data['aux_hidden_state'] = data['aux_hidden_state'][:, :MAX_S, :]

            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(data, output_path)
            return "truncated", None

        # 2. 直接复制
        elif current_s == MAX_S:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, output_path)
            return "copied", None

        # 3. 跳过
        else:
            return "skipped", current_s

    except Exception as e:
        return "error", str(e)


def main():
    # 扫描文件
    print("正在扫描文件...")
    ckpt_files = list(SRC_ROOT.rglob("*.ckpt"))
    total_files = len(ckpt_files)
    print(f"共发现 {total_files} 个文件。开始并行处理...")

    # 统计计数
    counts = {"truncated": 0, "copied": 0, "skipped": 0, "error": 0}

    # 设置进程数：通常设为 CPU 核心数，或者根据磁盘 I/O 承载能力调整
    # 如果是机械硬盘，建议设小一点（如 4-8）；如果是 SSD 或高速分布式文件系统，可以设大一点
    max_workers = min(os.cpu_count(), 16)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = {executor.submit(process_single_file, f): f for f in ckpt_files}

        # 使用 tqdm 实时刷新
        with tqdm(total=total_files, desc="并行处理中") as pbar:
            for future in as_completed(futures):
                status, info = future.result()
                counts[status] += 1

                # 如果是跳过或错误，打印提示
                if status == "skipped":
                    file_name = futures[future].name
                    pbar.write(f"[跳过] {file_name} (s={info})")
                elif status == "error":
                    file_name = futures[future].name
                    pbar.write(f"[错误] {file_name}: {info}")

                pbar.update(1)

    # 最终汇总
    print("\n" + "=" * 40)
    print(f"处理任务完成！使用进程数: {max_workers}")
    print(f"1. 截断保存 (s > {MAX_S}):  {counts['truncated']} 个")
    print(f"2. 直接复制 (s = {MAX_S}):  {counts['copied']} 个")
    print(f"3. 忽略跳过 (s < {MAX_S}):  {counts['skipped']} 个")
    if counts['error'] > 0:
        print(f"4. 发生错误: {counts['error']} 个")
    print(f"输出根目录: {DST_ROOT}")
    print("=" * 40)


if __name__ == "__main__":
    main()
