import json
import os


def filter_dataset(original_path, generated_path, output_path):
    # 1. 收集已经存在的 ID
    existing_ids = set()

    if os.path.exists(generated_path):
        print(f"正在读取已生成的数据文件: {generated_path}...")
        with open(generated_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        existing_ids.add(str(data['id']))  # 统一转为字符串以防万一
                    except json.JSONDecodeError:
                        continue
        print(f"已找到 {len(existing_ids)} 条已存在的数据。")
    else:
        print(f"提示: 未找到已生成的文件 {generated_path}，将导出全部数据。")

    # 2. 遍历原始文件并筛选
    count_total = 0
    count_rest = 0

    print(f"正在筛选数据并保存至: {output_path}...")
    with open(original_path, 'r', encoding='utf-8') as fin, \
        open(output_path, 'w', encoding='utf-8') as fout:

        for line in fin:
            if not line.strip():
                continue

            count_total += 1
            try:
                item = json.loads(line)
                item_id = str(item.get('id'))

                # 如果 id 不在已生成的集合中，则保留
                if item_id not in existing_ids:
                    fout.write(json.dumps(item, ensure_ascii=False) + '\n')
                    count_rest += 1
            except json.JSONDecodeError:
                print(f"警告: 无法解析原始文件的某一行数据。")

    print("-" * 30)
    print(f"处理完成！")
    print(f"原始总数: {count_total}")
    print(f"跳过重复: {len(existing_ids)}")
    print(f"剩余保存: {count_rest}")


if __name__ == "__main__":
    # 路径配置
    ORIGINAL = ""
    GENERATED = ""
    OUTPUT = ""

    filter_dataset(ORIGINAL, GENERATED, OUTPUT)
