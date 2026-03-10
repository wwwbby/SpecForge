import os
import json

FIXED_TEXT = ""  # 预设的固定提示文本，例如：'请复述语音内容,并进行内容总结,不超过500字'

def generate_wav_jsonl(root_dir, output_file):
    """
    遍历目录下的所有 wav 文件并生成指定格式的 jsonl 文件
    """
    # 预设的固定文本
    fixed_text = FIXED_TEXT
    # 确保根目录路径以斜杠结尾，方便后续截取相对路径
    root_dir = os.path.normpath(root_dir)

    with open(output_file, 'w', encoding='utf-8') as f:
        # os.walk 会递归进入所有子目录
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.lower().endswith('.wav'):
                    # 获取文件的绝对路径
                    full_path = os.path.join(dirpath, filename)

                    # 计算相对于 root_dir 的相对路径
                    relative_path = os.path.relpath(full_path, root_dir)

                    # 构建数据字典
                    data = {
                        "text": fixed_text,
                        "audio": relative_path
                    }

                    # 写入 jsonl (每一行是一个 json 对象)
                    f.write(json.dumps(data, ensure_ascii=False) + '\n')

    print(f"处理完成！数据已保存至: {output_file}")

if __name__ == "__main__":
    # 配置路径
    TARGET_DIR = ""
    OUTPUT_JSONL = ""

    if os.path.exists(TARGET_DIR):
        generate_wav_jsonl(TARGET_DIR, OUTPUT_JSONL)
    else:
        print(f"错误：目录 {TARGET_DIR} 不存在，请检查路径。")
