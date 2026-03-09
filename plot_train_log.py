import json
import matplotlib.pyplot as plt
import pandas as pd


def plot_jsonl_data(file_path, save_path):
    steps = []
    losses = []
    accuracies = []

    # 1. 读取并解析 JSONL 文件
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                steps.append(int(data['step']))
                # 转换字符串为浮点数
                losses.append(float(data['loss']))
                accuracies.append(float(data['acc']))

    # 2. 转换为 DataFrame 方便处理（可选，但推荐）
    df = pd.DataFrame({
        'step': steps,
        'loss': losses,
        'acc': accuracies
    }).sort_values('step')

    # 3. 创建画布，画两张子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 绘制 Loss 图
    ax1.plot(df['step'], df['loss'], color='tab:red', label='Loss')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    # 绘制 Accuracy 图
    ax2.plot(df['step'], df['acc'], color='tab:blue', label='Accuracy')
    ax2.set_title('Training Accuracy')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

    # 4. 调整布局并展示
    plt.tight_layout()
    # plt.show()
    # 如果想保存图片，取消下面这行的注释
    plt.savefig(save_path)

if __name__ == "__main__":
    # 替换成你的文件名
    JSONL_FILE_PATH = ""
    SAVE_FIG_PATH = ""
    plot_jsonl_data(JSONL_FILE_PATH, SAVE_FIG_PATH)
