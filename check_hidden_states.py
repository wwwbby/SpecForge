import torch
import os
from transformers import AutoTokenizer
from safetensors.torch import load_file


MODEL_PATH = ""
SAFETENSOR_FILE_NAME = ""
LM_HEAD_KEY = ""

WEIGHT_FILE = os.path.join(MODEL_PATH, SAFETENSOR_FILE_NAME)


def inspect_single_ckpt(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    print(f"Checking file: {file_path}")
    print("=" * 60)

    # ================= 配置区 =================
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # 加载数据
    try:
        data = torch.load(file_path, map_location='cpu')
    except Exception as e:
        print(f"Load failed: {e}")
        return

    if not isinstance(data, dict):
        print(f"Unexpected data type: {type(data)}")
        return

    for key, value in data.items():
        if torch.is_tensor(value):
            # 打印基础信息
            print(f"Key: {key:<18}")
            print(f"  - Shape: {list(value.shape)}")
            print(f"  - Dtype: {value.dtype}")

            if "loss_mask" in key:
                print(f"  - Value: {value}")
                print(f"  - sum(value): {sum(value)}")

            # 如果是 Hidden States，进一步拆解维度
            if 'hidden_state' in key:
                # 统计数值范围，看看到底是哪一层在爆炸
                v_float = value.float()
                print(f"  - Value Range: [{v_float.min().item():.2f}, {v_float.max().item():.2f}]")
                print(f"  - Mean: {v_float.mean().item():.4f}, Std: {v_float.std().item():.4f}")

                if key == "hidden_state":
                    DEVICE = "npu" if torch.npu.is_available() else "cpu"
                    print(f"Loading Tokenizer...")
                    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

                    print(f"Loading LM Head Weight from {WEIGHT_FILE}...")
                    try:
                        weights = load_file(WEIGHT_FILE, device=DEVICE)
                        lm_head_weight = weights[LM_HEAD_KEY]  # Shape: [Vocab_Size, Hidden_Dim]
                        print(f"Loaded {LM_HEAD_KEY}, Shape: {lm_head_weight.shape}")
                    except Exception as e:
                        print(f"Failed to load weight: {e}")
                        return

                    h_states = value.to(DEVICE).to(lm_head_weight.dtype)

                    # 1. 维度处理
                    # 如果是 [Layers, Seq, Hidden]，取最后一层
                    if h_states.dim() == 3 and h_states.shape[0] > 10:
                        h_states = h_states[-1]  # 取最后一层

                    # 2. 矩阵乘法模拟 LM Head: logits = hidden @ weight.T
                    # 此时 h_states 可能为 [Seq, Hidden] 或 [Batch, Seq, Hidden]
                    with torch.no_grad():
                        logits = torch.matmul(h_states, lm_head_weight.t())
                        token_ids = torch.argmax(logits, dim=-1)  # 获取最高概率的 ID

                    # 3. 解码
                    # 扁平化处理，确保 decode 能处理
                    if token_ids.dim() > 1:
                        # 如果有 batch 或多余维度，取第一个序列
                        token_ids_to_decode = token_ids.view(-1)
                    else:
                        token_ids_to_decode = token_ids

                    decoded_text = tokenizer.decode(token_ids_to_decode, skip_special_tokens=True)
                    print(f"len(decoded_text)={len(decoded_text)}")

                    print(f"  - Value Range: [{h_states.min().item():.2f}, {h_states.max().item():.2f}]")
                    print(f"  - Decoded Text Output:")
                    print(f"    >>> {decoded_text}")

                # 如果第一维 > 1，说明存了多层
                if value.shape[0] > 1:
                    print(f"  - Layers found: {value.shape[0]} layers stacked in this key.")
                    # 分别看每一层的最大值，找出“罪魁祸首”
                    for i in range(value.shape[0]):
                        layer_max = v_float[i].max().item()
                        print(f"    -> Layer {i} Max: {layer_max:.2f}")

            print("-" * 40)
        else:
            print(f"Key: {key:<18} | Type: {type(value)}")


# 替换为你具体的路径
target_file = ""
inspect_single_ckpt(target_file)
