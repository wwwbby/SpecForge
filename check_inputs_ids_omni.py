import torch
from transformers import AutoTokenizer
import json

# ================= 配置区 =================
MODEL_PATH = ""
# 随便找一个让你怀疑的 ckpt
CKPT_PATH = ""
TOKENIZER_PATH = ""
MAX_SEQ_LEN = 2048


def parse_input_ids(input_ids):
    # 从tokenizer_config.json加载特殊token映射
    with open(TOKENIZER_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 构建token到ID的映射
    token_to_id = {}
    for id_str, token_info in config['added_tokens_decoder'].items():
        token_to_id[token_info['content']] = int(id_str)

    # 获取需要的特殊token ID
    endoftext_id = token_to_id['<|endoftext|>']
    im_start_id = token_to_id['<|im_start|>']
    im_end_id = token_to_id['<|im_end|>']
    audio_start_id = token_to_id['<|audio_start|>']
    audio_end_id = token_to_id['<|audio_end|>']
    audio_pad_id = token_to_id['<|audio_pad|>']
    think_start_id = token_to_id['<think>']
    think_end_id = token_to_id['</think>']

    # 统计特殊token个数
    endoftext_count = 0
    audio_pad_count = 0

    # 解析三个部分的起止下标
    pad_start = 0
    pad_end = -1
    user_start = -1
    user_end = -1
    model_start = -1
    model_end = -1
    think_start = -1
    think_end = -1

    # 第一部分：pad（<|endoftext|>）
    i = 0
    while i < len(input_ids) and input_ids[i] == endoftext_id:
        endoftext_count += 1
        i += 1
    pad_end = i - 1

    # 第二部分：用户提问（<|im_start|>...<|im_end|>）
    if i < len(input_ids) and input_ids[i] == im_start_id:
        user_start = i
        i += 1
        # 寻找对应的<|im_end|>
        while i < len(input_ids) and input_ids[i] != im_end_id:
            # 统计<|audio_pad|>的个数
            if input_ids[i] == audio_pad_id:
                audio_pad_count += 1
            i += 1
        if i < len(input_ids) and input_ids[i] == im_end_id:
            user_end = i
            i += 1
    while i < len(input_ids) and input_ids[i] != im_start_id:
        i += 1

    # 第三部分：模型回答（<|im_start|>...<|im_end|>）
    if i < len(input_ids) and input_ids[i] == im_start_id:
        model_start = i
        i += 1
        # 寻找对应的<think>
        while i < len(input_ids) and input_ids[i] != think_start_id:
            i += 1
        if i < len(input_ids) and input_ids[i] == think_start_id:
            think_start = i
        # 寻找对应的</think>
        while i < len(input_ids) and input_ids[i] != think_end_id:
            i += 1
        if i < len(input_ids) and input_ids[i] == think_end_id:
            think_end = i
        # 寻找对应的<|im_end|>
        while i < len(input_ids) and input_ids[i] != im_end_id:
            i += 1
        if i < len(input_ids) and input_ids[i] == im_end_id:
            model_end = i

    return {
        "pad": {"start": pad_start, "end": pad_end},
        "user": {"start": user_start, "end": user_end},
        "model": {"start": model_start, "end": model_end,},
        "think": {"start": think_start, "end": think_end},
        "end": {"start": model_end + 1, "end": len(input_ids) - 1},
        "counts": {
            "<|endoftext|>": endoftext_count,
            "<|audio_pad|": audio_pad_count
        }
    }


def print_part_text(input_ids, parse_result, tokenizer, name):
    st = parse_result[name]["start"]
    ed = parse_result[name]["end"]
    print("\n" + "=" * 80)
    print(f"name: {name}")
    print(f"📏 Token 序列长度: {ed-st}")
    print(f"📏 Token 下标起止: [{st}, {ed}]")
    print("=" * 80)
    print(tokenizer.decode(input_ids[st:ed+1], skip_special_tokens=False))


def fast_debug_ckpt():
    # 1. 直接从模型目录加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # 2. 加载 ckpt 数据
    ckpt = torch.load(CKPT_PATH, map_location="cpu")

    # 获取 input_ids (注意处理维度)
    input_ids = ckpt["input_ids"]
    if input_ids.ndim > 1:
        input_ids = input_ids[0]
    parse_result = parse_input_ids(input_ids)

    ## 3. 还原全文
    full_text = tokenizer.decode(input_ids, skip_special_tokens=False)

    print_part_text(input_ids, parse_result, tokenizer, "user")
    print_part_text(input_ids, parse_result, tokenizer, "model")
    print_part_text(input_ids, parse_result, tokenizer, "think")
    print_part_text(input_ids, parse_result, tokenizer, "end")

    # 4. 检查关键标签
    print("\n【关键特征检查】:")
    target_loss_mask = torch.zeros_like(ckpt["loss_mask"])
    target_loss_mask[parse_result["model"]["start"] + 3:parse_result["model"]["end"]] = 1
    if ckpt["loss_mask"].equal(target_loss_mask):
        print("loss_mask验证通过")
    else:
        for i in range(MAX_SEQ_LEN):
            if(target_loss_mask[i] != ckpt["loss_mask"][i]):
                print(f"loss_mask error in loss_mask[{i}], which token is {tokenizer.decode(input_ids[i], skip_special_tokens=False)}, expect {target_loss_mask[i]}, got ", ckpt["loss_mask"][i])
    print("- 左pad长度：", parse_result["counts"]["<|endoftext|>"])
    print(f"- audio长度:", parse_result["counts"]["<|audio_pad|"])
    print(f"- 是否包含 <think> 标签: {'✅ 是' if '<think>' in full_text else '❌ 否'}")
    print(f"- 是否包含 <|im_start|>system: {'✅ 是' if '<|im_start|>system' in full_text else '❌ 否'}")


if __name__ == "__main__":
    fast_debug_ckpt()
