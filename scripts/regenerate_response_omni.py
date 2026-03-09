import argparse
import json
import random
import base64
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

from openai import OpenAI
from tqdm import tqdm


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Re-generate training data for Qwen3-Omni using SGLang"
    )

    # 模型相关参数
    model_group = parser.add_argument_group("model")
    model_group.add_argument("--model", type=str, required=True, help="模型名称或路径")
    model_group.add_argument(
        "--is-reasoning-model",
        action="store_true",
        help="是否为 Thinking/Reasoning 模型（提取思维链）",
    )

    # 采样参数
    sampling_params_group = parser.add_argument_group("sampling parameters")
    sampling_params_group.add_argument("--temperature", type=float, default=0.7)
    sampling_params_group.add_argument("--top-p", type=float, default=None)
    sampling_params_group.add_argument("--max-tokens", type=int, default=4096)

    # 优化参数
    optimization_group = parser.add_argument_group("optimization")
    optimization_group.add_argument(
        "--concurrency",
        type=int,
        default=32,
        help="每个服务器地址的并发请求数",
    )

    # 数据相关参数
    data_group = parser.add_argument_group("data")
    data_group.add_argument("--input-file-path", type=str, required=True, help="输入 JSONL 路径")
    data_group.add_argument("--output-file-path", type=str, required=True, help="输出 JSONL 路径")
    data_group.add_argument("--audio-root", type=str, required=True, help="音频文件夹根目录")
    data_group.add_argument("--num-samples", type=int, default=None, help="处理样本数量上限")

    # SGLang 服务器地址
    server_group = parser.add_argument_group("sglang server")
    server_group.add_argument(
        "--server-address",
        type=str,
        nargs="+",
        default=["127.0.0.1:8003"],
        help="SGLang 服务器地址列表 (e.g. 127.0.0.1:8003 127.0.0.1:8004)",
    )
    return parser.parse_args()


def encode_audio_to_uri(audio_path: str) -> str:
    """将音频文件转为带有 Data URI 前缀的 Base64 字符串"""
    if not os.path.exists(audio_path):
        return None
    try:
        with open(audio_path, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode("utf-8")

        ext = audio_path.split(".")[-1].lower()
        mime_type = "audio/wav" if ext != "mp3" else "audio/mpeg"
        return f"data:{mime_type};base64,{audio_base64}"
    except Exception as e:
        print(f"Error encoding {audio_path}: {e}")
        return None


def build_query_kwargs(args, messages, max_tokens=None):
    """构造 OpenAI 兼容的请求参数"""
    return dict(
        model=args.model,
        messages=messages,
        max_tokens=max_tokens or args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stream=False,
    )


def call_sglang(args, server_address: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """单条数据处理核心逻辑"""
    client = OpenAI(base_url=f"http://{server_address}/v1", api_key="EMPTY")

    # 1. 提取输入字段
    input_text = data.get("text", "")
    audio_file = data.get("audio", None)
    audio_full_path = os.path.join(args.audio_root, audio_file) if audio_file else None

    # 2. 构造多模态 Content
    content_list = []
    if audio_full_path:
        audio_uri = encode_audio_to_uri(audio_full_path)
        if audio_uri:
            content_list.append({
                "type": "audio_url",
                "audio_url": {"url": audio_uri}
            })

    content_list.append({"type": "text", "text": input_text})
    messages = [{"role": "user", "content": content_list}]

    # 3. 请求模型
    try:
        query_kwargs = build_query_kwargs(args, messages)
        resp = client.chat.completions.create(**query_kwargs)
        result_text = resp.choices[0].message.content

        # 4. 构造符合示例要求的输出结构
        res_json = {
            "id": data.get("id", ""),
            "conversations": [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": audio_file},
                        {"type": "text", "text": input_text}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": result_text}]
                }
            ],
            "status": "success"
        }

        # 处理 Thinking 内容
        if args.is_reasoning_model:
            reasoning = getattr(resp.choices[0].message, 'reasoning_content', None)
            if reasoning:
                res_json["conversations"][1]["thinking"] = reasoning

        return res_json

    except Exception as e:
        data["status"] = "error"
        data["error"] = str(e)
        return data


def main():
    args = parse_arguments()

    # 预检
    if not os.path.exists(args.input_file_path):
        raise FileNotFoundError(f"输入文件不存在: {args.input_file_path}")

    # 获取服务器列表并测试
    valid_servers = args.server_address
    print(f"--- 启动配置 ---")
    print(f"模型: {args.model}")
    print(f"音频目录: {args.audio_root}")
    print(f"服务器: {valid_servers}")
    print("-" * 20)

    # 读取数据
    with open(args.input_file_path, "r", encoding="utf-8") as f:
        all_data = [json.loads(line) for line in f]

    if args.num_samples:
        all_data = all_data[:args.num_samples]

    error_file_path = args.output_file_path.replace(".jsonl", "_error.jsonl")

    success_count = 0
    error_count = 0

    # 使用线程池执行
    with open(args.output_file_path, "w", encoding="utf-8") as out_f, \
        open(error_file_path, "w", encoding="utf-8") as err_f:

        # 总并发 = 服务器数量 * 单服务器并发
        max_workers = args.concurrency * len(valid_servers)
        executor = ThreadPoolExecutor(max_workers=max_workers)

        futures = []
        for i, item in enumerate(all_data):
            # 简单的负载均衡：轮询服务器
            target_server = valid_servers[i % len(valid_servers)]
            futures.append(executor.submit(call_sglang, args, target_server, item))

        for future in tqdm(futures, desc="Regenerating Omni Data"):
            res = future.result()
            if res.get("status") == "success":
                # 移除状态标记后写入
                res.pop("status", None)
                out_f.write(json.dumps(res, ensure_ascii=False) + "\n")
                success_count += 1
            else:
                err_f.write(json.dumps(res, ensure_ascii=False) + "\n")
                error_count += 1

    print(f"\n处理完成！")
    print(f"成功: {success_count} 条")
    print(f"失败: {error_count} 条 (详情见 {error_file_path})")


if __name__ == "__main__":
    main()
