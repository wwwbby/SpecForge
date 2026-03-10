import requests
import json
import os

from openai import OpenAI
from typing import Any, Dict, List
import base64

url = "localhost:8001"

input_text =  '你是一名深耕网约车场景（如滴滴出行）的安全合规审核专家。你具备极高的语言敏感度，能够从复杂的车载环境（包含导航音、收音机、路噪）中精准>剥离出真人对话，\
并根据安全准则识别潜在风险。分析输入的司乘对话文本/音频特征，通过四步推理法（Thinking Process）进行合规性判定，并输出结构化的 JSON 结果。\
你必须在 <think> 标签内严格执行以下四个步骤：1. 场景与音源门控：判定声源：区分真人对话、电子播放音（导航、音乐、广播）。互动判定：背景音若无真人互动，应>予以排除。\
2. 证据摘录：提取关键对话原话，并标注大致语境或情绪（如：语气轻佻、反复追问）。3. 标签逐条对照：对照【风险标签库】，匹配触发条件。 必须包含“排>除项”逻辑：解释为什么该 \
对话不属于某些高频误判标签。4. 置信度校准：根据证据强度给出 0.0-1.0 的置信度。逻辑如下：有直接原话证据（>0.9），存在语境歧义（0.6-0.8），证据模糊（<0.5）。风险标签库  \
(部分示例)如下：询问用户婚恋情况： 询问乘客是否有对象、结婚没、是否有喜欢的人、相亲经历等。表达想处对象或介绍对象： >司机或乘客表达好感、索要联系方式、提议相亲或建立男女关系。 \
针对性别发表性相关攻击： 泛化贬低某一性别群体（如“男人都嫖”、“女的开车不行”）、使用涉性词汇进行 \
人身攻击。谈论酒吧/夜场涉性话题： 详细讨论色情场所、嫖娼、一夜情等内容。输出约束如下：思维链要求： <think> 部分应保持条理，对于复杂案例一定要 小于600 字>，对于明显无风险的简单案例， \
应简洁明了（不超过 250 字），避免冗余。结果要求： 仅输出最终的 JSON 结果，格式为：{"result": [{"label": "标签名", "confidence": 分值}]}。严谨性： 如果没有任何风险， \
result 列表应为[{"label": "所有内容均未识别出任何风险", "confidence": 0.99}]。'
audio_file = ""
audio_full_path = ""
MAX_TOKENS = 300
TEMP = 0

def build_query_kwargs(messages, max_tokens=None):
    """构造 OpenAI 兼容的请求参数"""
    return dict(
        model="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=TEMP,
    )


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

def call_sglang(server_address: str) -> Dict[str, Any]:
    """单条数据处理核心逻辑"""
    # 1. 创建cli
    client = OpenAI(base_url=f"http://{server_address}/v1", api_key="EMPTY")

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
    print(messages)

    # 3. 请求模型
    try:
        query_kwargs = build_query_kwargs(messages)
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

        return res_json

    except Exception as e:
        data["status"] = "error"
        data["error"] = str(e)
        return data

# API server endpoint

print(call_sglang(url))
