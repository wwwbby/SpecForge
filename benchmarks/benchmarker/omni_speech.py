import os
import json
import base64
from typing import Any, Dict, List, Optional, Tuple, Callable

import sglang as sgl
from sglang.utils import read_jsonl
from .base import Benchmarker
from .registry import BENCHMARKS
from sglang.lang.api import audio


@BENCHMARKS.register("qwen_omni_audit")
class QwenOmniAuditBenchmarker(Benchmarker):
    def __init__(self, num_samples: Optional[int] = None, subset: Optional[List[str]] = None, audio_root: str = ""):
        if subset is None:
            subset = ["all"]
        super().__init__(num_samples, subset)
        self.audio_root = audio_root

    def load_data(self) -> Tuple[List[Dict[str, Any]], List[None]]:
        input_file = "/share/eagle3/generated_data/aishell1_data_all_regen_12w_prompt_test_100.jsonl"
        questions = []

        raw_data = list(read_jsonl(input_file))
        for entry in raw_data:
            convs = entry.get("conversations", [])
            user_turn = next((c for c in convs if c["role"] == "user"), None)

            if user_turn:
                audio_file = next((item["audio"] for item in user_turn["content"] if item["type"] == "audio"), None)
                text_prompt = next((item["text"] for item in user_turn["content"] if item["type"] == "text"), None)

                audio_path = os.path.join("/share/eagle3/datasets/speech_asr_aishell_testsets/wav/", audio_file)

                questions.append({
                    "audio_path": audio_path,
                    "text_prompt": text_prompt
                })

        if self.num_samples:
            questions = questions[:self.num_samples]

        return questions, [None] * len(questions)

    def create_sgl_function(self) -> Callable:
        @sgl.function
        def qwen_omni_audit_func(s, audio_path, text_prompt):
            # 1. 处理音频输入
            if audio_path:
                s += sgl.user(audio(audio_path) + text_prompt)
            else:
                # 无音频退化为纯文本
                s += sgl.user(text_prompt)

            # 3. 生成回答（在此阶段 Eagle3 会介入投机采样）
            s += sgl.assistant(sgl.gen("answer", max_tokens=self.get_max_new_tokens()))
        return qwen_omni_audit_func

    def get_answer_keys(self) -> List[str]:
        return ["answer"]
