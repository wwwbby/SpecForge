import torch
import torch_npu
from transformers import AutoProcessor, AutoTokenizer


input_ids = []

MODEL_PATH = ""
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

input_ids = torch.tensor(input_ids, device="npu")
print(tokenizer.decode(input_ids, skip_special_tokens=True))

processor = AutoProcessor.from_pretrained(MODEL_PATH)
raw_template = processor.tokenizer.chat_template

if raw_template is None:
    raw_template = getattr(processor, "chat_template", None)

print(raw_template)
