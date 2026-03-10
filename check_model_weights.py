import torch
from safetensors.torch import load_file


MODEL_FILE_PATH = ""
weights = load_file(MODEL_FILE_PATH)


for k in weights.keys():
    if "visual" not in k and "audio_tower" not in k:
        if "lm_head" in k:
            print(f"{k}\n")
