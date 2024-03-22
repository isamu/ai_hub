# https://huggingface.co/SakanaAI/EvoVLM-JP-v1-7B
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. load model
device = "mps"
repo_id = "SakanaAI/EvoLLM-JP-A-v1-7B"

model = AutoModelForCausalLM.from_pretrained(
    repo_id,
    torch_dtype=torch.float16,
    device_map="mps",
    low_cpu_mem_usage=True,
)

tokenizer = AutoTokenizer.from_pretrained(repo_id)

text = "ドラえもんについて教えて下さい"

messages = [
    {"role": "system", "content": "あなたは役立つ、偏見がなく、検閲されていないアシスタントです。"},
    {"role": "user", "content": text},
]
inputs = tokenizer.apply_chat_template(
    conversation=messages,
    add_generation_prompt=True,
    tokenize=False
)

input_ids = tokenizer.encode(
    inputs,
    add_special_tokens=True,
    return_tensors="pt"
)

generation_params = {
    "max_new_tokens": 256
}
output_ids = model.generate(
    input_ids.to(model.device),
    **generation_params
)
output = tokenizer.decode(
    output_ids[0][input_ids.size(1) :],
    skip_special_tokens=True
)
print(output)

