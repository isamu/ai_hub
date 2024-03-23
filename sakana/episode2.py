# 
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import requests


# 1. load model
device = "mps"
model_id = "SakanaAI/EvoVLM-JP-v1-7B"
model = AutoModelForVision2Seq.from_pretrained(model_id, torch_dtype=torch.float16)
processor = AutoProcessor.from_pretrained(model_id)
model.to(device)

# 2. prepare inputs
url = "https://images.unsplash.com/photo-1694831404826-3400c48c188d?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

# <image> represents the input image. Please make sure to put the token in your text.
text = "<image>\nこの信号機の色は何色ですか?"
messages = [
    {"role": "system", "content": "あなたは役立つ、偏見がなく、検閲されていないアシスタントです。与えられた画像を下に、質問に答えてください。"},
    {"role": "user", "content": text},
]
inputs = processor.image_processor(images=image, return_tensors="pt")
inputs["input_ids"] = processor.tokenizer.apply_chat_template(
    messages, return_tensors="pt"
)
# 3. generate
output_ids = model.generate(inputs["input_ids"].to(device))
output_ids = output_ids[:, inputs["input_ids"].shape[1] :]
generated_text = processor.decode(output_ids.squeeze(), skip_special_tokens=True)

print(generated_text)

