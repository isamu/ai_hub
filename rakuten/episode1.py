import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "Rakuten/RakutenAI-7B-chat"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
model.eval()

requests = [
    "「馬が合う」はどう言う意味ですか",
    "How to make an authentic Spanish Omelette?",
]

system_message = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {user_input} ASSISTANT:"

for req in requests:
    input_req = system_message.format(user_input=req)
    input_ids = tokenizer.encode(input_req, return_tensors="pt").to(device=model.device)
    tokens = model.generate(
        input_ids,
        max_new_tokens=1024,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    out = tokenizer.decode(tokens[0][len(input_ids[0]):], skip_special_tokens=True)
    print("USER:\n" + req)
    print("ASSISTANT:\n" + out)
    print()
    print()
