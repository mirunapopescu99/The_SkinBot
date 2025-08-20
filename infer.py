import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

# Choose either the merged model (full HF) or base + adapter (LoRA).
# Easiest: use the merged model.
MODEL_DIR = "checkpoints/merged"  # change to 'checkpoints/adapter' if you prefer adapters
USE_GPU = torch.cuda.is_available()

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float16 if USE_GPU else torch.float32,
    device_map="auto" if USE_GPU else {"": "cpu"},
)

prompt = "<|user|>\nHelp routine\nI have combination skin with uneven tone.\n<|assistant|>\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
with torch.no_grad():
    _ = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        streamer=streamer,
        eos_token_id=tokenizer.eos_token_id,
    )
