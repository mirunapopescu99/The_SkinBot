import os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE    = "mistralai/Mistral-7B-Instruct-v0.2"
ADAPTER = "unsloth_finetuned_adapter"
OUT     = "merged_skinbot_fp16"
HF_TOKEN = os.environ.get("HUGGINGFACE_HUB_TOKEN")

os.makedirs(OUT, exist_ok=True)

tok = AutoTokenizer.from_pretrained(BASE, use_fast=True, token=HF_TOKEN)
base = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.float16, device_map="cpu", token=HF_TOKEN
)

model = PeftModel.from_pretrained(base, ADAPTER)
merged = model.merge_and_unload()      # bake LoRA into base

# save a clean HF folder (safetensors)
merged.save_pretrained(OUT, safe_serialization=True)
tok.save_pretrained(OUT)
print("✅ merged to", OUT)
