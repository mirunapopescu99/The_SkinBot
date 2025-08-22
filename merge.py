# merge.py (relative paths)
import os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE    = "unsloth/mistral-7b-instruct-v0.2"
ADAPTER = "unsloth_finetuned_adapter"
MERGED  = "merged_skinbot"

def main():
    os.makedirs(MERGED, exist_ok=True)
    tok  = AutoTokenizer.from_pretrained(BASE, use_fast=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.float16, device_map="auto"
    )
    model  = PeftModel.from_pretrained(base, ADAPTER, device_map="auto")
    merged = model.merge_and_unload()
    merged.save_pretrained(MERGED, safe_serialization=True)
    tok.save_pretrained(MERGED)
    print("✅ Merged model at", MERGED)

if __name__ == "__main__":
    main()
