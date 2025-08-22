# infer_adapter.py
import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig
from peft import PeftModel

BASE    = "mistralai/Mistral-7B-Instruct-v0.2"
ADAPTER = "unsloth_finetuned_adapter"  # from training
HF_TOKEN = os.environ.get("HUGGINGFACE_HUB_TOKEN")

# 4-bit load to fit nicely on a single GPU
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

tok = AutoTokenizer.from_pretrained(BASE, use_fast=True, token=HF_TOKEN)
tok.padding_side = "right"
tok.pad_token = tok.pad_token or tok.eos_token

base = AutoModelForCausalLM.from_pretrained(
    BASE,
    quantization_config=bnb,
    torch_dtype=torch.float16,
    device_map="auto",
    token=HF_TOKEN,
)

model = PeftModel.from_pretrained(base, ADAPTER)

def chat(prompt, max_new_tokens=256, temperature=0.6):
    text = f"<s>[INST] {prompt} [/INST]"
    inputs = tok(text, return_tensors="pt").to(model.device)
    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        top_p=0.9,
        repetition_penalty=1.1,
    )
    with torch.inference_mode():
        out = model.generate(**inputs, generation_config=gen_cfg)
    return tok.decode(out[0], skip_special_tokens=True)

if __name__ == "__main__":
    print(chat("I have combination, sensitive skin with forehead texture. Give a simple AM/PM routine."))
