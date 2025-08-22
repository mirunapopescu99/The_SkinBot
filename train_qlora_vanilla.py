# train_qlora_vanilla.py — QLoRA on Windows without Unsloth
# Uses Hugging Face gated model with token login
# Works with: transformers==4.42.4, trl==0.8.6, accelerate==0.33.0,
#             peft==0.11.1, datasets==2.20.0, bitsandbytes==0.43.1

import os, torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from huggingface_hub import login

# --------------------
# Config
# --------------------
DATA = "data"                          # formatted.arrow dir from prepare_data.py
OUT  = "unsloth_finetuned_adapter"     # save folder for adapter
BASE = "mistralai/Mistral-7B-Instruct-v0.2"  # gated model; requires HF token

# Token (safer: set env var HUGGINGFACE_HUB_TOKEN)
HF_TOKEN = os.environ.get("HUGGINGFACE_HUB_TOKEN", "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")

def _login_hf():
    if HF_TOKEN:
        login(token=HF_TOKEN)
    else:
        raise RuntimeError(
            "No Hugging Face token provided. "
            "Set HUGGINGFACE_HUB_TOKEN env var or paste it into HF_TOKEN."
        )

def main():
    # --- Authenticate with Hugging Face ---
    _login_hf()

    # --- Load dataset ---
    if not os.path.exists(DATA):
        raise FileNotFoundError("Run prepare_data.py first to create ./data")
    ds = load_from_disk(DATA)  # expects a 'text' column

    # --- Try 4-bit (bitsandbytes); fall back to fp16 if not installed ---
    use_4bit = False
    bnb_config = None
    try:
        from transformers import BitsAndBytesConfig
        import importlib.metadata as im
        im.version("bitsandbytes")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        use_4bit = True
        print("✅ Using 4-bit QLoRA with bitsandbytes")
    except Exception as e:
        print("⚠️ bitsandbytes not available; falling back to fp16:", e)

    # --- Tokenizer ---
    tok = AutoTokenizer.from_pretrained(BASE, use_fast=True, token=HF_TOKEN)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # --- Base model ---
    if use_4bit:
        model = AutoModelForCausalLM.from_pretrained(
            BASE,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map="auto",
            token=HF_TOKEN,
        )
        model = prepare_model_for_kbit_training(model)
        optim_name = "adamw_8bit"
    else:
        model = AutoModelForCausalLM.from_pretrained(
            BASE,
            torch_dtype=torch.float16,
            device_map="auto",
            token=HF_TOKEN,
        )
        optim_name = "adamw_torch_fused"

    # --- LoRA ---
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    # --- Training ---
    args = TrainingArguments(
        output_dir="outputs",
        per_device_train_batch_size=2 if use_4bit else 1,
        gradient_accumulation_steps=4 if use_4bit else 8,
        max_steps=60,   # increase for real training
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        optim=optim_name,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        train_dataset=ds,
        dataset_text_field="text",
        max_seq_length=2048,
        packing=False,
        args=args,
    )

    trainer.train()

    # --- Save ---
    os.makedirs(OUT, exist_ok=True)
    trainer.save_model(OUT)
    tok.save_pretrained(OUT)
    print("✅ Training done ->", OUT)

if __name__ == "__main__":
    # Enable TF32 for faster CUDA math on 4090
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    main()
