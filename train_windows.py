# train_windows.py  — Unsloth QLoRA fine-tune on Windows (relative paths, no Triton)
# -------------------------------------------------------
# - Uses PyTorch SDPA (and xFormers if present), explicitly disables Triton paths.
# - Works with bitsandbytes 4-bit; if bnb fails on your system, switch optimizer below.
# - Uses dataset saved by prepare_data.py under ./data (HuggingFace Arrow directory).
# - Keeps dataset_num_proc=1 for Windows stability.

import os

# --- Attention / backend knobs (safe on Windows) ---
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"      # harmless on Windows
os.environ["XFORMERS_FORCE_DISABLE_TRITON"] = "1"    # ensure no Triton path
os.environ["FLASH_ATTENTION_FORCE_DISABLE"] = "1"    # force SDPA if needed

from datasets import load_from_disk
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments

# --- Relative paths ---
DATA = "data"                        # folder containing 'formatted.arrow' from prepare_data.py
OUT  = "unsloth_finetuned_adapter"   # where the LoRA adapter will be saved
BASE = "unsloth/mistral-7b-instruct-v0.2"

def main():
    # 1) Load dataset
    if not os.path.exists(DATA):
        raise FileNotFoundError(f"Missing dataset folder: {DATA} — run prepare_data.py first.")
    ds = load_from_disk(DATA)        # loads data/formatted.arrow

    # 2) Load base model in 4-bit (requires bitsandbytes)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,            # if bnb fails on your box, set to False and reduce batch size
    )

    # Optional: enforce SDPA attention implementation (non-Triton)
    try:
        model.config.attn_implementation = "sdpa"
    except Exception:
        pass

    # 3) Attach LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj","k_proj","v_proj","o_proj",
            "gate_proj","up_proj","down_proj"
        ],
        lora_alpha=16,
        lora_dropout=0.05,
        use_gradient_checkpointing=True,  # saves VRAM
    )

    # 4) Training arguments
    args = TrainingArguments(
        output_dir="outputs",
        per_device_train_batch_size=2,    # drop to 1 if you see CUDA OOM
        gradient_accumulation_steps=4,
        max_steps=60,                     # increase for real training
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        # If bitsandbytes is problematic, change to: optim="adamw_torch_fused"
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="none",
    )

    # 5) Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        dataset_text_field="text",
        max_seq_length=2048,
        packing=False,
        args=args,
        dataset_num_proc=1,               # Windows: keep single process dataloader
    )

    # 6) Go!
    trainer.train()

    # 7) Save adapter + tokenizer
    os.makedirs(OUT, exist_ok=True)
    trainer.save_model(OUT)
    tokenizer.save_pretrained(OUT)
    print(f"✅ Training done → {OUT}")

if __name__ == "__main__":
    main()
