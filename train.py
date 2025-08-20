# train.py
# Robust Unsloth trainer for Windows (RTX 4090) + cross-platform
# - Early CUDA checks with clear guidance
# - Windows: 4-bit OFF (adamw_torch). Linux/WSL: 4-bit ON if bitsandbytes is available
# - Dataset path resolved relative to this file
# - Saves adapter and merged models

from __future__ import annotations
import os
import sys
import platform
import importlib
from importlib import metadata
from pathlib import Path

# -------------------------
# 0) Environment diagnostics
# -------------------------
try:
    import torch
except Exception as e:
    raise SystemExit(
        "PyTorch is not installed in this environment.\n"
        "Install a CUDA build on Windows with:\n"
        "  pip install --index-url https://download.pytorch.org/whl/cu121 "
        "torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1\n"
    ) from e

torch_cuda_built = torch.backends.cuda.is_built()
torch_cuda_ver = getattr(torch.version, "cuda", None)
cuda_available = torch.cuda.is_available()

print(
    f"[torch] version={torch.__version__} | cuda_built={torch_cuda_built} | "
    f"torch.version.cuda={torch_cuda_ver} | cuda_available={cuda_available}"
)
if cuda_available:
    try:
        print(f"[torch] device0={torch.cuda.get_device_name(0)}")
    except Exception:
        pass

if platform.system() == "Windows":
    # On Windows native you MUST have cu121 wheels + modern NVIDIA driver.
    if not torch_cuda_built or torch_cuda_ver is None or not cuda_available:
        raise SystemExit(
            "\n❌ CUDA is not visible to this Python interpreter.\n"
            "Fix on Windows (RTX 4090):\n"
            "  1) Activate THIS env:  conda activate skincare\n"
            "  2) Reinstall CUDA wheels:\n"
            "       pip uninstall -y torch torchvision torchaudio\n"
            "       pip install --index-url https://download.pytorch.org/whl/cu121 "
            "torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1\n"
            "  3) Update NVIDIA Driver (Game Ready or Studio), then reboot.\n"
            "  4) Verify:\n"
            '       python -c "import torch; print(torch.cuda.is_available()); '
            'print(torch.version.cuda); print(torch.cuda.get_device_name(0))"\n'
        )

# Optional: warn about transformers >= 4.45 pulling torchao
try:
    tf_ver = metadata.version("transformers")
    print(f"[transformers] {tf_ver}")
    # Not enforcing a pin here, but warn if too new.
    if tuple(int(x) for x in tf_ver.split(".")[:2]) >= (4, 45):
        print(
            "⚠️  transformers >= 4.45 may import torchao on import and crash on Windows. "
            "If you see a torchao error, pin: pip install transformers==4.44.2"
        )
except metadata.PackageNotFoundError:
    print("[transformers] not installed? (It should be in requirements.txt)")

# -------------------------
# 1) Imports (now that CUDA is good)
# -------------------------
try:
    from datasets import load_dataset
except ModuleNotFoundError as e:
    raise SystemExit(
        "⚠️ The 'datasets' library is not installed in this interpreter.\n"
        "Activate your venv and run:  pip install datasets"
    ) from e

try:
    from unsloth import FastLanguageModel, is_bfloat16_supported
except Exception as e:
    # Common misses: unsloth_zoo not installed
    raise SystemExit(
        f"Failed to import Unsloth: {e}\n"
        "Fixes:\n"
        "  pip install -U unsloth unsloth_zoo\n"
        "If you still see a torchao error, pin transformers: pip install transformers==4.44.2\n"
    )

from transformers import TrainingArguments
from trl import SFTTrainer

# -------------------------
# 2) Config
# -------------------------
HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "data" / "skincare.jsonl"
MODEL_NAME = "unsloth/mistral-7b-instruct-v0.2"
MAX_SEQ_LEN = 2048

IS_WINDOWS = platform.system() == "Windows"
HAS_BNB = importlib.util.find_spec("bitsandbytes") is not None
# Windows native: keep 4-bit OFF (bnb unreliable). Linux/WSL: enable if bnb found.
USE_4BIT = (not IS_WINDOWS) and HAS_BNB
OPTIM = "adamw_8bit" if (USE_4BIT and HAS_BNB) else "adamw_torch"

print(f"[config] platform={platform.system()} | bitsandbytes={HAS_BNB} | 4bit={USE_4BIT} | optim={OPTIM}")

# -------------------------
# 3) Load dataset
# -------------------------
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Could not find dataset at {DATA_PATH}\n"
        "Ensure you have data/skincare.jsonl next to train.py."
    )

raw = load_dataset("json", data_files=str(DATA_PATH), split="train")

def to_text(ex):
    # Your chat template
    return {
        "text": f"<|user|>\n{ex['instruction']}\n{ex['input']}\n<|assistant|>\n{ex['output']}"
    }

dataset = raw.map(to_text, remove_columns=[c for c in raw.column_names if c != "text"])

# -------------------------
# 4) Load base model + LoRA
# -------------------------
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    dtype=None,
    load_in_4bit=USE_4BIT,   # stays False on Windows native
)

# Tokenizer padding safety
if getattr(tokenizer, "pad_token", None) is None:
    tokenizer.pad_token = tokenizer.eos_token

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing=True,
)

# -------------------------
# 5) Train
# -------------------------
args = TrainingArguments(
    output_dir=str(HERE / "outputs"),
    per_device_train_batch_size=2,      # tweak down to 1 if VRAM is tight
    gradient_accumulation_steps=4,
    max_steps=60,                        # smoke test; raise for real runs
    learning_rate=2e-4,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    logging_steps=1,
    optim=OPTIM,                         # adamw_torch on Windows, adamw_8bit elsewhere if bnb present
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LEN,
    packing=False,
    args=args,
)

try:
    trainer.train()
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        raise SystemExit(
            "CUDA OOM. Try:\n"
            "  - per_device_train_batch_size=1\n"
            "  - gradient_accumulation_steps=8\n"
            "  - max_seq_length=1024\n"
        ) from e
    raise

# -------------------------
# 6) Save adapters + merged
# -------------------------
adapter_dir = HERE / "checkpoints" / "adapter"
merged_dir  = HERE / "checkpoints" / "merged"
adapter_dir.mkdir(parents=True, exist_ok=True)
merged_dir.mkdir(parents=True, exist_ok=True)

# Save LoRA adapter
trainer.model.save_pretrained(str(adapter_dir))
tokenizer.save_pretrained(str(adapter_dir))

# Merge LoRA -> full weights and save
merged = trainer.model.merge_and_unload()
merged.save_pretrained(str(merged_dir))
tokenizer.save_pretrained(str(merged_dir))

print("\n✅ Training complete.")
print(f"   • LoRA adapter: {adapter_dir}")
print(f"   • Merged model: {merged_dir}")
