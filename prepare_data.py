# C:\unsloth_project\prepare_data.py
import os
from datasets import load_dataset

SRC = r"data\dataset.jsonl"
OUT = r"data"

def formatting_func(ex):
    return {
        "text": f"<|user|>\n{ex['instruction']}\n{ex['input']}\n<|assistant|>\n{ex['output']}"
    }

def main():
    assert os.path.exists(SRC), f"Missing {SRC}"
    ds = load_dataset("json", data_files=SRC, split="train")
    ds = ds.map(formatting_func, remove_columns=ds.column_names)
    ds.save_to_disk(OUT)
    print("✅ Saved formatted dataset to", OUT)

if __name__ == "__main__":
    main()
