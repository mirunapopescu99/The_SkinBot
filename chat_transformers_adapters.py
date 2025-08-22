# === SkinBot chat — Mistral-7B-Instruct v0.2 + LoRA adapter (Correct chat template) ===
import os
import re
import threading
import requests
import torch
import gradio as gr

from typing import List
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextIteratorStreamer,
    StoppingCriteria,
    StoppingCriteriaList,
)
from peft import PeftModel

# -------------------- Config --------------------
MODEL_BASE = "mistralai/Mistral-7B-Instruct-v0.2"
ADAPTER    = "unsloth_finetuned_adapter"   # your trained LoRA
HF_TOKEN   = os.environ.get("HUGGINGFACE_HUB_TOKEN")

# Optional: web price lookup (SerpAPI). If not set, browsing is skipped gracefully.
SERPAPI_KEY = os.environ.get("a7a52d2c02b415500b730609fb93ba64456c9f688887d88d9db5c793cf3c4aca")

SYSTEM_MESSAGE = (
    "You are SkinBot, a cautious, evidence-based skincare assistant. Build safe, minimal routines that "
    "solve the user’s concerns with the fewest products and lowest irritation risk.\n\n"
    "BOUNDARIES\n"
    "- Do NOT diagnose disease or suggest prescription dosing. If condition may be medical (infected lesions, "
    "cysts with fever, shingles, eye involvement, rapidly spreading rash, severe pain, non-healing wounds, "
    "sudden pigment change or ABCDE mole changes), recommend in-person dermatology.\n"
    "- Avoid pregnancy-unsafe advice (topical/oral retinoids, high-dose salicylic peels, hydroquinone unless supervised). "
    "For teens, keep actives gentle and avoid aggressive exfoliation.\n"
    "- Acknowledge allergies/sensitivities (benzoyl peroxide, fragrance, lanolin, essential oils, formaldehyde releasers).\n\n"
    "INFORMATION TO GATHER (only if missing and relevant)\n"
    "- Skin type (oily/combination/normal/dry), sensitivity/rosacea, Fitzpatrick I–VI if sunscreen nuance matters.\n"
    "- Primary concerns (acne type, PIH/PIE, melasma, texture, dryness, redness), current routine/actives, frequency, tolerance history.\n"
    "- Medications (isotretinoin, tretinoin, antibiotics), pregnancy/breastfeeding, shaving habits, budget/region.\n\n"
    "BUDGET AWARENESS\n"
    "- Ask (or infer) budget. Tiers: Low <£15; Mid £15–40; Premium £40+. Default to low-cost if unspecified.\n"
    "- Always include at least one low-cost product archetype.\n\n"
    "OUTPUT FORMAT\n"
    "1) Snapshot (3 bullets)\n"
    "2) AM Routine (numbered)\n"
    "3) PM Routine (numbered; cadence for retinoids/acids)\n"
    "4) Use / Avoid (short bullets + reasons)\n"
    "5) Ramp-up & Patch Test (exact schedule)\n"
    "6) Troubleshooting (what to do if dryness/stinging/new breakouts)\n"
    "7) Red Flags (when to stop and see a clinician)\n\n"
    "FORMULATION HINTS\n"
    "- Gentle low-pH cleanser; barrier moisturizer (ceramides/FA/cholesterol); sunscreen ≥SPF30 (≥50 if high UV/peels/retinoids).\n"
    "- Typical OTC ranges: Niacinamide 2–5%; Azelaic 10–15%; Salicylic 0.5–2%; Glycolic 4–8% (weekly→2–3×/wk); "
    "Lactic 5–10%; BP 2.5–5%; Adapalene 0.1% (OTC some regions).\n"
    "- Layering: water-based → serums → creams → occlusives; separate strong acids and retinoids by nights."
)

# -------------------- Perf toggles --------------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
os.environ.setdefault("PYTORCH_SDP_KERNEL", "mem_efficient")

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# -------------------- Optional browsing helpers --------------------
def serpapi_shopping(query: str, max_items: int = 5, gl="gb", hl="en"):
    if not SERPAPI_KEY:
        return []
    try:
        r = requests.get(
            "https://serpapi.com/search.json",
            params={
                "engine": "google_shopping",
                "q": query,
                "api_key": SERPAPI_KEY,
                "num": max_items,
                "hl": hl,
                "gl": gl,
            },
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        items = []
        for it in data.get("shopping_results", [])[:max_items]:
            price = it.get("extracted_price") or it.get("price")
            items.append({
                "title": it.get("title", "")[:120],
                "price": price,
                "source": it.get("source"),
            })
        return items
    except Exception as e:
        print("SerpAPI error:", e)
        return []

def build_context_from_web(user_msg: str):
    ask = user_msg.lower()
    intent = any(k in ask for k in ["routine", "recommend", "product", "budget", "cheap", "under", "affordable", "price"])
    if not intent:
        return ""
    queries = [
        "salicylic acid cleanser 2% price",
        "ceramide moisturizer drugstore price",
        "broad spectrum sunscreen spf 50 price",
    ]
    results = []
    for q in queries:
        results.extend(serpapi_shopping(q, max_items=3, gl="gb", hl="en"))
    if not results:
        return ""
    lines = ["CONTEXT (live-price candidates; approx, region-dependent):"]
    for r in results[:8]:
        price_str = (f"£{r['price']:.2f}" if isinstance(r["price"], (int, float))
                     else (str(r["price"]) if r["price"] else "n/a"))
        lines.append(f"- {r['title']} — {price_str} — {r.get('source','?')}")
    return "\n".join(lines)

# -------------------- Load model --------------------
print("=== SkinBot chat (Mistral default chat template) ===")
tok = AutoTokenizer.from_pretrained(MODEL_BASE, use_fast=True, token=HF_TOKEN)
tok.padding_side = "right"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

base = AutoModelForCausalLM.from_pretrained(
    MODEL_BASE,
    quantization_config=bnb,
    torch_dtype=torch.float16,
    device_map="auto",
    token=HF_TOKEN,
)
model = PeftModel.from_pretrained(base, ADAPTER)

# Stop on start of a new user turn tag for Mistral’s template (“[/INST]” closes; next turn starts with “[INST]”)
class StopOnStrings(StoppingCriteria):
    def __init__(self, tok, stop_strings: List[str]):
        super().__init__()
        self.stop_ids_list = [tok.encode(s, add_special_tokens=False) for s in stop_strings]
        self.stop_ids_list = [torch.tensor(s, device="cuda" if torch.cuda.is_available() else "cpu") for s in self.stop_ids_list]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for s in self.stop_ids_list:
            if input_ids.shape[1] >= s.shape[0]:
                if torch.equal(input_ids[0, -s.shape[0]:], s):
                    return True
        return False

stopper = StopOnStrings(tok, stop_strings=["[INST]"])

_first_debug_print = True

def chat_fn(message, history):
    global _first_debug_print

    # Optional: live-priced context
    context_block = build_context_from_web(message)

    # Build messages for apply_chat_template with Mistral’s default template
    messages = [{"role": "system", "content": SYSTEM_MESSAGE}]
    if context_block:
        messages.append({"role": "system", "content": context_block})
    for u, a in history:
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": message})

    # Use the model's DEFAULT chat template (do NOT override tok.chat_template)
    prompt = tok.apply_chat_template(
        messages,
        add_generation_prompt=True,   # opens the assistant turn
        tokenize=False,
    )

    if _first_debug_print:
        print("DEBUG prompt (first 400 chars):\n", prompt[:400], "\n---")
        _first_debug_print = False

    inputs = tok(prompt, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tok, skip_special_tokens=True)

    gen_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=400,
        do_sample=True,
        temperature=0.3,         # steadier outputs
        top_p=0.9,
        repetition_penalty=1.1,
        eos_token_id=tok.eos_token_id,
        stopping_criteria=StoppingCriteriaList([stopper]),
    )

    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    partial = ""
    for token_text in streamer:
        partial += token_text
        # In case the model tries to start a new user turn, trim
        if "[INST]" in partial:
            partial = partial.split("[INST]")[0].rstrip()
        yield partial

with gr.Blocks(title="SkinBot — Transformers (LoRA adapter on Mistral-7B-Instruct)") as demo:
    gr.Markdown("# SkinBot — Transformers (LoRA adapter on Mistral-7B-Instruct)")
    gr.ChatInterface(chat_fn, type="messages")

if __name__ == "__main__":
    # If 7860 is already in use, change the port here.
    demo.launch(server_name="0.0.0.0", server_port=7861)


