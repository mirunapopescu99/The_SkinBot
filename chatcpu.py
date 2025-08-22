# C:\unsloth_project\chat_cpu.py
import gradio as gr
from llama_cpp import Llama
import os, sys

GGUF = r"C:\unsloth_project\skinbot.Q4_K_M.gguf"  # replace after conversion
SYSTEM = "You are a helpful, concise skincare assistant. Answer clearly."

def ensure_model():
    if not os.path.exists(GGUF):
        return None
    llm = Llama(
        model_path=GGUF,
        n_ctx=4096,
        n_threads=None,
        n_batch=512,
        verbose=False,
    )
    return llm

LLM = ensure_model()

def format_prompt(history, user):
    text = f"<|user|>\n{SYSTEM}\n<|assistant|>\nOkay.\n"
    for u, a in history:
        text += f"<|user|>\n{u}\n<|assistant|>\n{a}\n"
    text += f"<|user|>\n{user}\n<|assistant|>\n"
    return text

def stream_reply(user, history):
    if LLM is None:
        yield "Model not found. Put your GGUF at C:\\unsloth_project\\skinbot.Q4_K_M.gguf and restart."
        return
    prompt = format_prompt(history or [], user)
    out = LLM(
        prompt,
        max_tokens=512,
        temperature=0.7,
        top_p=0.9,
        stop=["<|user|>", "<|assistant|>"],
        stream=True,
    )
    partial = ""
    for tok in out:
        piece = tok["choices"][0]["text"]
        partial += piece
        yield partial

with gr.Blocks(title="Unsloth Chatbot — CPU") as demo:
    gr.Markdown("## 🐆 Unsloth Chatbot — CPU (GGUF)\nLoad your quantized model and chat locally.")
    chat = gr.Chatbot(height=400)
    msg = gr.Textbox(placeholder="Ask me about skincare…")
    clear = gr.Button("Clear")

    def add_user(m,h): return "", (h or []) + [[m,""]]
    def add_bot(h):
        user = h[-1][0]
        gen = stream_reply(user, h[:-1])
        final=""
        for chunk in gen:
            final=chunk
            chat.update(h[:-1] + [[user, final]])
        return h[:-1] + [[user, final]]

    msg.submit(add_user, [msg, chat], [msg, chat]).then(add_bot, chat, chat)
    clear.click(lambda: None, None, chat)

demo.launch()
