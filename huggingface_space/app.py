"""
CROM-IA V3.5b - Rosa Chat (HuggingFace Spaces Edition)
Motor de Inferencia Organica com RAG Injector DNA O(1)
"""

import gradio as gr
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
import multiprocessing
import time
import json
import re
import os

# ====================================================================
# 1. DOWNLOAD DO MODELO E CODEBOOK
# ====================================================================
print("[CROM-IA] Baixando modelo GGUF V3.5b...")
model_path = hf_hub_download(
    repo_id="CromIA/CROM-IA-V3.5-Qwen-1.5B-Organic",
    filename="Qwen2.5-1.5B-Instruct.Q4_K_M-v3.5b_117k.gguf"
)
codebook_path = hf_hub_download(
    repo_id="CromIA/CROM-IA-V3.5-Qwen-1.5B-Organic",
    filename="logs_and_codebooks/bugged_codebook_v3_recipes.json"
)

# ====================================================================
# 2. CARREGAR CODEBOOK DNA
# ====================================================================
with open(codebook_path, "r", encoding="utf-8") as f:
    codebook_data = json.load(f)
dna_entries = codebook_data.get("entries", {})
total_bytes = sum(e["bytes"] for e in dna_entries.values())
print(f"[CROM-IA] {len(dna_entries)} Ponteiros DNA ({total_bytes} bytes)")

# ====================================================================
# 3. INSTANCIAR LLM
# ====================================================================
cores = max(multiprocessing.cpu_count(), 2)
print(f"[CROM-IA] {cores} threads | n_batch=512 | n_ctx=4096")
llm = Llama(
    model_path=model_path,
    n_ctx=4096,
    n_threads=cores,
    n_threads_batch=cores,
    n_batch=512,
    use_mmap=True,
    verbose=False
)
print("[CROM-IA] Pronto!")

# ====================================================================
# 4. RAG INJECTOR DNA
# ====================================================================
def rag_inject(text):
    hits = 0
    saved = 0
    result = []
    i = 0
    while i < len(text):
        if text[i] == '@' and i + 1 < len(text) and text[i + 1] == '@':
            found = False
            for kl in [4, 3, 2]:
                end = i + 2 + kl
                if end <= len(text):
                    key = text[i + 2:end]
                    if key in dna_entries:
                        result.append(dna_entries[key]["text"])
                        saved += dna_entries[key]["bytes"]
                        hits += 1
                        i = end
                        found = True
                        break
            if not found:
                result.append(text[i])
                i += 1
        else:
            result.append(text[i])
            i += 1
    return "".join(result), hits, saved

# ====================================================================
# 5. FORMATADOR DE PROMPT
# ====================================================================
def safe_str(val):
    if isinstance(val, list):
        return " ".join(str(v) for v in val)
    if val is None:
        return ""
    return str(val)

def clean_metrics(text):
    text = safe_str(text)
    return re.sub(r'\n*---\n.*$', '', text, flags=re.DOTALL).strip()

def formatar_prompt(message, history):
    prompt = "Abaixo esta uma instrucao CROM-IA.\n\n"
    for entry in history:
        if isinstance(entry, dict):
            role = entry.get("role", "")
            content = clean_metrics(entry.get("content", ""))
            if role == "user":
                prompt += f"### Instruction:\n{content}\n\n### Input:\n\n\n"
            elif role == "assistant":
                prompt += f"### Response:\n{content}\n\n"
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
            user_msg = safe_str(entry[0])
            bot_msg = clean_metrics(entry[1])
            prompt += f"### Instruction:\n{user_msg}\n\n### Input:\n\n\n### Response:\n{bot_msg}\n\n"
    prompt += f"### Instruction:\n{message}\n\n### Input:\n\n\n### Response:\n"
    return prompt

# ====================================================================
# 6. GERADOR DE RESPOSTA
# ====================================================================
def generate(message, history, temperature, repeat_penalty, max_tokens):
    prompt = formatar_prompt(message, history)

    stream = llm(
        prompt,
        max_tokens=int(max_tokens),
        temperature=temperature,
        repeat_penalty=repeat_penalty,
        stop=["### Instruction:", "### Input:", "<|endoftext|>", "<|im_end|>", "</s>", "Abaixo esta", "#", "\n\n\n\n\n"],
        stream=True
    )

    resposta = ""
    t0 = time.time()
    toks = 0

    for chunk in stream:
        token = chunk["choices"][0]["text"]
        resposta += token
        toks += 1
        yield resposta

    dt = time.time() - t0
    tps = toks / dt if dt > 0 else 0

    resposta_rag, hits, saved = rag_inject(resposta)

    m = f"\n\n---\n**🧬 CROM-IA** ⚡ {tps:.1f} t/s · 📝 {toks} tokens · ⏱️ {dt:.1f}s · 🧵 {cores} threads"

    if hits > 0:
        m += f" · 🔬 RAG: {hits} hits ({saved}B expandidos)"
        final = resposta_rag
    else:
        final = resposta

    yield final + m

# ====================================================================
# 7. INTERFACE VISUAL
# ====================================================================
GITHUB = "https://github.com/MrJc01/crompressor-ia"
HF_MODEL = "https://huggingface.co/CromIA/CROM-IA-V3.5-Qwen-1.5B-Organic"

base_dir = os.path.dirname(os.path.abspath(__file__))

def read_static(filename):
    with open(os.path.join(base_dir, filename), "r", encoding="utf-8") as f:
        return f.read()

CSS = read_static("style.css")
HEADER = read_static("header.html").format(
    GITHUB=GITHUB,
    HF_MODEL=HF_MODEL,
    cores=cores,
    dna_count=len(dna_entries)
)
STATS = read_static("stats.html").format(dna_count=len(dna_entries))
FOOTER = read_static("footer.html")

with gr.Blocks(title="Rosa - CROM-IA V3.5b", fill_height=False) as demo:
    gr.HTML(HEADER)
    gr.HTML(STATS)

    gr.ChatInterface(
        generate,
        chatbot=gr.Chatbot(height=500),
        fill_height=False,
        additional_inputs=[
            gr.Slider(0.0, 1.5, step=0.05, value=0.1, label="🔥 Temperatura"),
            gr.Slider(1.0, 2.0, step=0.05, value=1.15, label="🔁 Penalidade Repeticao"),
            gr.Slider(256, 4096, step=256, value=2048, label="📏 Max Tokens"),
        ],
        additional_inputs_accordion=gr.Accordion("⚙️ Configuracoes Avancadas", open=False),
    )

    gr.HTML(FOOTER)

if __name__ == "__main__":
    demo.launch(ssr_mode=False, css=CSS)
