#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║  🧬 CROM-IA V2: Script Unificado de Treino LoRA (Colab)    ║
║                                                              ║
║  Uso: Execute no Google Colab com GPU T4                     ║
║  Aceita parâmetros --taxa e --modo para treinar qualquer     ║
║  combinação de codebook semântico DNA                        ║
╚══════════════════════════════════════════════════════════════╝

INSTRUÇÕES:
-----------
1. Faça upload deste arquivo + dataset correspondente no Colab
2. Altere Runtime → T4 GPU
3. Execute cada CÉLULA em ordem

Parâmetros adaptativos por taxa:
- 1:3  → r=16, alpha=16, epochs=3  (5K códigos, fácil)
- 1:5  → r=32, alpha=64, epochs=5  (15K códigos, médio)
- 1:10 → r=64, alpha=128, epochs=7 (50K códigos, difícil)
- 1:20 → r=64, alpha=128, epochs=10 (100K+ códigos, exploratório)
"""

# ============================================================
# CONFIGURAÇÃO — ALTERE AQUI CONFORME A TAXA/MODO DESEJADO
# ============================================================
TAXA = "1x3"        # Opções: "1x3", "1x5", "1x10", "1x20"
MODO = "fixo"        # Opções: "fixo", "dinamico"
# ============================================================

# Parâmetros adaptativos baseados na taxa
PARAMS = {
    "1x3":  {"r": 16, "alpha": 16,  "epochs": 3, "max_steps": 1500, "lr": 2e-4},
    "1x5":  {"r": 32, "alpha": 64,  "epochs": 5, "max_steps": 2500, "lr": 1.5e-4},
    "1x10": {"r": 64, "alpha": 128, "epochs": 7, "max_steps": 4000, "lr": 1e-4},
    "1x20": {"r": 64, "alpha": 128, "epochs": 10, "max_steps": 5000, "lr": 8e-5},
}

config = PARAMS[TAXA]
DATASET_FILE = f"dataset_dna_{TAXA}_{MODO}.jsonl"

print(f"""
╔══════════════════════════════════════════════════════════════╗
║  🧬 CROM-IA V2: Treinamento LoRA — Codebook Semântico      ║
╠══════════════════════════════════════════════════════════════╣
║  Taxa     : {TAXA:<47}║
║  Modo     : {MODO:<47}║
║  Dataset  : {DATASET_FILE:<47}║
║  LoRA r   : {config['r']:<47}║
║  LoRA α   : {config['alpha']:<47}║
║  LR       : {config['lr']:<47}║
║  Steps    : {config['max_steps']:<47}║
╚══════════════════════════════════════════════════════════════╝
""")


# ============================================================
# === CÉLULA 1: INSTALAÇÃO DAS DEPENDÊNCIAS ==================
# ============================================================
"""
%%capture
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps xformers trl peft accelerate bitsandbytes triton
!pip install datasets sentencepiece protobuf

import torch
print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ CUDA disponível: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("⚠️  SEM GPU! Mude para T4!")
"""


# ============================================================
# === CÉLULA 2: CARREGAR MODELO + ATIVAR LORA =================
# ============================================================
"""
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
    dtype=None,
)

# LoRA adaptativo baseado na taxa
model = FastLanguageModel.get_peft_model(
    model,
    r=config['r'],
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=config['alpha'],
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

print(f"✅ Modelo carregado! LoRA r={config['r']}, α={config['alpha']}")
print(f"   Parâmetros treináveis: {model.print_trainable_parameters()}")
"""


# ============================================================
# === CÉLULA 3: CARREGAR DATASET ==============================
# ============================================================
"""
from datasets import load_dataset

dataset = load_dataset("json", data_files=DATASET_FILE, split="train")

print(f"✅ Dataset carregado: {len(dataset)} amostras")
print(f"   Colunas: {dataset.column_names}")
print(f"\\n📋 Exemplo:")
print(f"   Input: {dataset[0]['input'][:100]}...")
print(f"   Output: {dataset[0]['output'][:100]}...")

# Template ChatML do Qwen
def formatar_prompt(exemplos):
    textos = []
    for inst, inp, out in zip(exemplos["instruction"], exemplos["input"], exemplos["output"]):
        prompt = (
            "<|im_start|>system\\n"
            f"{inst}<|im_end|>\\n"
            "<|im_start|>user\\n"
            f"{inp}<|im_end|>\\n"
            "<|im_start|>assistant\\n"
            f"{out}<|im_end|>"
        )
        textos.append(prompt)
    return {"text": textos}

dataset = dataset.map(formatar_prompt, batched=True)
print(f"\\n✅ Dataset formatado ({len(dataset)} amostras)")
"""


# ============================================================
# === CÉLULA 4: TREINAR! ======================================
# ============================================================
"""
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_steps=config['max_steps'],
        warmup_steps=50,
        learning_rate=config['lr'],
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=25,
        output_dir=f"crom_lora_{TAXA}_{MODO}",
        save_strategy="steps",
        save_steps=500,
        optim="adamw_8bit",
        seed=42,
    ),
)

print(f"🚀 Treinando {TAXA} ({MODO})...")
print(f"   LoRA r={config['r']}, α={config['alpha']}")
print(f"   Max steps: {config['max_steps']}")

stats = trainer.train()

print(f"\\n{'=' * 55}")
print(f" ✅ TREINO {TAXA} ({MODO}) CONCLUÍDO!")
print(f"{'=' * 55}")
print(f" ⏱️  Tempo   : {stats.metrics['train_runtime']:.0f}s")
print(f" 📉 Loss    : {stats.metrics['train_loss']:.4f}")
print(f" 🔄 Steps   : {stats.global_step}")
print(f"{'=' * 55}")
"""


# ============================================================
# === CÉLULA 5: VALIDAÇÃO =====================================
# ============================================================
"""
from unsloth import FastLanguageModel
FastLanguageModel.for_inference(model)

testes = [
    {
        "titulo": f"TESTE 1: Compressão Codebook ({TAXA})",
        "prompt": f"<|im_start|>system\\nVocê é um compressor CROM DNA (taxa {TAXA.replace('x',':')}). Comprima a resposta usando códigos do codebook semântico DNA. Use prefixo @@ para palavras sem código.<|im_end|>\\n<|im_start|>user\\nO que é inteligência artificial?<|im_end|>\\n<|im_start|>assistant\\n",
    },
    {
        "titulo": f"TESTE 2: Decodificação ({TAXA})",
        "prompt": f"<|im_start|>system\\nDecodifique os códigos DNA CROM (codebook semântico {TAXA.replace('x',':')}) para linguagem humana em Português.<|im_end|>\\n<|im_start|>user\\nAA TT AT CG @@artificial<|im_end|>\\n<|im_start|>assistant\\n",
    },
    {
        "titulo": "TESTE 3: Sanidade (resposta normal)",
        "prompt": "<|im_start|>system\\nVocê é um assistente. Responda em Português.<|im_end|>\\n<|im_start|>user\\nO que é o Sol?<|im_end|>\\n<|im_start|>assistant\\n",
    },
]

print(f"{'=' * 55}")
print(f" 🧪 VALIDAÇÃO PÓS-TREINO ({TAXA}, {MODO})")
print(f"{'=' * 55}")

for teste in testes:
    print(f"\\n{'─' * 55}")
    print(f"  {teste['titulo']}")
    print(f"{'─' * 55}")
    
    inputs = tokenizer(teste["prompt"], return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.3,
        do_sample=True,
        use_cache=True,
    )
    resposta = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    print(f"  Resposta: {resposta[:200]}")
    
    chars_dna = sum(1 for c in resposta if c in 'ATCG')
    total_alpha = sum(1 for c in resposta if c.isalpha())
    if total_alpha > 0:
        ratio = chars_dna / total_alpha * 100
        print(f"  📊 DNA chars: {ratio:.0f}% ({chars_dna}/{total_alpha})")

print(f"\\n{'=' * 55}")
print(f" ✅ Validação concluída!")
print(f"{'=' * 55}")
"""


# ============================================================
# === CÉLULA 6: EXPORTAR GGUF =================================
# ============================================================
"""
import os, glob

OUTPUT_BASE = f"crom_dna_{TAXA}_{MODO}"
print(f"📦 Exportando {OUTPUT_BASE} para GGUF Q4_K_M...")
model.save_pretrained_gguf(
    OUTPUT_BASE,
    tokenizer,
    quantization_method="q4_k_m"
)

# Unsloth cria subdiretório com sufixo _gguf
gguf_files = glob.glob(f"{OUTPUT_BASE}*/**/*.gguf", recursive=True) + glob.glob(f"{OUTPUT_BASE}*/*.gguf")

FINAL_NAME = f"crom-dna-{TAXA}-{MODO}.gguf"

if gguf_files:
    # Renomeia para nome do projeto
    original = gguf_files[0]
    import shutil
    shutil.copy2(original, FINAL_NAME)
    size_mb = os.path.getsize(FINAL_NAME) / 1024**2
    print(f"\\n✅ GGUF renomeado:")
    print(f"   📄 {original} → {FINAL_NAME} ({size_mb:.1f} MB)")
else:
    print("   ⚠️ Nenhum GGUF encontrado! Verificando diretórios...")
    for d in glob.glob(f"{OUTPUT_BASE}*"):
        if os.path.isdir(d):
            print(f"   📁 {d}/: {os.listdir(d)}")

print(f"\\n{'=' * 55}")
print(f" 🎯 Baixe o .gguf e coloque em:")
print(f" crompressor-ia/models/{FINAL_NAME}")
print(f"{'=' * 55}")
"""


# ============================================================
# === CÉLULA 7: DOWNLOAD =====================================
# ============================================================
"""
from google.colab import files
import os

FINAL_NAME = f"crom-dna-{TAXA}-{MODO}.gguf"

if os.path.exists(FINAL_NAME):
    size_mb = os.path.getsize(FINAL_NAME) / 1024**2
    print(f"⬇️  Baixando: {FINAL_NAME} ({size_mb:.1f} MB)")
    files.download(FINAL_NAME)
else:
    print(f"❌ {FINAL_NAME} não encontrado!")
    print("   Execute a Célula 6 primeiro para exportar e renomear.")
"""
