#!/usr/bin/env python3
"""
CROM-IA V4.2 — FASE 1: SFT Base Conversacional (SEM DNA)
==========================================================
COLE ESTE CÓDIGO NO GOOGLE COLAB (A100/T4)

O modelo aprende a CONVERSAR BEM em PT-BR primeiro.
Sem DNA, sem tokens @@. Apenas fluência e inteligência.

Dataset: Canarim 30K + OpenHermes 10K traduzido = ~40K
Parâmetros: rank 16, 800 steps, lr 1e-5, cosine scheduler
"""

# ══════════════════════════════════════════════════════════════
# CÉLULA 1 — INSTALAÇÃO (rodar primeiro, esperar terminar)
# ══════════════════════════════════════════════════════════════
CELULA_1_INSTALACAO = """
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install trl>=0.7.0
"""

# ══════════════════════════════════════════════════════════════
# CÉLULA 2 — TREINO FASE 1 (copiar e colar no Colab)
# ══════════════════════════════════════════════════════════════
CELULA_2_TREINO = """
import os, gc, torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# ════════════════════════════════════════════════
# CONFIGURAÇÃO V4.2 — FASE 1 (SEM DNA)
# ════════════════════════════════════════════════
max_seq_length = 2048
qwen_base = "unsloth/Qwen3-0.6B-unsloth-bnb-4bit"

print("=" * 60)
print("🧠 CROM-IA V4.2 — FASE 1: Base Conversacional (SEM DNA)")
print("   Modelo: Qwen3-0.6B")
print("   Rank: 16 | Steps: 800 | LR: 1e-5")
print("   DNA: 0% (o modelo aprende a FALAR primeiro)")
print("=" * 60)

# ── Montar Drive ──────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')
os.makedirs('/content/drive/MyDrive/CROM-V4.2/adapters', exist_ok=True)
os.makedirs('/content/drive/MyDrive/CROM-V4.2/gguf_merged', exist_ok=True)

# ── Carregar modelo base ──────────────────────────────────
print("\\n📦 Carregando Qwen3-0.6B...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=qwen_base,
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=True,
)

# ── LoRA conservador (rank 16, SÓ attention) ─────────────
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                    # Era 64 na V4.1 (causou forgetting!)
    lora_alpha=32,           # 2x rank
    lora_dropout=0,
    bias="none",
    target_modules=[         # SÓ ATTENTION — sem MLP!
        "q_proj", "k_proj", "v_proj", "o_proj"
    ],
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# ── Carregar dataset ──────────────────────────────────────
# Upload dos arquivos para /content/ antes de rodar:
#   - Base_PTBR.jsonl (nossa fusão local de 40k)

datasets_fase1 = []
for arq in ["Base_PTBR.jsonl"]:
    if os.path.exists(arq):
        datasets_fase1.append(arq)
        print(f"   ✅ {arq}")
    else:
        print(f"   ⚠️ {arq} não encontrado")

if not datasets_fase1:
    raise FileNotFoundError("Nenhum dataset encontrado! Faça upload primeiro.")

dataset = load_dataset("json", data_files=datasets_fase1, split="train")
print(f"\\n📊 Dataset Fase 1: {len(dataset)} amostras")

# ── Formatting ────────────────────────────────────────────
def formatting_func(example):
    output = example["text"]
    if isinstance(output, list):
        return [str(x) for x in output]
    return [str(output)]

# ── Treinar ───────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    formatting_func=formatting_func,
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    args=TrainingArguments(
        per_device_train_batch_size=8,       # Era 16 (mais conservador)
        gradient_accumulation_steps=4,
        warmup_ratio=0.05,                   # 5% warmup
        max_steps=800,                       # Era 2000 (causou overfitting!)
        learning_rate=1e-5,                  # Era 2e-5 (mais suave)
        lr_scheduler_type="cosine",          # Convergência melhor
        optim="adamw_8bit",
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        output_dir="./outputs_Base_PTBR",
        logging_steps=25,
        save_steps=200,
    ),
)

print("\\n🚀 Iniciando treino Fase 1...")
trainer.train()
print("✅ Treino Fase 1 concluído!")

# ── Salvar adaptador LoRA ─────────────────────────────────
adapter_dir = "/content/drive/MyDrive/CROM-V4.2/adapters/Base_PTBR"
os.makedirs(adapter_dir, exist_ok=True)
model.save_pretrained(adapter_dir)
tokenizer.save_pretrained(adapter_dir)
print(f"✅ LoRA Base_PTBR salvo em: {adapter_dir}")

# ── Salvar GGUF fundido (para testes standalone) ─────────
gguf_dir = "/content/drive/MyDrive/CROM-V4.2/gguf_merged/Base_PTBR"
os.makedirs(gguf_dir, exist_ok=True)
model.save_pretrained_gguf(gguf_dir, tokenizer, quantization_method="q4_k_m")
print(f"✅ GGUF Base_PTBR salvo em: {gguf_dir}")

# ── Salvar modelo BASE puro (sem LoRA) para stacking ─────
print("\\n📦 Salvando modelo BASE puro...")
del model; del trainer; gc.collect(); torch.cuda.empty_cache()
base_model, base_tok = FastLanguageModel.from_pretrained(
    model_name=qwen_base, max_seq_length=max_seq_length,
    dtype=None, load_in_4bit=True,
)
base_dir = "/content/drive/MyDrive/CROM-V4.2/base_model"
os.makedirs(base_dir, exist_ok=True)
base_model.save_pretrained_gguf(base_dir, base_tok, quantization_method="q4_k_m")
del base_model; del base_tok; gc.collect(); torch.cuda.empty_cache()
print("✅ Modelo base GGUF salvo!")

print("\\n" + "=" * 60)
print("🎉 FASE 1 CONCLUÍDA!")
print("   → Adapters: CROM-V4.2/adapters/Base_PTBR/")
print("   → GGUF: CROM-V4.2/gguf_merged/Base_PTBR/")
print("   → Base: CROM-V4.2/base_model/")
print("\\n   PRÓXIMO: Execute 02_CROM_V42_TRAINING_FASE2.py")
print("=" * 60)
"""

if __name__ == "__main__":
    print("=" * 60)
    print("📋 CROM-IA V4.2 — Fase 1: SFT Base")
    print("=" * 60)
    print("\n🔧 CÉLULA 1 (Instalação):")
    print(CELULA_1_INSTALACAO)
    print("\n🏋️ CÉLULA 2 (Treinamento):")
    print(CELULA_2_TREINO)
