#!/usr/bin/env python3
"""
CROM-IA V4.2 — FASE 3: DPO (Direct Preference Optimization)
=============================================================
COLE ESTE CÓDIGO NO GOOGLE COLAB (A100/T4)
PRÉ-REQUISITO: Fase 1 e Fase 2 já rodaram

O modelo já sabe conversar (Fase 1) e conhece DNA (Fase 2).
Agora aprende a PREFERIR respostas com DNA sobre sem DNA.

Dataset: 5K pares {prompt, chosen(DNA), rejected(sem DNA)}
Parâmetros: beta=0.1, 300 steps, lr 5e-6 (muito suave)
"""

# ══════════════════════════════════════════════════════════════
# CÉLULA 1 — INSTALAÇÃO EXTRA (se não fez na Fase 1)
CELULA_1_DEPS = """
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install trl>=0.7.0
!pip install mergekit
!pip install llm-blender
!pip install weave
"""

# ══════════════════════════════════════════════════════════════
# CÉLULA 2 — TREINO DPO (copiar e colar no Colab)
# ══════════════════════════════════════════════════════════════
CELULA_2_DPO = """
import os, gc, torch
import transformers

# Vacina de SRE para o conflito do TRL c/ Transformers moderno:
if not hasattr(transformers.utils.hub, 'TRANSFORMERS_CACHE'):
    transformers.utils.hub.TRANSFORMERS_CACHE = os.getenv('HF_HOME', '~/.cache/huggingface/hub')

from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset

# ════════════════════════════════════════════════
# CONFIGURAÇÃO V4.2 — FASE 3 (DPO)
# ════════════════════════════════════════════════
max_seq_length = 2048
qwen_base = "unsloth/Qwen3-0.6B-unsloth-bnb-4bit"

print("=" * 60)
print("🎯 CROM-IA V4.2 — FASE 3: DPO (Preferência DNA)")
print("   O modelo aprende: DNA = resposta PREFERIDA")
print("   Beta: 0.1 | Steps: 300 | LR: 5e-6")
print("=" * 60)

# ── Montar Drive ──────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

# ── Carregar modelo com Base da Fase 1 ────────────────────
print("\\n📦 Carregando Qwen3-0.6B...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=qwen_base,
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=True,
)

# NOTA SRE: Fundir LoRA com merge_and_unload causaria BFloat16 vs Float exception.
# A Fase de DPO (Preferência de DNA) deve ser puramente treinada a partir das 
# fundações matrizes 4-bits do Qwen. Todos os LoRAs serão empilhados juntos localmente.

# ── Novo LoRA para DPO ────────────────────────────────────
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# ── Dataset DPO ───────────────────────────────────────────
# Upload do dataset DPO para /content/:
#   - dataset_DPO_canarim_30k.jsonl (ou similar)
#   Formato: {"prompt": "...", "chosen": "...(com DNA)...", "rejected": "...(sem DNA)..."}

dpo_files = [f for f in os.listdir('.') if f.startswith('dataset_DPO') and f.endswith('.jsonl')]
if not dpo_files:
    raise FileNotFoundError(
        "Nenhum dataset DPO encontrado!\\n"
        "Faça upload de um arquivo dataset_DPO_*.jsonl\\n"
        "Gere com: python3 gerador_pares_dpo.py --input ... --codebook ..."
    )

print(f"\\n📊 Datasets DPO encontrados: {dpo_files}")
dataset = load_dataset("json", data_files=dpo_files, split="train")
print(f"   Total: {len(dataset)} pares")

# ── Tokenizer padding ────────────────────────────────────
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

# Vacina de SRE para o conflito Peft/Unsloth vs DPOTrainer:
if not hasattr(model, "warnings_issued"):
    model.warnings_issued = {}

# ── Treinar com DPO ───────────────────────────────────────
print("\\n🎯 Iniciando DPO...")

training_args = DPOConfig(
    per_device_train_batch_size=4,       # Menor — DPO usa 2x memória por amostra
    gradient_accumulation_steps=4,
    max_steps=300,                        # Pouco — DPO converge rápido
    learning_rate=5e-6,                   # METADE do SFT (refinamento, não reeducação)
    beta=0.1,                             # Sutil — não forçar DNA demais
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    optim="adamw_8bit",
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    output_dir="./outputs_DPO",
    logging_steps=25,
    max_length=max_seq_length,
    max_prompt_length=512,
)

trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)

trainer.train()
print("✅ DPO concluído!")

# ── Salvar adaptador DPO ──────────────────────────────────
adapter_dir = "/content/drive/MyDrive/CROM-V4.2/adapters/DPO_Preference"
os.makedirs(adapter_dir, exist_ok=True)
model.save_pretrained(adapter_dir)
tokenizer.save_pretrained(adapter_dir)
print(f"✅ LoRA DPO salvo: {adapter_dir}")

# ── GGUF fundido completo (Base + DPO) ───────────────────
gguf_dir = "/content/drive/MyDrive/CROM-V4.2/gguf_merged/DPO_Preference"
os.makedirs(gguf_dir, exist_ok=True)
model.save_pretrained_gguf(gguf_dir, tokenizer, quantization_method="q4_k_m")
print(f"✅ GGUF DPO salvo: {gguf_dir}")

del model; del trainer; gc.collect(); torch.cuda.empty_cache()

# ── Relatório Final ───────────────────────────────────────
print("\\n" + "=" * 60)
print("🎉🎉🎉 TODAS AS 3 FASES CONCLUÍDAS!")
print("=" * 60)
print("\\nArquivos no Google Drive → CROM-V4.2/")
print("├── base_model/          ← Qwen3-0.6B puro (Q4_K_M)")
print("├── adapters/")
print("│   ├── Base_PTBR/       ← Fase 1: Conversação PT-BR")
print("│   ├── Python_DNA/      ← Fase 2: Python com DNA 25%")
print("│   ├── Medicina_DNA/    ← Fase 2: Medicina com DNA 25%")
print("│   └── DPO_Preference/  ← Fase 3: Preferência DNA")
print("└── gguf_merged/         ← Modelos standalone")
print("")
print("📋 PRÓXIMOS PASSOS:")
print("   1. Baixar os GGUFs para o i5")
print("   2. Converter adaptadores PEFT → GGUF-LoRA:")
print("      python3 convert_lora_to_gguf.py --base qwen.gguf --adapter adapter/")
print("   3. Testar empilhamento:")
print("      llama-cli -m base.gguf --lora Base.gguf --lora Python.gguf")
print("   4. Abrir o Monitor:")
print("      ./chat_v42_brain.sh")
print("=" * 60)
"""

if __name__ == "__main__":
    print("=" * 60)
    print("📋 CROM-IA V4.2 — Fase 3: DPO")
    print("=" * 60)
    print("\n🔧 CÉLULA 1 (Instalação):")
    print(CELULA_1_DEPS)
    print("\n🎯 CÉLULA 2 (DPO Training):")
    print(CELULA_2_DPO)
