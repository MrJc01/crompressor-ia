#!/usr/bin/env python3
"""
CROM-IA V4.2 — FASE 2: SFT Especialização com DNA 25%
=======================================================
COLE ESTE CÓDIGO NO GOOGLE COLAB (A100/T4)
PRÉ-REQUISITO: Fase 1 já rodou (Base_PTBR treinado)

O modelo já sabe conversar (Fase 1).
Agora aprende DNA sutil (25%) por domínio.

LoRAs: Python_DNA (15K, 500 steps) + Medicina_DNA (8K, 500 steps)
"""

# ══════════════════════════════════════════════════════════════
# CÉLULA 1 — TREINO FASE 2 (copiar e colar no Colab)
# ══════════════════════════════════════════════════════════════
CELULA_TREINO = """
import os, gc, torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# ════════════════════════════════════════════════
# CONFIGURAÇÃO V4.2 — FASE 2 (DNA 25%)
# ════════════════════════════════════════════════
max_seq_length = 2048
qwen_base = "unsloth/Qwen3-0.6B-unsloth-bnb-4bit"

def formatting_func(example):
    output = example["text"]
    if isinstance(output, list):
        return [str(x) for x in output]
    return [str(output)]


def treinar_cerebro_dna(nome_cerebro, path_dataset, max_steps=500):
   
    if not os.path.exists(path_dataset):
        print(f"⚠️ {path_dataset} não encontrado! Pulando...")
        return

    print(f"\\n{'='*60}")
    print(f"🧬 FASE 2: Treinando {nome_cerebro} (DNA 25%)")
    print(f"   Dataset: {path_dataset}")
    print(f"   Steps: {max_steps} | Rank: 16 | LR: 1e-5")
    print(f"{'='*60}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=qwen_base,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    # NOTA SRE: Fundir LoRAs em 4-bit no Unsloth via merge_and_unload causa 'BFloat16 != float' (Crash de Type).
    # O CROM-IA usará Stacking (Empilhamento Local) depois. O Cérebro de Python 
    # DEVE ser treinado sob a camada limpa do Qwen, e o orquestrador Shell somará os 2 LoRAs!

    # Novo LoRA para especialização DNA
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

    dataset = load_dataset("json", data_files=path_dataset, split="train")
    print(f"   📊 Amostras: {len(dataset)}")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        formatting_func=formatting_func,
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        args=TrainingArguments(
            per_device_train_batch_size=8,
            gradient_accumulation_steps=4,
            warmup_ratio=0.05,
            max_steps=max_steps,
            learning_rate=1e-5,
            lr_scheduler_type="cosine",
            optim="adamw_8bit",
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            output_dir=f"./outputs_{nome_cerebro}",
            logging_steps=25,
        ),
    )

    trainer.train()

    # Salvar LoRA SEPARADO (para empilhar depois)
    adapter_dir = f"/content/drive/MyDrive/CROM-V4.2/adapters/{nome_cerebro}"
    os.makedirs(adapter_dir, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"   ✅ LoRA salvo: {adapter_dir}")

    # GGUF fundido (standalone)
    gguf_dir = f"/content/drive/MyDrive/CROM-V4.2/gguf_merged/{nome_cerebro}"
    os.makedirs(gguf_dir, exist_ok=True)
    model.save_pretrained_gguf(gguf_dir, tokenizer, quantization_method="q4_k_m")
    print(f"   ✅ GGUF salvo: {gguf_dir}")

    del model; del trainer; gc.collect(); torch.cuda.empty_cache()
    print(f"   🎉 {nome_cerebro} CONCLUÍDO!")


# ── Montar Drive ──────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

print("\\n" + "=" * 60)
print("🧬 CROM-IA V4.2 — FASE 2: Especialização DNA (25%)")
print("=" * 60)

# Upload dos datasets transpilados para /content/:
#   - python_DNA25.jsonl
#   - medicina_DNA25.jsonl

# ── Fábrica de Cérebros DNA ───────────────────────────────
cerebros = [
    ("Python_DNA",   "Python_DNA25.jsonl",   500),
    # ("Medicina_DNA", "medicina_DNA25.jsonl",  500),
]

for nome, arq, steps in cerebros:
    treinar_cerebro_dna(nome, arq, max_steps=steps)

print("\\n" + "=" * 60)
print("🎉 FASE 2 CONCLUÍDA!")
print("   → Python_DNA e Medicina_DNA treinados com DNA 25%")
print("   → Adaptadores em: CROM-V4.2/adapters/")
print("\\n   PRÓXIMO: Execute 03_CROM_V42_DPO_TRAINING.py")
print("=" * 60)
"""

if __name__ == "__main__":
    print("=" * 60)
    print("📋 CROM-IA V4.2 — Fase 2: SFT DNA 25%")
    print("=" * 60)
    print("\n🏋️ CÉLULA (Treinamento):")
    print(CELULA_TREINO)
