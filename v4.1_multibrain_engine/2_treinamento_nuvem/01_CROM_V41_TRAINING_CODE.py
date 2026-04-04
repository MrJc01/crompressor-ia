#!/usr/bin/env python3
"""
CROM-IA V4.1 — Código de Treino para Google Colab (A100)
=========================================================
MUDANÇAS vs V4.0:
- Modelo: Qwen3-0.6B (menor, mais rápido, melhor PT-BR)
- Steps: 2000 (vs 400)
- Rank LoRA: 64 (vs 32)
- Salva LoRA SEPARADO (empilhável), não fundido
- 4 Cérebros: Python, Medicina, CROM_Self, Conversa
=========================================================

INSTRUÇÃO: Cole este código em DUAS células do Colab.
CÉLULA 1: Instalação (rode primeiro e espere terminar)
CÉLULA 2: Treinamento (rode depois)
"""

# =================================================================
# CÉLULA 1 — INSTALAÇÃO (Copiar e colar no Colab)
# =================================================================
CELULA_1 = """
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
"""

# =================================================================
# CÉLULA 2 — TREINAMENTO COMPLETO (Copiar e colar no Colab)
# =================================================================
CELULA_2 = """
import os
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
from unsloth import is_bfloat16_supported
import gc, torch

# ============================================================
# CONFIGURAÇÃO V4.1
# ============================================================
max_seq_length = 2048
qwen_base = "unsloth/Qwen3-0.6B-unsloth-bnb-4bit"  # UPGRADE: Qwen3 0.6B

def formatting_func(example):
    output = example["text"]
    if isinstance(output, list):
        return [str(x) for x in output]
    return [str(output)]

def treinar_cerebro_v41(nome_cerebro, path_dataset):
    if not os.path.exists(path_dataset):
        print(f"⚠️ Arquivo {path_dataset} não encontrado! Pulando...")
        return
        
    print(f"\\n{'='*60}")
    print(f"🚀 INICIANDO TREINO V4.1: {nome_cerebro}")
    print(f"   Modelo: Qwen3-0.6B | Steps: 2000 | Rank: 64")
    print(f"{'='*60}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = qwen_base, max_seq_length = max_seq_length,
        dtype = None, load_in_4bit = True,
    )
    
    # LoRA V4.1: Rank DOBRADO (64 vs 32)
    model = FastLanguageModel.get_peft_model(
        model, r = 64, lora_alpha = 128, lora_dropout = 0, bias = "none",
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "down_proj", "up_proj", "gate_proj"],
        use_gradient_checkpointing = "unsloth", random_state = 3407,
    )
    
    dataset = load_dataset("json", data_files=path_dataset, split="train")
    
    trainer = SFTTrainer(
        model = model, tokenizer = tokenizer, train_dataset = dataset,
        formatting_func = formatting_func,
        max_seq_length = max_seq_length, dataset_num_proc = 2,
        args = TrainingArguments(
            per_device_train_batch_size = 16,
            gradient_accumulation_steps = 2,
            warmup_steps = 20,
            max_steps = 2000,              # 5x mais que V4.0!
            learning_rate = 2e-5,
            optim = "adamw_8bit",
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            output_dir = f"./outputs_{nome_cerebro}",
            logging_steps = 50,
        ),
    )
    
    trainer.train()
    
    # ==========================================================
    # V4.1 DIFERENÇA CRÍTICA: Salvar LoRA SEPARADO (empilhável)
    # ==========================================================
    adapter_dir = f"/content/drive/MyDrive/CROM-V4.1/adapters/{nome_cerebro}"
    os.makedirs(adapter_dir, exist_ok=True)
    
    # Salvar adaptador LoRA separado (NÃO fundir!)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"✅ Adaptador LoRA salvo em: {adapter_dir}")
    
    # TAMBÉM salvar versão GGUF fundida para uso standalone
    gguf_dir = f"/content/drive/MyDrive/CROM-V4.1/gguf_merged/{nome_cerebro}"
    os.makedirs(gguf_dir, exist_ok=True)
    model.save_pretrained_gguf(gguf_dir, tokenizer, quantization_method = "q4_k_m")
    print(f"✅ GGUF fundido salvo em: {gguf_dir}")
    
    del model; del trainer; gc.collect(); torch.cuda.empty_cache()
    print(f"🎉 {nome_cerebro} CONCLUÍDO!")

# ============================================================
# MONTAR DRIVE E EXECUTAR FÁBRICA DE 4 CÉREBROS
# ============================================================
from google.colab import drive
drive.mount('/content/drive')
os.makedirs('/content/drive/MyDrive/CROM-V4.1/adapters', exist_ok=True)
os.makedirs('/content/drive/MyDrive/CROM-V4.1/gguf_merged', exist_ok=True)

# TAMBÉM salvar o modelo base GGUF puro (para empilhar LoRAs depois)
print("\\n📦 Salvando modelo BASE Qwen3-0.6B como GGUF puro...")
base_model, base_tok = FastLanguageModel.from_pretrained(
    model_name = qwen_base, max_seq_length = max_seq_length,
    dtype = None, load_in_4bit = True,
)
base_dir = "/content/drive/MyDrive/CROM-V4.1/base_model"
os.makedirs(base_dir, exist_ok=True)
base_model.save_pretrained_gguf(base_dir, base_tok, quantization_method = "q4_k_m")
del base_model; del base_tok; gc.collect(); torch.cuda.empty_cache()
print("✅ Modelo base GGUF salvo!")

# Lista de cérebros para treinar
cerebros = [
    ("Python_DNA",       "dataset_Python_DNA75_smart.jsonl"),
    ("Medicina_DNA",     "dataset_Medicina_DNA75_smart.jsonl"),
    ("CROM_Self",        "dataset_CROM_Self_DNA75_smart.jsonl"),
    ("Conversa_PTBR",    "dataset_Conversa_DNA75_smart.jsonl"),
]

for nome, arquivo in cerebros:
    treinar_cerebro_v41(nome, arquivo)

print("\\n" + "="*60)
print("🎉🎉🎉 FÁBRICA V4.1 CONCLUÍDA! 4 CÉREBROS TREINADOS!")
print("   Verifique no Google Drive: CROM-V4.1/")
print("   - adapters/ → LoRAs empilháveis")
print("   - gguf_merged/ → Modelos standalone")
print("   - base_model/ → Qwen3-0.6B puro para empilhamento")
print("="*60)
"""

if __name__ == "__main__":
    print("=" * 60)
    print("📋 CROM-IA V4.1 — Código de Treino para Colab")
    print("=" * 60)
    print("\n🔧 CÉLULA 1 (Instalação):")
    print(CELULA_1)
    print("\n🏋️ CÉLULA 2 (Treinamento):")
    print(CELULA_2)
