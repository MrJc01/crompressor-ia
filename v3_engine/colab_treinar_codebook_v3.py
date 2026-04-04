#!/usr/bin/env python3
"""
🧬 CROM-IA V3.5: Notebook de Treinamento Colab (Qwen2.5-1.5B)
================================================================
INSTRUÇÕES: Cada bloco abaixo é UMA CÉLULA no Google Colab.
Copie célula por célula. NÃO pule nenhuma.

ORDEM OBRIGATÓRIA:
  Célula 1 → Instalar pacotes
  (REINICIAR SESSÃO)
  Célula 2 → Checar GPU
  Célula 3 → Carregar Modelo + LoRA
  Célula 4 → Upload + Carregar Dataset 117k
  Célula 5 → Treinar (vai demorar ~3-4h)
  Célula 6 → Exportar GGUF e Baixar
================================================================
"""

# ╔═══════════════════════════════════════════════════════════════╗
# ║  CÉLULA 1: INSTALAR PACOTES                                  ║
# ║  Depois de rodar, vá em:                                      ║
# ║  Menu → Ambiente de Execução → Reiniciar Sessão               ║
# ╚═══════════════════════════════════════════════════════════════╝
"""
!nvidia-smi
!pip install unsloth
!pip uninstall unsloth -y && pip install --upgrade --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps xformers trl peft accelerate bitsandbytes datasets

# ╔═════════════════════════════════════════════╗
# ║  AGORA REINICIE A SESSÃO ANTES DE PROSSEGUIR  ║
# ║  Menu → Ambiente de Execução → Reiniciar       ║
# ╚═════════════════════════════════════════════╝
"""

# ╔═══════════════════════════════════════════════════════════════╗
# ║  CÉLULA 2: CHECAR GPU (rodar DEPOIS de reiniciar)             ║
# ╚═══════════════════════════════════════════════════════════════╝
"""
import torch
print("GPU OK:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Placa:", torch.cuda.get_device_name(0))
    print("VRAM:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")
    print("✅ GPU pronta! Siga para a Célula 3.")
else:
    raise RuntimeError("🚨 GPU não detectada! Vá em Ambiente de Execução → Alterar tipo → T4 GPU")
"""

# ╔═══════════════════════════════════════════════════════════════╗
# ║  CÉLULA 3: CARREGAR MODELO + APLICAR LORA                    ║
# ╚═══════════════════════════════════════════════════════════════╝

from unsloth import FastLanguageModel
import torch

max_seq_length = 2048
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 32,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 32,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)
print("✅ Modelo Qwen2.5-1.5B carregado com LoRA r=32!")

# ╔═══════════════════════════════════════════════════════════════╗
# ║  CÉLULA 4: UPLOAD DO ZIP + CARREGAR DATASET 117k              ║
# ║                                                               ║
# ║  ANTES de rodar: faça upload do colab_v3_training_kit.zip     ║
# ║  pela barra lateral (ícone de pasta 📁 → botão ⬆️)            ║
# ╚═══════════════════════════════════════════════════════════════╝

!unzip -o colab_v3_training_kit.zip

from datasets import load_dataset

dataset = load_dataset("json", data_files="data/dataset_v3_lora.jsonl", split="train")

alpaca_prompt = """Abaixo está uma instrução CROM-IA.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token

def formatar_causal(amostras):
    instrucoes = amostras["instruction"]
    inputs     = amostras["input"]
    outputs    = amostras["output"]
    textos     = []
    for instr, inp, out in zip(instrucoes, inputs, outputs):
        texto_final = alpaca_prompt.format(instr, inp, out) + EOS_TOKEN
        textos.append(texto_final)
    return { "text" : textos }

dataset_mapeado = dataset.map(formatar_causal, batched = True)
print(f"✅ {len(dataset_mapeado)} amostras reais formatadas!")

# ╔═══════════════════════════════════════════════════════════════╗
# ║  CÉLULA 5: TREINAMENTO (vai demorar ~3-4 horas)              ║
# ╚═══════════════════════════════════════════════════════════════╝

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset_mapeado,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 8, #apenas em A100, em um a normal T4 seria 2
        gradient_accumulation_steps = 1, #apenas em A100, em um a normal T4 seria 4
        warmup_steps = 50,
        num_train_epochs = 1,
        max_steps = -1,
        learning_rate = 1e-5,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 25,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        seed = 3407,
        output_dir = "outputs_crom_100k",
        save_strategy = "steps",
        save_steps = 500,
    ),
)

import os
print("🚀 Iniciando treinamento com 117k amostras reais...")
print("⏱️ Tempo estimado: ~1 hora na A100/L4, ou ~4 horas na T4")

# Lógica inteligente: se a pasta de checkpoints existir (Google Drive mount ou mesma sessão), retoma!
if os.path.exists("outputs_crom_100k") and any("checkpoint" in f for f in os.listdir("outputs_crom_100k")):
    print("🎯 Checkpoint salvo encontrado! Acionando Modo Turbo a partir de onde parou...")
    trainer_stats = trainer.train(resume_from_checkpoint=True)
else:
    print("🟢 Iniciando ciclo (Checkpoints limpos pelo reset de sessão).")
    trainer_stats = trainer.train()

print("✅ Treinamento concluído!")

# ╔═══════════════════════════════════════════════════════════════╗
# ║  CÉLULA 6: EXPORTAR GGUF E BAIXAR                            ║
# ╚═══════════════════════════════════════════════════════════════╝

print("⏳ Exportando para GGUF Q4_K_M (pode levar alguns minutos)...")
model.save_pretrained_gguf("modelo_rosa_v35", tokenizer, quantization_method = "q4_k_m")

import glob
gguf_files = glob.glob("modelo_rosa_v35*/*.gguf") + glob.glob("modelo_rosa_v35*.gguf")
print("Arquivos GGUF encontrados:", gguf_files)

if gguf_files:
    from google.colab import files
    for f in gguf_files:
        print(f"📦 Baixando {f}...")
        files.download(f)
else:
    print("❌ Nenhum GGUF encontrado. Liste os arquivos com: !find . -name '*.gguf'")
