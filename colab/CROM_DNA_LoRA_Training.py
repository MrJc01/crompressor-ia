#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║  🧬 CROM-IA: Treinamento LoRA DNA Radix-4 no Google Colab  ║
║                                                              ║
║  Como usar:                                                  ║
║  1. Abra https://colab.research.google.com                   ║
║  2. Altere Runtime → T4 GPU                                  ║
║  3. Faça upload deste arquivo e do dataset_dna.jsonl          ║
║  4. Copie e cole cada CÉLULA abaixo como células separadas   ║
║  5. Execute uma por uma (Play)                               ║
║                                                              ║
║  Ou cole o arquivo inteiro numa única célula e execute!       ║
╚══════════════════════════════════════════════════════════════╝

INSTRUÇÕES:
-----------
Cada seção marcada com "# === CÉLULA X ===" é uma célula do Colab.
Copie o bloco inteiro (de CÉLULA até a próxima CÉLULA) e cole
como uma nova célula no Colab.

Ao final de CADA célula, copie o output do Colab e envie
de volta para o assistente CROM-IA para diagnóstico.
"""

# ============================================================
# === CÉLULA 1: INSTALAÇÃO DAS DEPENDÊNCIAS ==================
# ============================================================
# Cole isto na PRIMEIRA célula do Colab e aperte Play.
# Vai demorar ~2 minutos. Pode aparecer "Restart Runtime" — 
# se aparecer, clique "Restart" e pule para a CÉLULA 2.
#
# COPIE DAQUI ↓↓↓
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
    print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
else:
    print("⚠️  SEM GPU DETECTADA! Mude o Runtime para T4 GPU!")
"""

# ============================================================
# === CÉLULA 2: CARREGAR MODELO + ATIVAR LORA =================
# ============================================================
# Após a CÉLULA 1 (e restart se necessário), cole isto:
#
# COPIE DAQUI ↓↓↓
"""
from unsloth import FastLanguageModel
import torch

# Carrega o Qwen 2.5 0.5B na GPU em 4-bit (mesmo modelo do seu PC!)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
    dtype=None,  # auto-detect
)

# Injeta adaptadores LoRA nos módulos de atenção
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                    # rank do LoRA (16 = bom equilíbrio qualidade/velocidade)
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",  # economiza 30% VRAM
    random_state=42,
)

print("✅ Modelo carregado e LoRA ativado!")
print(f"   Parâmetros treináveis: {model.print_trainable_parameters()}")
"""

# ============================================================
# === CÉLULA 3: CARREGAR DATASET DNA ==========================
# ============================================================
# ANTES desta célula: faça upload do dataset_dna.jsonl
# no painel de arquivos do Colab (ícone de pasta à esquerda).
#
# COPIE DAQUI ↓↓↓
"""
from datasets import load_dataset

# Carrega o seu dataset DNA gerado localmente
dataset = load_dataset("json", data_files="dataset_dna.jsonl", split="train")

print(f"✅ Dataset carregado: {len(dataset)} amostras")
print(f"   Colunas: {dataset.column_names}")
print(f"\\n📋 Exemplo da FASE A (Humano → DNA):")
print(f"   Input: {dataset[0]['input'][:80]}...")
print(f"   Output: {dataset[0]['output'][:80]}...")

# Formatar prompts no template ChatML do Qwen
def formatar_prompt(exemplos):
    textos = []
    for inst, inp, out in zip(exemplos["instruction"], exemplos["input"], exemplos["output"]):
        # Template ChatML (formato nativo do Qwen 2.5)
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

print(f"\\n✅ Dataset formatado (ChatML)!")
print(f"   Exemplo formatado:")
print(dataset[0]["text"][:200] + "...")
"""

# ============================================================
# === CÉLULA 4: TREINAR! ======================================
# ============================================================
# Esta é a célula que faz a mágica. Vai levar ~20-40 minutos
# dependendo do tamanho do dataset. Observe o Loss caindo!
#
# COPIE DAQUI ↓↓↓
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
        # === Batch & Steps ===
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_steps=1500,          # ~1500 steps para 10K amostras
        warmup_steps=50,

        # === Learning Rate ===
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        weight_decay=0.01,

        # === Precision ===
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),

        # === Logging ===
        logging_steps=25,
        output_dir="crom_lora_output",
        save_strategy="steps",
        save_steps=500,

        # === Otimização ===
        optim="adamw_8bit",
        seed=42,
    ),
)

print("🚀 Iniciando treinamento LoRA...")
print("   Observe o Loss abaixo. Ele deve cair de ~2.5 para < 1.0")
print("   Se o Loss parar de cair, o modelo saturou (normal para 0.5B).")
print("")

stats = trainer.train()

# Relatório final
print("\\n" + "=" * 55)
print(" ✅ TREINAMENTO CONCLUÍDO!")
print("=" * 55)
print(f" ⏱️  Tempo total    : {stats.metrics['train_runtime']:.0f} segundos")
print(f" 📉 Loss final     : {stats.metrics['train_loss']:.4f}")
print(f" 🔄 Steps          : {stats.metrics['train_steps']}")
print(f" ⚡ Amostras/seg   : {stats.metrics['train_samples_per_second']:.1f}")
print("=" * 55)
"""

# ============================================================
# === CÉLULA 5: TESTAR ANTES DE EXPORTAR ======================
# ============================================================
# Vamos validar se o modelo aprendeu DNA antes de exportar!
#
# COPIE DAQUI ↓↓↓
"""
from unsloth import FastLanguageModel
FastLanguageModel.for_inference(model)

# Testes A/B: DNA Codificação e Decodificação
testes = [
    {
        "titulo": "TESTE 1: Codificação Humano → DNA",
        "prompt": "<|im_start|>system\\nVocê é uma Célula CROM. Transcreva a resposta como Cadeia Quaternária DNA CROM Base-4 (usar apenas A, T, C, G).<|im_end|>\\n<|im_start|>user\\nO que é inteligência artificial?<|im_end|>\\n<|im_start|>assistant\\n",
    },
    {
        "titulo": "TESTE 2: Decodificação DNA → Humano",
        "prompt": "<|im_start|>system\\nDecodifique a sequência biológica DNA CROM Base-4 para linguagem humana em Português.<|im_end|>\\n<|im_start|>user\\nTACGTACGCCGGATCT<|im_end|>\\n<|im_start|>assistant\\n",
    },
    {
        "titulo": "TESTE 3: Resposta Normal (sanidade)",
        "prompt": "<|im_start|>system\\nVocê é um assistente inteligente. Responda em Português.<|im_end|>\\n<|im_start|>user\\nO que é o Crompressor?<|im_end|>\\n<|im_start|>assistant\\n",
    },
]

print("=" * 55)
print(" 🧪 VALIDAÇÃO PÓS-TREINO")
print("=" * 55)

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

    # Verificar se output contém DNA válido
    chars_dna = sum(1 for c in resposta if c in 'ATCG')
    total_alpha = sum(1 for c in resposta if c.isalpha())
    if total_alpha > 0:
        ratio_dna = chars_dna / total_alpha * 100
        print(f"  📊 Conteúdo DNA: {ratio_dna:.0f}% ({chars_dna}/{total_alpha} chars)")

print(f"\\n{'=' * 55}")
print(" ✅ Validação concluída! Se os testes parecem OK,")
print("    prossiga para a CÉLULA 6 para exportar.")
print(f"{'=' * 55}")
"""

# ============================================================
# === CÉLULA 6: EXPORTAR COMO GGUF ============================
# ============================================================
# Esta célula salva o adaptador no formato que o llama-cli lê!
# O arquivo gerado terá ~20-50 MB.
#
# COPIE DAQUI ↓↓↓
"""
# Salvar em formato GGUF Q4_K_M (mesmo formato do seu llama-cli)
print("📦 Exportando para GGUF Q4_K_M...")
model.save_pretrained_gguf(
    "crom_dna_lora_gguf",
    tokenizer,
    quantization_method="q4_k_m"
)

import os
# Listar arquivos gerados
print("\\n✅ Arquivos gerados:")
for f in os.listdir("crom_dna_lora_gguf"):
    size_mb = os.path.getsize(f"crom_dna_lora_gguf/{f}") / 1024 / 1024
    print(f"   📄 {f} ({size_mb:.1f} MB)")

print("\\n" + "=" * 55)
print(" 🎯 PRÓXIMO PASSO:")
print(" Baixe o arquivo 'unsloth.Q4_K_M.gguf' da pasta")
print(" 'crom_dna_lora_gguf' (clique direito → Download)")
print(" e coloque em:")
print(" /home/j/Área de trabalho/crompressor-ia/models/")
print("=" * 55)
"""

# ============================================================
# === CÉLULA 7 (OPCIONAL): DOWNLOAD AUTOMÁTICO =================
# ============================================================
# Se o download manual não funcionar, use esta célula:
#
# COPIE DAQUI ↓↓↓
"""
from google.colab import files
import glob

# Encontra o GGUF gerado
gguf_files = glob.glob("crom_dna_lora_gguf/*.gguf")
if gguf_files:
    for f in gguf_files:
        print(f"⬇️  Iniciando download de: {f}")
        files.download(f)
else:
    print("❌ Nenhum arquivo GGUF encontrado! Verifique a CÉLULA 6.")
"""

# ============================================================
# FIM DO SCRIPT COLAB
# ============================================================
print("""
╔══════════════════════════════════════════════════════════════╗
║  📋 RESUMO: O QUE COPIAR NO COLAB                          ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  CÉLULA 1 → Instalar dependências (~2 min)                  ║
║  CÉLULA 2 → Carregar Qwen 0.5B + LoRA (~1 min)              ║
║  CÉLULA 3 → Carregar seu dataset_dna.jsonl                   ║
║  CÉLULA 4 → TREINAR! (~20-40 min)                           ║
║  CÉLULA 5 → Testar DNA encode/decode                        ║
║  CÉLULA 6 → Exportar GGUF                                   ║
║  CÉLULA 7 → Download do arquivo (opcional)                   ║
║                                                              ║
║  ⚠️  Mande o output de cada célula para o assistente!        ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")
