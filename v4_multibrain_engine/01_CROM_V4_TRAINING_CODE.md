# CÓDIGO FONTE - COLAB V4 (COPIE E COLE)

Mestre, crie um "Notebook" em branco no Google Colab de sua preferência. Selecione o ambiente T4 ou A100.
Na lateral esquerda, faça o upload dos seus arquivos que eu deixei na sua máquina: **`dataset_P_Hibrido.jsonl`** e **`dataset_M_Hibrido.jsonl`**.

Após enviar, **copie o bloco abaixo** e cole inteiramente lá e aperte Play:

```python
# 1. Instalações
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes datasets

import os
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
from unsloth import is_bfloat16_supported
import gc, torch

# 2. Configurações Básicas
max_seq_length = 2048
qwen_base = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"

# 3. Orquestrador de Adaptadores (Treinamento O(1))
def treinar_modulo_cerebral(nome_plugin, caminho_dataset_hibrido):
    if not os.path.exists(caminho_dataset_hibrido):
        print(f"⚠️ Arquivo {caminho_dataset_hibrido} não encontrado! O pulando...")
        return
        
    print(f"\\n🚀 INICIANDO TREINO DO CÉREBRO: {nome_plugin}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = qwen_base, max_seq_length = max_seq_length,
        dtype = None, load_in_4bit = True,
    )
    
    model = FastLanguageModel.get_peft_model(
        model, r = 32, lora_alpha = 64, lora_dropout = 0, bias = "none",
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "down_proj", "up_proj", "gate_proj"],
        use_gradient_checkpointing = "unsloth", random_state = 3407,
    )
    
    dataset = load_dataset("json", data_files=caminho_dataset_hibrido, split="train")
    
    trainer = SFTTrainer(
        model = model, tokenizer = tokenizer, train_dataset = dataset,
        dataset_text_field = "output", max_seq_length = max_seq_length, dataset_num_proc = 2,
        args = TrainingArguments(
            per_device_train_batch_size = 2, gradient_accumulation_steps = 4,
            warmup_steps = 10, max_steps = 400, # Treino focado para o Adaptador
            learning_rate = 2e-5, optim = "adamw_8bit",
            fp16 = not is_bfloat16_supported(), bf16 = is_bfloat16_supported(),
            output_dir = f"./outputs_{nome_plugin}",
        ),
    )
    
    trainer.train()
    
    print(f"\\n🗜️ SALVANDO NO FORMATO C++ GGUF: {nome_plugin}...")
    model.save_pretrained_gguf(f"./MicroCerebro_{nome_plugin}", tokenizer, quantization_method = "q4_k_m")
    
    # Limpeza brutal de matriz térmica para não destruir a RAM entre LoRAS
    del model; del trainer; gc.collect(); torch.cuda.empty_cache()
    print(f"✅ {nome_plugin} CONCLUÍDO E EMBALADO!\n")

# 4. START DO TREINAMENTO MULTIPLO
plugins = [
    ("Python_DNA", "dataset_P_Hibrido.jsonl"),
    ("Medicina_DNA", "dataset_M_Hibrido.jsonl")
]

for cer in plugins:
    treinar_modulo_cerebral(cer[0], cer[1])

print("🎉 A FÁBRICA CONCLUIU! Baixe as pastas 'MicroCerebro_XXX' inteiras desta aba lateral!")
```
