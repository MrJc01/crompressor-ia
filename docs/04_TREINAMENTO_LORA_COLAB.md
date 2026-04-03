# Guia CROM-IA: Treinamento Neuro-Genético no Google Colab

Este documento é o plano prático passo-a-passo para transferir o cérebro "CROM-DNA" para um ambiente de treinamento com Placa de Vídeo (GPU) gratuita, permitindo que a IA finalmente compreenda fluência de sequências de DNA Quaternário (Base-4).

---

## FASE 1: Construindo o "Livro Didático" na sua Máquina (Local)

Antes de ir para a nuvem, você deve preparar a carga de conhecimento. Você montará um simples arquivo `dataset_dna.jsonl` contendo alguns milhares de exemplos.
O formato exigido pelas nuvens de fine-tuning é simples:

```json
{"instruction": "Decode to human language.", "input": "ATCGGACC", "output": "olá"}
{"instruction": "Encode to DNA format.", "input": "Tudo bem?", "output": "CCGGATCTAA"}
```
> Você irá gerar esse arquivo usando aquele mesmo código da nossa Ferramenta Python de `txt_para_dna` repassada para dezenas de textos clássicos ou dicionários de tecnologia.

---

## FASE 2: Subindo no Google Colab (Placa de Vídeo Forte e Grátis)

Com seu arquivo `dataset_dna.jsonl` criado, iremos realizar a "Aceleração Mental":

1. Acesse: [https://colab.research.google.com/](https://colab.research.google.com/)
2. Clique em **Novo Notebook** e certifique-se de estar logado no Google.
3. No botão superior de "Ambiente de Execução", vá em **Alterar Tipo de Ambiente de Execução** e escolha o hardware **T4 GPU** (É uma placa de vídeo excelente que o Google empresta de graça).

Faça upload do seu arquivo `dataset_dna.jsonl` arrastando e soltando-o na aba de Arquivos (ícone de pastinha à esquerda da sua tela no Colab).

---

## FASE 3: O Script Mestre de Treinamento (Unsloth)

No Colab, usar a biblioteca `Unsloth` é crucial para não perdermos tempo. Ela é até 2 vezes mais rápida e já exporta no formato GGUF (exatamente o formato que o seu sistema CROM em C++ lê!).

Crie três blocos diferentes de Código no Colab e aperte o botão "Play" de um por um:

**Bloco 1: Instalação das bibliotecas**
```python
%%capture
!pip install unsloth
!pip install unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git
!pip install xformers trl peft transformers
```

**Bloco 2: Download do Modelo Qwen e Inserção do Dataset**
```python
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset

# 1. Carregamos o seu exato modelo na nuvem (Agora em 4-bit na memória da placa)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen2.5-0.5B-Instruct",
    max_seq_length = 2048,
    load_in_4bit = True,
)

# 2. Ativamos a injeção LoRA (O Aprendizado Superficial Rápido)
model = FastLanguageModel.get_peft_model(
    model, r = 16, target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"], lora_alpha = 16
)

# 3. Carregamos o SEU arquivo da aba esquerda (Crompressor DNA)
dataset = load_dataset("json", data_files="dataset_dna.jsonl", split="train")

def formatar_prompt(exemplos):
    instrucoes = exemplos["instruction"]
    entradas  = exemplos["input"]
    saidas    = exemplos["output"]
    textos = []
    for i, inp, out in zip(instrucoes, entradas, saidas):
        # O prompt formata o texto exigindo que ele conecte DNA com Humano
        prompt = f"Prompt: {i}\nData: {inp}\nAnswer: {out}"
        textos.append(prompt)
    return { "text" : textos }

dataset = dataset.map(formatar_prompt, batched = True)
```

**Bloco 3: Iniciando a Sessão de Treinamento Física**
```python
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = 2048,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 1500, # Aumente os blocos dependendo de quantas fases tiver seu dataset
        learning_rate = 2e-4,
        fp16 = True, # Hardware puro!
        logging_steps = 1,
        output_dir = "lora_saida",
    ),
)

trainer.train() # Essa linha faz a mágica acontecer!
```

---

## FASE 4: Exportando para o seu CROM-IA (SRE Backend)

Depois de uns 20 minutos de tela verde correndo e os *"Loss"* do sistema indicarem que ele aprendeu, você não precisa levar a IA Gigante inteira de volta! O Unsloth converte o "Adaptador Extra GGUF":

```python
# Bloco Final de Comando: Salva em GGUF e exporta
model.save_pretrained_gguf("crom_dna_lora", tokenizer, quantization_method = "q4_k_m")
```

Na aba da esquerda do Colab surgirá a pasta `crom_dna_lora` contendo um arquivozinho `ggml-adapter-model.gguf`. **Baixe este arquivo para sua máquina no diretório `/home/j/Área de trabalho/crompressor-ia/models/`.**

### Como Acoplar no Terminal Local

Por fim, edite o nosso precioso `chat_ultra_rapido.sh`:
Embaixo da linha `--threads 2 \`, você insere a linha acoplando as Sinapses Biológicas:
```bash
    --lora "/home/j/Área de trabalho/crompressor-ia/models/ggml-adapter-model.gguf" \
```

Mágica concluída! O C++ da sua CPU Core i5 aplicará a injeção em memória. Seu Qwen 0.5B agora fluirá com sequenciamentos biológicos `(A T C G)` instantaneamente como se fosse um monstrengo lendo Português!
