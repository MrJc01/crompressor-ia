# Tradutor de Dataset "OpenHermes" Otimizado para GPU (Nvidia A100/A10G)

Este script visa resolver a etapa pesada de tradução bloqueada no ambiente local (OOM de CPU). Rodando no Google Colab com pacote _transformers_, a extração de linguagem será exponencialmente mais rápida.

## 0. Baixando o Dataset OpenHermes 10K Diretamente da Nuvem
Como a máquina local teve o download interrompido, baixe em segundos a base bruta (em Inglês) no seu Colab, executando esta célula:
```python
import json
from datasets import load_dataset
from tqdm import tqdm

print("📥 Extraindo OpenHermes (Cloud Speed)...")
ds = load_dataset('teknium/OpenHermes-2.5', split='train')

candidatos = []
print("🔍 Filtrando as 10.000 melhores conversas ricas (Isso leva uns ~3 minutinhos)...")
for item in tqdm(ds, desc="Analisando 1 Milhão de Amostras"):
    conversations = item.get('conversations', [])
    if len(conversations) < 2: continue
    
    instruction = ""
    output = ""
    for msg in conversations:
        role = msg.get('from', msg.get('role', ''))
        value = msg.get('value', msg.get('content', ''))
        if role in ('human', 'user'): instruction = value
        elif role in ('gpt', 'assistant'): output = value

    if not instruction or not output: continue
    if len(output) >= 200:
        candidatos.append({'instruction': instruction, 'output': output, 'length': len(output)})

candidatos.sort(key=lambda x: x['length'], reverse=True)
candidatos = candidatos[:10000]

with open("openhermes_10k_en.jsonl", 'w', encoding='utf-8') as f:
    for item in candidatos:
        f.write(json.dumps({'instruction': item['instruction'], 'output': item['output']}, ensure_ascii=False) + '\n')
print("✅ Dataset openhermes_10k_en.jsonl gerado no Colab!")
```

## 1. Instalando Dependências do CROM no Colab
Rode esta primeira célula no notebook:
```python
!pip install -q transformers accelerate datasets tqdm sacremoses
```

## 2. Ingestor e Pipeline Neural de Tradução Seq2Seq
Faça o upload do seu dataset local (`openhermes_10k_en.jsonl`) nos arquivos do Google Colab. Em seguida, crie uma nova célula com o script:

```python
import json
import re
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

print("🚀 Iniciando Motor de Tradução na Nuvem (GPU-Accellerated)")
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Verificando Hardware: ", "🟢 GPU Ativada" if device == "cuda" else "🔴 CPU (Lento!)")

# 1. Carregador Nativo em FP16 (Tensor Cores Acionados p/ Dobro de Velocidade)
model_name = "Helsinki-NLP/opus-mt-tc-big-en-pt"
print(f"📥 Baixando modelo {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float16,  # Ativa o FP16 Cortando a VRAM pela METADE!
).to(device)

INPUT_FILE = "openhermes_10k_en.jsonl"
OUTPUT_FILE = "openhermes_10k_ptbr.jsonl"

def split_and_protect_code(text):
    """ Protege marcadores markdown inteiros antes de tacar na AI de tradução """
    parts = re.split(r'(```.*?```)', text, flags=re.DOTALL)
    return parts

# 2. Carrega Memória e Processamento em Batch Dinâmico (Explosão de GPU)
dataset = []
with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        dataset.append(json.loads(line))

print(f"📦 Sucesso: {len(dataset)} sentenças enviadas para memória.")

def translate_batch(texts):
    """Traduz micro-lotes via Tensor Cores (FP16)"""
    if not texts: return []
    res = []
    MINI_BATCH = 64 # Quatro vezes mais matrizes em paralelo!
    
    for i in range(0, len(texts), MINI_BATCH):
        chunk = texts[i:i+MINI_BATCH]
        inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=512)
        res.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
        
        # O Pytorch gerencia o cache automaticamente sem congelar a GPU
        del inputs, outputs
        
    return res

BATCH_SIZE = 64  # Engolindo 64 amostras (até centenas de strings) por vez!
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
    
    for i in tqdm(range(0, len(dataset), BATCH_SIZE), desc="Injerindo Lotes na A100", unit="lote"):
        batch_items = dataset[i:i+BATCH_SIZE]
        
        # 1. Enfileirar requisições para a GPU
        queries = []
        strukt = []
        
        for item in batch_items:
            inst_parts = split_and_protect_code(item.get('instruction', ''))
            out_parts = split_and_protect_code(item.get('output', ''))
            
            struct_item = {'inst': [], 'out': []}
            
            # Mapeamento do Instruction
            for p in inst_parts:
                if p.startswith('```') or not p.strip():
                    struct_item['inst'].append(p)
                else:
                    struct_item['inst'].append(len(queries)) # Guarda o Índice
                    queries.append(p)
                    
            # Mapeamento do Output
            for p in out_parts:
                if p.startswith('```') or not p.strip():
                    struct_item['out'].append(p)
                else:
                    struct_item['out'].append(len(queries)) # Guarda o Índice
                    queries.append(p)
                    
            strukt.append(struct_item)
            
        # 2. Fogo na Bomba (A100 entra Aqui!)
        translated_queries = translate_batch(queries) if queries else []
        
        # 3. Remontar o JSON original já traduzido
        for idx, s in enumerate(strukt):
            new_inst = ""
            for frag in s['inst']:
                if isinstance(frag, int): # Se era índice, pega do cache traduzido
                    new_inst += translated_queries[frag] + " "
                else: # Se era markdown (código), junta intacto
                    new_inst += frag
            
            new_out = ""
            for frag in s['out']:
                if isinstance(frag, int):
                    new_out += translated_queries[frag] + " "
                else:
                    new_out += frag
                    
            system_msg = "Você é CROM-IA, um assistente inteligente especializado em raciocínio."
            texto_chatml = (
                f"<|im_start|>system\n{system_msg}<|im_end|>\n"
                f"<|im_start|>user\n{new_inst.strip()}<|im_end|>\n"
                f"<|im_start|>assistant\n{new_out.strip()}<|im_end|>"
            )

            f_out.write(json.dumps({"text": texto_chatml}, ensure_ascii=False) + '\n')

print(f"\\n✅ Tradução perfeitamente concluída! Baixe o arquivo: {OUTPUT_FILE}")
```

### O que fazer depois?
- Faça o Download do arquivo JSONL gerado pelo Colab.
- Mova ele para a pasta local (`v4.2_multibrain_engine/1_extracao_local/datasets_hibridos/`).
- Siga suas scripts normais de concatenação de Dataset, e Geração de DPO.
