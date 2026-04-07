# 🧬 vPureDna — RELATÓRIO FINAL + GUIA COLAB

**Data:** 07/Abril/2026  
**Autor:** CROM-IA Lab  
**Status:** ✅ Pronto para Colab

---

## 📊 RESUMO EXECUTIVO

Validamos que é **viável treinar o Qwen3.5-2B via LoRA** para usar `⌬` como 
marcador de compressão inteligente. O modelo aprenderá a emitir `⌬F42` em vez 
de "inteligência artificial", e depois novos "cérebros" (codebooks) poderão ser
adicionados em runtime sem retreinar.

### Resultados dos Testes

| # | Teste | Resultado | Detalhe |
|---|-------|-----------|---------|
| 1 | Regra de ouro | ✅ | "oi"→"oi", frases longas→⌬F (até 2.8x) |
| 2 | Cobertura PT-BR | ✅ | 1.69x compressão média, 35% dos tokens viram ⌬ |
| 3 | Round-trip | ✅ | 8/8 textos 100% corretos |
| 4 | Edge cases | ✅ | URLs, emojis, acentos, código — tudo OK |
| 5 | Smoke test neural | ✅ | Loss 4.33→1.53 (-65%), modelo emite ⌬ |
| 6 | Threshold | ✅ | Palavras < 4 chars nunca viram ⌬ |
| 7 | ⌬ vs BPE Qwen | ⚠️ | ⌬ = 2 tokens BPE, resolver com special_token |

### Achado Crítico
O BPE do Qwen fragmenta `⌬` em 2 tokens e cada dígito de ID custa 1 token extra.
**Solução:** Adicionar `⌬` como `additional_special_token` no LoRA training.
O valor do ⌬ está nos **parágrafos** (10x savings) e na **modularidade Multi-Brain**,
não na eficiência BPE word-level.

---

## 🚀 COMO TREINAR NO COLAB (Passo a Passo)

### Pré-requisitos
- Conta Google com Colab (GPU T4 grátis ou A100 paga)
- Token HuggingFace com permissão de leitura (para baixar Qwen)

### Passo 1: Criar Notebook no Colab

Abra https://colab.research.google.com e crie um novo notebook.
Selecione **Runtime → Change runtime type → T4 GPU**.

### Passo 2: Instalar Dependências

Cole na primeira célula e execute:

```python
# Célula 1: Setup
!pip install -q transformers peft datasets trl accelerate bitsandbytes
!pip install -q huggingface_hub

# Login HuggingFace (para baixar Qwen)
from huggingface_hub import login
login()  # Cole seu token quando pedir
```

### Passo 3: Clonar o Repositório

```python
# Célula 2: Clonar CROM-IA
!git clone https://github.com/MrJc01/crompressor-ia.git
%cd crompressor-ia
```

### Passo 4: Executar o Treino

```python
# Célula 3: Treinar (o script faz tudo automaticamente)
!python3 vPureDna/05_colab/treino_colab_vpuredna.py
```

**O que o script faz automaticamente:**
1. Baixa Alpaca-PT do HuggingFace (52K amostras)
2. Comprime respostas com DNACompressor (⌬F, ⌬W)
3. Gera dataset trifásico (5K amostras: comprimir/expandir/normal)
4. Carrega Qwen3-1.7B em 4-bit (cabe em T4 16GB)
5. Adiciona ⌬ como special token
6. Treina com LoRA (r=16, alpha=32, 1000 steps)
7. Valida se o modelo emite ⌬ corretamente
8. Salva em `vPureDna/05_colab/output/final/`

**Tempo estimado:** ~30-60 min em T4, ~15-20 min em A100

### Passo 5: Validação Completa (Opcional)

```python
# Célula 4: Validação detalhada (5 testes)
!python3 vPureDna/04_avaliacao/validacao_pos_treino.py \
    --model-path vPureDna/05_colab/output/final
```

### Passo 6: Exportar Modelo

```python
# Célula 5: Merge LoRA + Upload
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Merge LoRA com modelo base
base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-1.7B", torch_dtype=torch.float16, device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    "vPureDna/05_colab/output/final", trust_remote_code=True,
)
base.resize_token_embeddings(len(tokenizer))
model = PeftModel.from_pretrained(base, "vPureDna/05_colab/output/final")
merged = model.merge_and_unload()

# Salvar modelo merged
merged.save_pretrained("vPureDna_merged")
tokenizer.save_pretrained("vPureDna_merged")
print("✅ Modelo merged salvo!")

# Upload para HuggingFace
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path="vPureDna_merged",
    repo_id="CromIA/CROM-IA-vPureDna",
    repo_type="model",
)
print("🎉 Upload completo!")
```

### Passo 7: Converter para GGUF (para llama-cli local)

```python
# Célula 6: GGUF conversion
!pip install -q llama-cpp-python
!git clone https://github.com/ggml-org/llama.cpp.git /tmp/llama.cpp

# Converter para GGUF
!python3 /tmp/llama.cpp/convert_hf_to_gguf.py vPureDna_merged \
    --outfile vPureDna.gguf --outtype f16

# Quantizar para Q4_K_M (~1 GB)
!cd /tmp/llama.cpp && make -j quantize
!/tmp/llama.cpp/build/bin/llama-quantize vPureDna.gguf vPureDna-Q4_K_M.gguf Q4_K_M

print("✅ GGUF pronto! Baixar vPureDna-Q4_K_M.gguf para usar localmente")
```

---

## 📁 ARQUIVOS DO PROJETO

```
vPureDna/
├── README.md                          # Visão geral
├── PLANEJAMENTO_COMPLETO.md           # Tudo documentado (decisões, testes)
├── DNA_FORMAT_SPEC.md                 # Spec do formato ⌬
├── DIARIO.md                          # Log cronológico
├── RELATORIO_FINAL_COLAB.md           # ← ESTE ARQUIVO
│
├── 01_encoder/
│   └── tokenizer_dna.py              # DNACompressor (⌬W + ⌬F)
│
├── 02_dataset/
│   ├── gerar_dataset_puro.py          # Gerador trifásico
│   └── dataset_vpuredna.jsonl         # Dataset smoke test (300 amostras)
│
├── 03_modelo/
│   ├── model.py                       # NanoGPT (smoke test only)
│   ├── train.py                       # Trainer smoke test
│   └── checkpoints/                   # Best model + log
│
├── 04_avaliacao/
│   └── validacao_pos_treino.py        # Suite de 5 testes pós-treino
│
└── 05_colab/
    └── treino_colab_vpuredna.py       # 🔥 Script Colab (LoRA + Qwen + ⌬)
```

---

## ⚠️ TROUBLESHOOTING COLAB

### "CUDA out of memory"
→ Reduzir `batch_size` de 4 para 2 no CONFIG do script

### "⌬ não aparece nas respostas"
→ Verificar que `add_special_tokens(["⌬"])` rodou antes do treino
→ Aumentar `max_steps` (tentar 2000)
→ Verificar que Fase A (emissão ⌬) está no dataset

### "Loss não converge"
→ Aumentar `max_amostras` (tentar 10000)
→ Reduzir `learning_rate` (tentar 1e-4)
→ Verificar qualidade do dataset com `head -5 dataset_vpuredna_colab.jsonl`

### "Modelo 'esqueceu' Português"
→ A Fase C (33% normal) deveria prevenir isso
→ Se persistir, aumentar Fase C para 50%

---

## 🧠 DECISÕES-CHAVE (Para Referência Rápida)

| Decisão | Escolha | Motivo |
|---------|---------|--------|
| Demarcador | `⌬` (U+232C) | Único, não conflita, identidade CROM |
| Modelo base | Qwen3-1.7B | Apache 2.0, convergiu no V4.3, PT-BR |
| Método | LoRA (não from scratch) | 2h Colab vs semanas GPU |
| Dataset | Trifásico (A/B/C) | Ensina emitir + entender + manter base |
| ⌬ no BPE | add_special_token | Resolve fragmentação (2→1 token) |
| Prioridade | ⌬P > ⌬F > ⌬W | Parágrafos comprimem 10x, palavras não |

---

*Gerado em 07/Abr/2026 — CROM-IA vPureDna Lab*
