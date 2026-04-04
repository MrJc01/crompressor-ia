# CROM-IA V4.1 — Multi-Brain DNA Engine (Corrigido)
> **Status:** Pronto para re-treino
> **Base Model:** Qwen3.5-0.8B (`Qwen/Qwen3.5-0.8B`)
> **Mudança Central:** LoRAs Empilháveis + DNA Conservador + Datasets Reais

---

## Lições Aprendidas da V4.1-alpha (Treino Falho)

### O que DEU ERRADO:
| Erro | Valor usado | Valor correto |
|---|---|---|
| DNA mutação | 75% (destruiu coerência) | **25%** máximo |
| Steps | 2000 (overfitting) | **500-800** |
| Rank LoRA | 64 (reescreveu demais o modelo) | **16-32** |
| Datasets conversa | 15 templates repetidos | **30K+ do HuggingFace** |
| Modelo base | Qwen3-0.6B (pouca capacidade) | **Qwen3.5-0.8B** |
| Épocas CROM_Self | 133 (memorizou) | **máximo 10** |

### O que DEU CERTO:
- ✅ Velocidade: 7-9 t/s (2x mais rápido que V4.0)
- ✅ RAM: 635MB (metade da V4.0)
- ✅ DNA ativo: tokens `@@PWAT`, `@@PWC` apareceram na saída
- ✅ Pipeline completo funciona (extração → codebook → treino → deploy)
- ✅ Codebook data-driven por frequência real (Crompressor)

---

## Arquitetura V4.1 Final (Corrigida)

### Modelo Base
- **Qwen3.5-0.8B** — https://huggingface.co/Qwen/Qwen3.5-0.8B
- Unsloth: `unsloth/Qwen3.5-0.8B-bnb-4bit` (verificar disponibilidade)
- Fallback: quantizar manualmente com `BitsAndBytesConfig`

### Parâmetros de Treino (Conservadores)
```python
# LoRA
r = 16                           # Rank conservador (era 64)
lora_alpha = 32                  # 2x o rank
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]  # Sem MLP

# Treino
max_steps = 800                  # Era 2000
per_device_train_batch_size = 8  # Era 16
gradient_accumulation_steps = 4
learning_rate = 1e-5             # Era 2e-5 (mais conservador)
lr_scheduler_type = "cosine"
warmup_ratio = 0.05
```

### Datasets (Qualidade Real)
| Cérebro | Fonte | Amostras | DNA % |
|---|---|---|---|
| **Base_PTBR** | `dominguesm/Canarim-Instruct-PTBR` | 30.000 | 0% (puro) |
| **Python** | `Vezora/Tested-22k-Python-Alpaca` + nosso | 15.000 | 25% |
| **Medicina** | Nosso dataset V4.0 (8K) | 8.000 | 25% |
| **CROM_Self** | Docs .md do projeto | 500 | 0% |
| **Destilação_GPT4** | `teknium/OpenHermes-2.5` (10K traduzidos) | 10.000 | 0% |

**REGRA:** Nenhum dataset com menos de 1.000 amostras únicas.
**REGRA:** DNA máximo 25%. O modelo deve PRIMEIRO saber falar, DEPOIS usar DNA.
**REGRA:** Máximo 10 épocas por dataset.

### Estratégia de Treino: 2 Fases

**Fase 1 — Base Conversacional (SEM DNA)**
Treinar um LoRA de "personalidade" com 30K do Canarim + 10K OpenHermes traduzido.
O modelo aprende a conversar bem em PT-BR primeiro.

**Fase 2 — Especialização DNA (COM DNA a 25%)**
Treinar LoRAs especializados com DNA sutil sobre a base já conversacional.
Os LoRAs de Python e Medicina são empilhados sobre o base.

### Inferência: LoRA Stacking
```bash
# Base + Personalidade + Especialização
llama-cli -m qwen3.5-0.8b-base.gguf \
    --lora Base_PTBR_lora.gguf \
    --lora Python_DNA_lora.gguf \
    --lora Medicina_DNA_lora.gguf \
    --conversation
```

---

## Estrutura de Diretórios

```
v4.1_multibrain_engine/
├── 00_CROM_IA_V4.1_DOCUMENTATION.md    ← Este arquivo
├── 1_extracao_local/
│   ├── codebooks/                       ← Codebooks data-driven
│   ├── datasets_hibridos/               ← Datasets transpilados
│   ├── gerador_codebook_v41.py          ← Mineração por frequência real
│   ├── gerador_datasets_v41.py          ← Auto-conhecimento + conversa
│   └── transpilador_v41.py              ← Taxa DNA configurável
├── 2_treinamento_nuvem/
│   ├── 01_CROM_V41_TRAINING_CODE.py     ← Código Colab (ATUALIZAR!)
│   └── arquivos_para_colab/             ← Datasets para upload
├── 3_inferencia_local/
│   ├── chat_v41_stacked.sh              ← Inferência empilhada
│   ├── decodificador_dna/
│   │   └── decodificador_dna.py         ← Traduz @@tokens → palavras
│   └── micro_cerebros/                  ← LoRAs GGUF empilháveis
└── adicionar_cerebro.py                 ← Adicionar cérebro em 1 comando
```

---

## Checklist para Próxima Sessão

### Preparação (Local)
- [ ] Baixar Canarim-PTBR do HuggingFace (30K)
- [ ] Traduzir OpenHermes-2.5 top 10K (Argos Translate)
- [ ] Re-transpilar datasets com DNA a 25%
- [ ] Gerar novos codebooks data-driven
- [ ] Verificar se existe `unsloth/Qwen3.5-0.8B-bnb-4bit`

### Treino (Colab)
- [ ] Fase 1: Base_PTBR (30K Canarim, 0% DNA, 800 steps, rank 16)
- [ ] Fase 2: Python_DNA (15K, 25% DNA, 500 steps, rank 16)
- [ ] Fase 2: Medicina_DNA (8K, 25% DNA, 500 steps, rank 16)
- [ ] Salvar adaptadores PEFT separados
- [ ] Converter PEFT → GGUF-LoRA com llama.cpp

### Deploy (Local)
- [ ] Baixar Qwen3.5-0.8B base GGUF
- [ ] Baixar LoRAs convertidos
- [ ] Testar `--lora` stacking no llama-cli
- [ ] Benchmark: velocidade + qualidade
