# CROM-IA V4.2 — Multi-Brain DNA Engine + DPO + RAG-Chat
> **Status:** Em construção
> **Base Model:** Qwen3-0.6B (`unsloth/Qwen3-0.6B-unsloth-bnb-4bit`)
> **Mudanças vs V4.1:** DPO, Chat com ingestão de arquivos, DNA conservador 25%, Datasets reais

---

## Evolução do Projeto

| Versão | Modelo | Velocidade | RAM | DNA | Resultado |
|---|---|---|---|---|---|
| V4.0 | Qwen2.5-1.5B | 3-5 t/s | ~1.2GB | 50% | ✅ Funcional mas lento |
| V4.1-α | Qwen3-0.6B | 7-9 t/s | 635MB | 75% | ❌ Catastrophic forgetting |
| **V4.2** | **Qwen3-0.6B** | **7-9 t/s** | **635MB** | **25%** | **🔧 Em construção** |

---

## Lições Aprendidas (V4.1-alpha → V4.2)

### O que DESTRUIU a V4.1:
| Erro | V4.1 (errado) | V4.2 (corrigido) |
|---|---|---|
| DNA mutação | 75% (destruiu coerência) | **25%** máximo |
| Steps | 2000 (overfitting) | **500-800** |
| Rank LoRA | 64 (reescreveu o modelo inteiro) | **16** |
| Target modules | q,k,v,o + gate,down,up (MLP!) | **q,k,v,o** (só attention) |
| Datasets | 15 templates repetidos | **30K+ do HuggingFace** |
| Épocas CROM_Self | 133 (memorizou) | **máximo 10** |
| Learning rate | 2e-5 | **1e-5** (mais suave) |
| LR scheduler | Linear | **Cosine** (convergência melhor) |

### O que DEU CERTO (mantemos):
- ✅ Pipeline completo: extração → codebook → transpilação → treino → deploy
- ✅ Velocidade: 7-9 t/s no i5-3320M (2x V4.0)
- ✅ RAM: 635MB (metade da V4.0)
- ✅ DNA ativo: tokens `@@PWAT`, `@@PWC` apareceram na saída
- ✅ Codebook data-driven por frequência real (filosofia Crompressor)
- ✅ Script `adicionar_cerebro.py` — adiciona cérebro em 1 comando

---

## Arquitetura V4.2

### Modelo Base
- **Qwen3-0.6B** — Velocidade é prioridade no i5 sem GPU
- Unsloth: `unsloth/Qwen3-0.6B-unsloth-bnb-4bit`
- GGUF: Q4_K_M (~379MB)

### Parâmetros de Treino (Conservadores)
```python
# LoRA
r = 16                           # Rank (era 64)
lora_alpha = 32                  # 2x rank
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]  # SEM MLP!

# SFT (Fases 1 e 2)
max_steps = 800                  # Fase 1 / 500 para Fase 2
per_device_train_batch_size = 8
gradient_accumulation_steps = 4
learning_rate = 1e-5
lr_scheduler_type = "cosine"
warmup_ratio = 0.05

# DPO (Fase 3)
beta = 0.1                       # Força da preferência
max_steps = 300
learning_rate = 5e-6             # Mais suave que SFT
```

### Datasets (Qualidade > Quantidade)
| Cérebro | Fonte | Amostras | DNA % | Fase |
|---|---|---|---|---|
| **Base_PTBR** | Canarim 30K + OpenHermes 10K trad. | 40.000 | 0% | Fase 1 |
| **Python_DNA** | `Vezora/Tested-22k-Python-Alpaca` | 15.000 | 25% | Fase 2 |
| **Medicina_DNA** | Dataset V4.0 + nosso | 8.000 | 25% | Fase 2 |
| **CROM_Self** | Docs .md do projeto | 500 | 0% | Fase 2 |
| **DPO_Pares** | Gerado automaticamente | 5.000 | chosen=DNA | Fase 3 |

**REGRAS:**
- DNA máximo 25%. O modelo PRIMEIRO sabe falar, DEPOIS usa DNA.
- Máximo 10 épocas por dataset.
- Filtros de qualidade: resposta > 100 chars, sem duplicatas.

---

## Estratégia de Treino: 3 Fases

### Fase 1 — SFT Base Conversacional (SEM DNA)
Treinar LoRA de "personalidade" com 40K conversas reais.
O modelo aprende a conversar bem em PT-BR primeiro.
```
Dataset: Canarim 30K + OpenHermes 10K (traduzido)
DNA: 0%
Steps: 800
Output: Base_PTBR_lora
```

### Fase 2 — SFT Especialização DNA (25% DNA)
Treinar LoRAs especializados com DNA sutil sobre a base conversacional.
```
Dataset Python: 15K com 25% DNA → Python_DNA_lora
Dataset Medicina: 8K com 25% DNA → Medicina_DNA_lora
Steps: 500 cada
```

### Fase 3 — DPO (Direct Preference Optimization)
O modelo aprende a PREFERIR respostas com DNA sobre texto normal.
```python
# Par DPO
{
  "prompt": "Explique arritmia cardíaca",
  "chosen": "Uma @@DGN de @@CRC onde o ritmo...",   # DNA = preferido
  "rejected": "Um diagnóstico de coração onde..."     # Normal = rejeitado
}
```
- Usa `trl.DPOTrainer`
- 5K pares gerados automaticamente pelo `gerador_pares_dpo.py`
- Resultado: modelo usa DNA quando oportuno, sem forçar

---

## Inferência: Monitor de Orquestração + RAG

### O Monitor TUI
O `chat_v42_brain.sh` abre um **painel interativo** onde você configura tudo ANTES de iniciar o chat:

```
╔══════════════════════════════════════════════════════════════╗
║       🧠 CROM-IA V4.2 — Monitor de Orquestração           ║
╠══════════════════════════════════════════════════════════════╣
║  Configure seus cérebros e contexto antes de iniciar       ║
╚══════════════════════════════════════════════════════════════╝

── Modelo Base ─────────────────────────────────────────────
   ✅ qwen3-0.6b-q4_k_m.gguf (379MB)

── Micro-Cérebros (LoRAs) ──────────────────────────────────
   [1] ✅ ON  Base_PTBR (32MB)
   [2] ✅ ON  Python_DNA (28MB)
   [3] ⬚ OFF Medicina_DNA (30MB)     ← desativado!

── Contexto RAG (Arquivos/Pastas) ──────────────────────────
   📄 main.py
   📂 ./src/ (15 arquivos)

── Ações ───────────────────────────────────────────────────
   [1-9]  Toggle cérebro ON/OFF
   [a]    Adicionar arquivo     [p] Adicionar pasta
   [r]    Remover último RAG    [c] Limpar RAG
   [t]    Temperatura           [w] Janela de contexto
   [*]    Ativar TODOS          [0] Desativar TODOS
   [ENTER] 🚀 INICIAR CHAT
   [q]     Sair
```

### Uso
```bash
# Abrir monitor (interativo)
./chat_v42_brain.sh

# Pré-carregar arquivos e abrir monitor
./chat_v42_brain.sh --arquivo main.py --pasta ./src/
```

### Como Funciona
1. **Monitor TUI:** Painel interativo para orquestrar cérebros e contexto
2. **Toggle:** Ativa/desativa cérebros individuais com tecla numérica
3. **RAG-lite:** Adiciona arquivos/pastas que são lidos e injetados no prompt
4. **Config:** Ajusta temperatura, contexto, max tokens na hora
5. **Launch:** ENTER lança o chat com a config escolhida
6. **Retorno:** Ctrl+C no chat volta ao monitor para reconfigurar

---

## Estrutura de Diretórios V4.2

```
v4.2_multibrain_engine/
├── 00_CROM_IA_V4.2_DOCUMENTATION.md      ← Este arquivo
├── 00_CROM_IA_V4.2_ROADMAP.md            ← Roadmap original (referência)
├── 01_DPO_GUIDE.md                        ← Guia DPO detalhado
├── 02_CHAT_RAG_GUIDE.md                   ← Guia do Monitor + RAG
├── 1_extracao_local/
│   ├── codebooks/                          ← Codebooks data-driven
│   ├── datasets_hibridos/                  ← Datasets prontos para Colab
│   ├── download_datasets_v42.py            ← Baixa HuggingFace
│   ├── tradutor_batch_argos.py             ← Traduz EN→PT offline
│   ├── transpilador_v42.py                 ← DNA a 25% (era 75%)
│   ├── gerador_codebook_v42.py             ← Mineração por frequência
│   └── gerador_pares_dpo.py                ← Gera pares DPO automáticos
├── 2_treinamento_nuvem/
│   ├── 01_CROM_V42_TRAINING_FASE1.py       ← SFT Base (40K, 0% DNA)
│   ├── 02_CROM_V42_TRAINING_FASE2.py       ← SFT DNA (23K, 25% DNA)
│   ├── 03_CROM_V42_DPO_TRAINING.py         ← DPO (5K pares, preferência)
│   ├── adapters_lora/                      ← LoRAs PEFT do Colab
│   └── colab/                              ← Notebooks prontos
├── 3_inferencia_local/
│   ├── chat_v42_brain.sh                   ← 🎛️ MONITOR TUI + Chat
│   ├── rag_contexto.py                     ← Motor RAG-lite (sem GPU)
│   ├── decodificador_dna/
│   │   └── decodificador_dna.py            ← Traduz @@tokens → palavras
│   └── micro_cerebros/                     ← LoRAs GGUF empilháveis
└── adicionar_cerebro.py                    ← Adicionar cérebro em 1 cmd
```

---

## Checklist de Execução

### Preparação (Local)
- [ ] Instalar argostranslate (`pip install argostranslate`)
- [ ] Baixar Canarim-PTBR 30K do HuggingFace
- [ ] Baixar Python Alpaca 15K
- [ ] Traduzir OpenHermes-2.5 top 10K (Argos)
- [ ] Gerar codebooks data-driven para novos datasets
- [ ] Transpilar datasets com DNA a 25%
- [ ] Gerar pares DPO (5K)

### Treino (Colab)
- [ ] Fase 1: Base_PTBR (40K, 0% DNA, 800 steps, rank 16)
- [ ] Fase 2: Python_DNA (15K, 25% DNA, 500 steps, rank 16)
- [ ] Fase 2: Medicina_DNA (8K, 25% DNA, 500 steps, rank 16)
- [ ] Fase 3: DPO (5K pares, 300 steps, rank 16)
- [ ] Converter PEFT → GGUF-LoRA com llama.cpp

### Deploy (Local)
- [ ] Baixar Qwen3-0.6B base GGUF
- [ ] Baixar LoRAs convertidos
- [ ] Testar chat_v42_brain.sh sem arquivos
- [ ] Testar chat_v42_brain.sh --arquivo test.py
- [ ] Testar chat_v42_brain.sh --pasta ./projeto/
- [ ] Testar LoRA stacking (2+ LoRAs)
- [ ] Benchmark: velocidade + qualidade + DNA %

---

## Hardware Alvo
- **CPU:** Intel i5-3320M @ 2.60GHz (4 threads)
- **RAM:** 7.4GB total
- **GPU:** Nenhuma
- **Disco:** ~25GB livres
- **llama-cli:** `/home/j/Área de trabalho/crompressor-ia/pesquisa/poc_llama_cpp_fuse/llama.cpp/build/bin/llama-cli`
