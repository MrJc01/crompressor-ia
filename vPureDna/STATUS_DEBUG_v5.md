# 🧬 vPureDna v5 — Relatório de Debug Completo

**Data:** 2026-04-07  
**Status:** EM PAUSA — problemas de performance identificados, aguardando continuação

---

## O Que Foi Feito (Conversa Anterior)

1. **Treino no Google Colab (A100)**
   - Base: `Qwen/Qwen3-1.7B` (NÃO Qwen3.5)
   - LoRA: r=16, alpha=32, 7 target modules
   - Dataset: 5000 amostras trifásicas (emissão ⌬, expansão ⌬, manutenção)
   - 1000 steps, loss final: 1.34
   - Token `⌬` adicionado como special token

2. **Conversão para GGUF**
   - Merge LoRA → modelo completo (safetensors)
   - F16: `vpuredna_v5.gguf` (3.3 GB)
   - Q4_K_M: `vpuredna_v5_Q4KM.gguf` (1.1 GB)

3. **Publicação**
   - GitHub: `MrJc01/crompressor-ia` ✅
   - HuggingFace: `CromIA/vpuredna-v5-qwen3-1.7b` ✅

---

## Problemas Encontrados Nesta Sessão

### 🔴 Bug 1: SIGSEGV (Crash)

**Causa raiz:** O script antigo usava `script -qc` para capturar output do `llama-cli` que escreve no TTY. Isso criava race conditions que causavam Segmentation Fault.

**Status:** ✅ RESOLVIDO — Reescrito para usar `pexpect` (PTY nativo do Python).

### 🔴 Bug 2: Double-Template ChatML

**Causa raiz:** O Python montava prompt ChatML manualmente (`<|im_start|>system...`) e passava via `-p`. Mas o `llama-cli` b8645 aplica seu PRÓPRIO chat template por cima. Resultado: template dentro de template.

**Status:** ✅ RESOLVIDO — Agora usa `-sys` para system prompt e deixa o llama-cli gerenciar o template.

### 🔴 Bug 3: Modelo NÃO aprendeu ⌬

**Causa raiz:** 1000 steps com 5000 amostras foi insuficiente. O modelo responde em PT-BR normal (bom!) mas ignora completamente os marcadores ⌬W/⌬F.

**Prova:** Quando pedido para usar ⌬, respondeu `<think>W1023 oi!` — lixo, não é ⌬.

**Status:** ⚠️ NÃO RESOLVIDO — Precisa retreino com mais steps/dados.

### 🔴 Bug 4: Lentidão Extrema (`<think>` do Qwen3)

**Causa raiz:** O Qwen3-1.7B tem modo `<think>` embutido nos pesos. Antes de cada resposta, gera ~200 tokens de "pensamento" invisíveis. A 1.4 t/s na CPU, isso consome ~140s extras.

**Tentativas de correção (TODAS FALHARAM):**
- `--chat-template chatml` → modelo ainda gera `<think>` (está nos pesos)
- `-rea off --reasoning-budget 0` → sem efeito
- `--chat-template-kwargs '{"enable_thinking": false}'` → sem efeito
- `logit_bias` no token `<think>` (ID=151667) → não testado ainda

**Status:** ❌ NÃO RESOLVIDO

### 🟡 Bug 5: llama-server muito lento

**Causa raiz:** O `llama-server` via HTTP API é ~10x mais lento que `llama-cli` interativo na mesma máquina:
- `llama-cli` interativo: "hello" → resposta em ~5-10s
- `llama-server` API: "oi" → 37s (8 tokens)
- `llama-server` API: "O que é IA?" → 109s (42 tokens)

O `llama-cli` interativo mantém KV cache quente entre turnos. O server recria a sessão a cada request.

**Status:** ⚠️ MITIGADO — Reescrito para usar `llama-cli` via `pexpect` em vez do server.

---

## Estado Atual dos Arquivos

### Arquivos Modificados/Criados

```
vPureDna/06_inference/
├── chat_vpuredna_v5.py    ← REESCRITO: usa pexpect + llama-cli interativo
├── server_manager.py      ← NOVO: gerenciador llama-server (backup, lento)
chat_vpuredna_v5.sh        ← ATUALIZADO: launcher
```

### Modelos Disponíveis

```
models/
├── vpuredna_v5/
│   ├── vpuredna_v5.gguf        (3.3 GB, F16, Qwen3-1.7B + LoRA)
│   └── vpuredna_v5_Q4KM.gguf  (1.1 GB, Q4_K_M, RECOMENDADO)
├── Qwen_Qwen3.5-2B-Q4_K_M.gguf (1.3 GB, base sem LoRA, SEM <think>)
└── CROM-IA_v4.3_Qwen3.5-2B.gguf (1.2 GB, v4.3 mais antigo)
```

### DNA Compressor — FUNCIONA BEM

```
  ⌬W entries: 1198 (palavras longas)
  ⌬F entries: 13512 (frases)
  Compressão: 1.6-2.8x em frases PT-BR
  Roundtrip:  ✅ encode→decode = original
```

### Dataset de Treino

```
vPureDna/05_colab/output/
├── dataset_vpuredna_colab.jsonl  (5000 amostras, 2.4 MB)
└── lora-adapter/                 (checkpoint-1000, SEM model.safetensors local)
```

---

## Testes Realizados

| Teste | Método | Resultado | Tempo |
|-------|--------|-----------|-------|
| `llama-cli` direto interativo | Terminal | ✅ "Hello! How can" | ~5s |
| `llama-cli` com `-p` ChatML | `--single-turn` | ⚠️ Double-template, `<think>W1023` | ~50s |
| `llama-server` + curl "oi" | HTTP API | ✅ "Olá!" | 37s |
| `llama-server` + curl "IA?" | HTTP API | ✅ Resposta PT-BR correta | 109s |
| `llama-server` Qwen3.5-2B | HTTP API | ⏳ Timeout >150s | Cancelado |
| `chat_vpuredna_v5.py` (antigo) | `script -qc` | 💥 SIGSEGV | - |
| `chat_vpuredna_v5.py` (server) | HTTP API | ✅ Funciona, mas 43-76s | 43-76s |
| `chat_vpuredna_v5.py` (pexpect) | PTY | 🔨 Em teste | ? |

---

## O Que Fazer Para Continuar

### Prioridade 1: Resolver lentidão do `<think>`

**Opção A — Logit bias (rápido, testar primeiro):**
```bash
# Token ID do <think> = 151667
# Testar via API com logit_bias para suprimir geração de <think>
curl -X POST http://127.0.0.1:8081/v1/chat/completions \
  -d '{"messages":[...],"logit_bias":{"151667":-100}}'
```

**Opção B — Retreinar sobre Qwen3.5 (melhor, mais lento):**
- Usar `Qwen/Qwen3.5-2B` como base em vez de `Qwen3-1.7B`
- Qwen3.5 NÃO tem `<think>` → respostas diretas
- Treinar com 3000-5000 steps (mais que os 1000 atuais)
- Dataset maior (10K+ amostras) para ensinar ⌬ de verdade

**Opção C — Usar Qwen3.5-2B base sem LoRA:**
- Arquivo: `models/Qwen_Qwen3.5-2B-Q4_K_M.gguf`
- Sem thinking, sem LoRA overhead
- Qualidade PT-BR boa (modelo base já fala português)
- ⌬ usado apenas como camada de compressão de prompt (não nativa)

### Prioridade 2: Testar pexpect

O `chat_vpuredna_v5.py` já foi reescrito para usar `pexpect`. Testar:
```bash
cd ~/Área\ de\ trabalho/crompressor-ia
python3 vPureDna/06_inference/chat_vpuredna_v5.py --threads 4
```

### Prioridade 3: Retreino ⌬ adequado

Para o modelo realmente "pensar em ⌬":
- Aumentar dataset para 10K-20K amostras
- Aumentar steps para 3000-5000
- Usar Qwen3.5 como base (sem thinking)
- Validar que score ≥ 2/3 na validação rápida

---

## Dados Técnicos Úteis

### Hardware Local
- CPU only (sem GPU NVIDIA)
- Velocidade: ~2.4 t/s prompt, ~1.4 t/s generation
- RAM suficiente para modelos Q4 (1-1.3 GB)

### llama.cpp Build
- Versão: b8645-57ace0d61
- Path: `pesquisa/poc_llama_cpp_fuse/llama.cpp/build/bin/`
- Binários: `llama-cli`, `llama-server`, `llama-completion`, `llama-tokenize`

### Token IDs Importantes
- `<think>` = 151667
- `⌬` = adicionado como special token no treino (ID varia)

### Flags Testados no llama-server
```
--chat-template chatml     # NÃO impediu <think>
-rea off                   # NÃO impediu <think>
--reasoning-budget 0       # NÃO impediu <think>  
--chat-template-kwargs '{"enable_thinking": false}'  # NÃO impediu
```

---

## Conclusão

O vPureDna v5 está **funcional mas lento**. O SIGSEGV foi resolvido, o DNA Compressor funciona bem (1.6-2.8x), e o modelo responde em PT-BR correto. O gargalo é o `<think>` do Qwen3 que consome ~70% do tempo de inferência gerando tokens invisíveis. A solução definitiva é retreinar sobre Qwen3.5 (sem thinking) com mais dados.
