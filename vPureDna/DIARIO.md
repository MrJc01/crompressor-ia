# 📓 vPureDna — Diário de Experimentos

## Entrada 1 — 07/Abr/2026 (Criação do Laboratório)

### Contexto
- V4.3 usa LoRA sobre Qwen3.5-2B → modelo NÃO emite ⌬ espontaneamente
- BPE herda overhead: "a" virava DNA maior que o original
- Decisão: vertente experimental para modelo que usa ⌬ como atalho inteligente

---

## Entrada 2 — 07/Abr/2026 (⌬ Format Spec + First Test)

### Decisão Fundamental: ⌬ é atalho, não substituto

**Regra de ouro**: `⌬ID` só aparece quando é MENOR que o texto original.
- "oi" fica "oi" (2 chars < ⌬W42 = 4 chars)
- "inteligência artificial" vira ⌬F186 (25 chars → 5 chars = 5x compressão)

### Hierarquia de Tiers
- `⌬W` = Palavra longa (≥ 5 chars)
- `⌬F` = Frase (2+ palavras)
- `⌬P` = Parágrafo (blocos inteiros)

### Resultados do Primeiro Teste (DNACompressor)

```
Codebook: codebook_1x5_dinamico_expandido.json
  ⌬W entries: 1.198 (palavras longas que comprimem)
  ⌬F entries: 13.512 (frases que comprimem)

Testes:
  "a"                     → "a"                    (1.0x, mantém) ✅
  "oi"                    → "oi"                   (1.0x, mantém) ✅
  "de que para"           → "⌬F76 ⌬W1"            (1.5x) ✅
  "Olá mundo"             → "⌬F11195"              (2.0x) ✅
  "IA é o futuro da comp" → "⌬F186 é ⌬F9415 comp" (1.8x) ✅
  "IA pode resolver..."   → "⌬F2413 ⌬F589..."     (2.8x) ✅ 🔥

  Roundtrip: 100% correto em todos os testes
```

### Insight
O modelo NÃO precisa de tokenizer DNA. Ele precisa de tokenizer NORMAL (BPE/char) e
aprender a emitir ⌬IDs como parte do seu vocabulário natural. É como aprender siglas:
"ONU" = "Organização das Nações Unidas".

### Próximo Passo
- Gerar dataset Alpaca-PT com frases substituídas por ⌬IDs
- Treinar modelo pequeno para ver se aprende o padrão
- Se o loss converge → vale Colab

---

## Entrada 3 — 07/Abr/2026 (Achado Crítico: ⌬ vs BPE)

### Teste: Como Qwen tokeniza ⌬?
- `⌬` sozinho = **2 tokens BPE** (fragmentado em bytes UTF-8)
- `⌬F42` = **5 tokens BPE** (⌬ + F + 4 + 2)
- `⌬F2413` = **7 tokens BPE**
- Texto normal equivalente = **3-4 tokens BPE**

### Impacto
O formato ⌬ com IDs numéricos é **MAIS CARO** em tokens BPE que o texto original
para frases curtas. Cada dígito custa 1 token BPE.

### Soluções Encontradas
1. Adicionar `⌬` como `additional_special_token` → vira 1 token ✅
2. `Δ` (delta grego U+0394) já é 1 token nativo no Qwen ✅
3. Priorizar ⌬P (parágrafos, 10x savings) sobre ⌬W (palavras, overhead)

### Pivô Estratégico
O valor do ⌬ NÃO é eficiência BPE para palavras curtas.
O valor REAL é:
- **⌬P** para parágrafos inteiros (10x compression)
- **Multi-Brain** para modulardade de conhecimento
- **Cognição compressora** como capacidade aprendida
- **Futuro tokenizer custom** que elimina o overhead BPE

### Status
Decisão: manter ⌬, adicionar como special token, focar em ⌬F longos + ⌬P.
