# 🧬 vPureDna — Planejamento Completo para Treino Cognitivo

**Última atualização:** 07/Abr/2026  
**Objetivo:** Treinar a cognição base do CROM-IA para usar `⌬` como atalho inteligente de compressão  
**Meta:** Modelo que ENTENDE e EMITE `⌬`, pronto para receber cérebros (codebooks)

---

## 📖 O QUE É ESTE PROJETO?

### O Problema
O CROM-IA V4.3 usa LoRA sobre Qwen3.5-2B com tokenizer BPE herdado. O modelo:
- ❌ NÃO emite marcadores DNA espontaneamente
- ❌ Fragmenta DNA via BPE (overhead em strings curtas)
- ❌ Precisa de LoRA + prompt engineering para cada novo domínio

### A Solução: Cognição ⌬
Treinar um modelo que **nativamente** entende o símbolo `⌬` como demarcador de conteúdo comprimido.
O modelo pensa em texto normal, mas usa `⌬ID` como atalho quando comprime.

**Analogia:** Assim como humanos usam siglas (ONU = "Organização das Nações Unidas"),
o modelo aprende a usar `⌬F42` = "inteligência artificial".

### A Visão Final
```
   Modelo Base (cognição ⌬)
        │
        ├── Cérebro PT-BR      (codebook base: ⌬W, ⌬F)
        ├── Cérebro Jurídico   (codebook: ⌬J1..⌬J500)
        ├── Cérebro Código     (codebook: ⌬C1..⌬C300)
        ├── Cérebro Médico     (codebook: ⌬M1..⌬M800)
        └── ... (hot-swap, zero retreino)

   O modelo base SÓ precisa saber:
   1. "⌬ seguido de ID = texto comprimido"
   2. "Quando o texto é longo/repetitivo, usar ⌬"
   3. "Quando pedem expandir ⌬, devolver texto"
```

---

## ⚗️ O QUE JÁ TESTAMOS E COMPROVAMOS

### Teste 1: Regra de Ouro (⌬ só quando comprime)
**Pergunta:** O formato ⌬ realmente comprime texto PT-BR?

**Resultado:** ✅ SIM
```
  "a"  →  "a"              (1.0x, mantém — ⌬ não compensa)
  "oi" →  "oi"             (1.0x, mantém — ⌬ não compensa)  
  "IA pode resolver..."  →  "⌬F2413 ⌬F589..."  (2.8x compressão!)
```
**Conclusão:** A regra funciona. Palavras curtas ficam intactas, frases longas comprimem até 2.8x.

### Teste 2: Cobertura em Texto Real PT-BR
**Pergunta:** Qual % de texto real pode ser comprimido?

**Resultado:** 
- Compressão média: **1.69x** (35% dos tokens viram ⌬)
- 43 de 122 tokens foram substituídos por ⌬
- Codebook: 1.198 palavras (⌬W) + 13.512 frases (⌬F) = 14.710 entries

**Conclusão:** Cobertura boa para texto técnico PT-BR. Texto coloquial terá cobertura menor.

### Teste 3: Round-trip Integrity
**Pergunta:** Comprimir→descomprimir produz texto idêntico?

**Resultado:** ✅ **8/8 textos 100% corretos**

**Conclusão:** Zero perda de informação. A compressão é lossless.

### Teste 4: Edge Cases
**Pergunta:** O sistema lida com código, URLs, emojis, acentos?

| Caso | Ratio | Round-trip | Nota |
|------|-------|-----------|------|
| Números | 1.5x | ✅ | "12345 é um número" funciona |
| URL | 1.2x | ✅ | Fragmenta URL mas reconstrói |
| Código Python | 1.1x | ✅ | Compressão mínima (esperado) |
| Emoji 🔥 | 1.5x | ✅ | Emoji preservado |
| Acentos PT-BR | 1.1x | ✅ | ã,é,ç,ú todos preservados |
| String vazia | - | ✅ | Sem crash |
| 1 char | 1.0x | ✅ | Mantém intacto |
| Repetição | 1.0x | ✅ | "a a a" fica intacto (correto) |

**Conclusão:** Robusto em todos os edge cases.

### Teste 5: Smoke Test — Modelo Aprende ⌬?
**Pergunta:** Um transformer do zero consegue aprender o padrão ⌬?

**Setup:** 0.82M params, char-level, 300 amostras, CPU, 300 steps

**Resultado:** 
- Loss: 4.33 → 1.53 (**-65% convergência**)
- O modelo **emite** tokens `⌬F` e `⌬W` corretamente
- Texto gibberish (esperado com 0.82M)

**Conclusão:** ✅ O CONCEITO é viável. Um modelo maior vai funcionar.

### Teste 6: Eficiência do Threshold por Tamanho
**Pergunta:** A partir de quantos chars uma palavra vira ⌬W?

> **Nota:** Este threshold se aplica ao texto RAW, não aos tokens BPE.
> Vide Teste 7 para a análise de eficiência BPE.

**Resultado:**
```
  ⌬W aceita palavras de:
    4 chars:   7 palavras  (ex: "para" → ⌬W3, salva 1 char)
    5 chars:  26 palavras  (ex: salva 1-2 chars)
    6 chars: 235 palavras  (ex: salva 2-3 chars)
    7 chars: 282 palavras  (sweet spot)
    8+ chars: 648 palavras (forte economia)
```
**Conclusão:** O threshold automático está correto. Palavras < 4 chars nunca viram ⌬.

### Teste 7: ⌬ no Tokenizer BPE do Qwen (ACHADO CRÍTICO)
**Pergunta:** O formato ⌬ economiza tokens BPE no Qwen?

**Resultado:** ⚠️ **NÃO para frases curtas. SIM para parágrafos.**
```
  ⌬ sozinho          = 2 tokens BPE (fragmentado!)
  ⌬F42               = 5 tokens BPE
  "intelig. artificial" = ~3-4 tokens BPE
  → ⌬F42 custa MAIS tokens que o texto original!

  MAS:
  Δ (grego, U+0394)  = 1 token BPE ✅
  ⌬ como special_token = 1 token BPE ✅ (precisa resize embeddings)

  Comparação de formatos (4 marcadores de frases):
    Texto normal          → 14 tokens BPE (baseline)
    ⌬F2413 x4 (atual)     → 30 tokens (pior)
    ΔF2413 x4 (delta grego)→ 23 tokens (pior)
    #F2413 x4             → 23 tokens (pior)
    [F2413] x4            → 26 tokens (pior)

  POR QUÊ: BPE quebra cada dígito em 1 token.
  F2413 = F + 2 + 4 + 1 + 3 = 5 tokens sempre.
```

**Conclusão e Pivô Estratégico:**

O benefício do ⌬ **NÃO É** eficiência no nível BPE para frases curtas.
O benefício REAL do ⌬ é:

1. **⌬P (Parágrafos):** 200 chars = ~50 tokens BPE → `⌬P7` = ~5 tokens = **10x savings** ✅
2. **Multi-Brain:** Trocar codebook em runtime = novo domínio sem retreinar
3. **Compressão RAW:** Texto com ⌬ é menor como texto bruto (APIs, disco, transmissão)
4. **Cognição modular:** O modelo aprende o CONCEITO de compressão, não os IDs específicos
5. **Futuro custom tokenizer:** Quando treinarmos tokenizer próprio, ⌬ vira native

**Decisão tomada:** Manter ⌬ mas PRIORIZAR ⌬P (parágrafos) e ⌬F (frases longas).
IDs curtos (⌬W para palavras) são menos valiosos no regime BPE.

---

## 🗺️ PLANEJAMENTO: O QUE FAZER PARA O COLAB

### Fase 0: Preparação Local (ANTES do Colab) ✅ JÁ FEITA
- [x] Definir formato ⌬ e regra de ouro
- [x] Implementar DNACompressor (`vPureDna/01_encoder/tokenizer_dna.py`)
- [x] Validar round-trip, edge cases, cobertura
- [x] Smoke test com modelo char-level (0.82M)
- [x] Documentar resultados

### Fase 1: Dataset Grande para Colab (A FAZER)
**O quê:** Gerar dataset trifásico de 5.000-10.000 amostras usando Alpaca-PT do HuggingFace
**Por quê:** 300 amostras não é suficiente para Qwen aprender. 5K+ é o mínimo.
**Como:**
```
1. Baixar Alpaca-PT (52K amostras) no Colab
2. Comprimir respostas com DNACompressor  
3. Gerar 3 fases:
   - Fase A (33%): Pergunta → Resposta com ⌬ (aprende a EMITIR)
   - Fase B (33%): Texto com ⌬ → Texto expandido (aprende a ENTENDER)
   - Fase C (33%): Pergunta → Resposta normal (mantém capacidade base)
4. Exportar como JSONL para LoRA
```
**Critério de sucesso:** Dataset ≥ 5.000 amostras, compressão média ≥ 1.5x

### Fase 2: Adaptar Pipeline LoRA para Qwen + ⌬ (A FAZER)
**O quê:** Modificar o pipeline de treino Colab (já funciona para V4.3) para usar dataset ⌬
**Por quê:** Qwen3.5-2B já provou convergir (Loss 13.28 no V4.3). Com dataset ⌬ deve convergir melhor.
**Como:**
```
1. Reusar colab/CROM_DNA_LoRA_Training.py como base
2. Trocar dataset por dataset_vpuredna_5k.jsonl
3. Ajustar template ChatML:
   <|im_start|>system
   Você é CROM-IA. Use ⌬ para comprimir frases conhecidas.
   <|im_end|>
   <|im_start|>user
   [instrução + input]
   <|im_end|>
   <|im_start|>assistant
   [output com ⌬ ou sem]
   <|im_end|>
4. Treinar com LoRA (r=16, alpha=32)
5. Loss alvo: < 5.0 em 500 steps
```
**Critério de sucesso:** Loss convergente, modelo emite ⌬ em respostas

### Fase 3: Validação Pós-Treino (NO COLAB)
**O quê:** Testar se o modelo treinado realmente usa ⌬ corretamente
**Como:**
```
Teste A: Dar prompt normal → modelo deve responder com ⌬ onde apropriado
Teste B: Dar ⌬ como input → modelo deve expandir corretamente
Teste C: Dar prompt normal sem pedir ⌬ → modelo deve responder em texto normal
Teste D: Trocar codebook (simular novo cérebro) → modelo adapta?
```
**Critério de sucesso:** 
- Teste A: ≥ 50% das respostas contêm ⌬ válidos
- Teste B: BLEU > 0.5 na expansão
- Teste C: Resposta coerente sem ⌬
- Teste D: Modelo usa ⌬ do novo codebook via few-shot

### Fase 4: Exportar e Integrar (PÓS-COLAB)
**O quê:** Converter modelo treinado para GGUF e integrar com pipeline local
**Como:**
```
1. Merge LoRA + Base → Modelo completo
2. Quantizar para Q4_K_M (~1.2 GB)
3. Testar com llama-cli
4. Integrar com DNACompressor como decoder
5. Push para HuggingFace (CromIA/CROM-IA-vPureDna)
```

---

## 🧠 DECISÕES TOMADAS E POR QUÊ

### 1. Por que ⌬ e não @@ ou outra sintaxe?
**Decisão:** `⌬` (hexágono químico) é o ÚNICO demarcador de DNA.
**Por quê:** 
- `@@` conflita com emails, Python decorators, e outras convenções
- `⌬` é visualmente único, não aparece em texto natural
- Remete à biologia/química (DNA, moléculas) — alinhado com a identidade CROM
- 1 caractere Unicode = eficiente para o tokenizer

### 2. Por que NÃO converter tudo para DNA?
**Decisão:** Manter texto normal e usar ⌬ só como atalho.
**Por quê:**
- "oi" → `⌬W42` é PIOR (2 chars → 4 chars)
- O modelo precisa pensar em linguagem humana para ser inteligente
- DNA puro (ATCG) explode sequências 4x → inviável
- A compressão só vale quando ECONOMIZA tokens

### 3. Por que tiers (⌬W, ⌬F, ⌬P)?
**Decisão:** Separar por nível semântico.
**Por quê:**
- Evita colisão de IDs entre palavras e frases
- Permite codebooks especializados por nível
- O modelo pode aprender regras diferentes por tier:
  - ⌬W: usar para palavras muito frequentes/longas
  - ⌬F: usar para frases recorrentes
  - ⌬P: usar para blocos inteiros (futuro)

### 4. Por que Qwen3.5-2B e não modelo menor?
**Decisão:** Usar Qwen3.5-2B como base para LoRA.
**Por quê:**
- V4.3 já provou que converge (Loss 13.28)
- Modelos < 1B colapsam (V4.2 demonstrou com 0.6B)
- 2B é o mínimo para raciocínio estável em PT-BR
- Apache 2.0 license = uso livre comercial
- Roda em hardware edge (~1.2 GB quantizado)

### 5. Por que dataset trifásico (A/B/C)?
**Decisão:** 3 fases com objetivos diferentes.
**Por quê:**
- **Fase A (comprimir):** Ensina o modelo QUANDO e COMO emitir ⌬
- **Fase B (expandir):** Ensina o modelo a ENTENDER ⌬ recebido (decoder)
- **Fase C (normal):** Mantém habilidade de responder SEM ⌬ (não perder capacidade)
- Sem Fase C, o modelo pode "esquecer" como falar normalmente

### 6. Por que NÃO treinamos do zero com tokenizer DNA custom?
**Decisão:** Usar LoRA sobre modelo existente em vez de treinar do zero.
**Por quê:**
- Treinar do zero com 50M-150M params gera texto gibberish (smoke test provou)
- Qwen 2B já sabe PT-BR, lógica, código — reaproveitamos essa inteligência
- LoRA é mais rápido (2-4h vs dias) e mais barato (T4 grátis)
- O ⌬ é adicionado ao VOCABULÁRIO do modelo, não substitui o tokenizer

### 7. Por que não mudar ⌬ para outro símbolo após o teste BPE?
**Decisão:** Manter ⌬ como demarcador, mas adicionar como special token no LoRA.
**Por quê:**
- `Δ` (delta grego) é single-token, mas é usado em matemática/ciência (conflito)
- `⌬` adicionado como special token vira 1 token (limpo, sem conflito)
- A identidade visual do projeto é ⌬ — mudar agora fragmenta a documentação
- O overhead BPE é aceitável porque o valor está nos PARÁGRAFOS, não nas palavras
- No futuro, tokenizer custom elimina completamente o overhead

### 8. Por que priorizar ⌬P e ⌬F longos sobre ⌬W?
**Decisão:** O dataset para Colab deve focar em frases longas e parágrafos.
**Por quê:**
- ⌬W (palavras curtas) NÃO economiza BPE → overhead negativo
- ⌬F com IDs curtos (⌬F1 a ⌬F99) economiza 1-2 chars por frase
- ⌬P (parágrafos) economiza MUITO: 200 chars → ⌬P7 = 10x savings
- O modelo aprende o PADRÃO de compressão, não IDs específicos
- Com Multi-Brain, o modelo reutiliza esse padrão para QUALQUER codebook

---

## 🔍 CAMINHOS QUE NÃO ESCOLHEMOS (e por quê)

### ❌ Caminho A: Vocabulário de 4 tokens (A,T,C,G puros)
**Por quê descartado:** Sequências ficam 4x maiores. "hello" (5 bytes) = 20 bases DNA.
Context window de 2048 tokens caberia ~500 chars de texto. Inviável.

### ❌ Caminho B: Treinar modelo do zero com tokenizer custom
**Por quê descartado:** Smoke test mostrou que 0.82M params = texto gibberish.
Precisaríamos de ~500M+ params e semanas de GPU. Não compensa vs LoRA.

### ❌ Caminho C: Usar ⌬ para TODAS as palavras (sem threshold)
**Por quê descartado:** "de" → `⌬W1` expande de 2 para 3 chars. Overhead negativo.
A regra de ouro (⌬ só quando comprime) é obrigatória.

### ❌ Caminho D: BPE custom treinado em DNA
**por quê descartado:** BPE sobre sequências A,T,C,G reconstrói basicamente o texto original.
É um ciclo inútil: texto → DNA → BPE → tokens ≈ texto → BPE → tokens.

### ⏸️ Caminho E: Modelo puramente sub-simbólico (futuro)
**Status:** Não descartado, apenas adiado. Se o Caminho atual (LoRA + ⌬) funcionar bem,
podemos explorar um encoder/decoder puramente neural no futuro.

---

## 📁 ARQUIVOS DO PROJETO

| Arquivo | Para que serve |
|---------|---------------|
| `vPureDna/README.md` | Visão geral e roadmap |
| `vPureDna/PLANEJAMENTO_COMPLETO.md` | **ESTE ARQUIVO** — tudo documentado |
| `vPureDna/DNA_FORMAT_SPEC.md` | Especificação técnica do formato ⌬ |
| `vPureDna/DIARIO.md` | Log cronológico de experimentos |
| `vPureDna/01_encoder/tokenizer_dna.py` | DNACompressor — comprime/expande ⌬ |
| `vPureDna/02_dataset/gerar_dataset_puro.py` | Gerador de dataset trifásico |
| `vPureDna/02_dataset/dataset_vpuredna.jsonl` | Dataset smoke test (300 amostras) |
| `vPureDna/03_modelo/train.py` | Trainer smoke test (char-level) |
| `vPureDna/03_modelo/model.py` | NanoGPT adaptado (não usado no Colab) |
| `vPureDna/03_modelo/checkpoints/` | Checkpoints do smoke test |

---

## ⚠️ RISCOS E MITIGAÇÕES

| Risco | Probabilidade | Impacto | Mitigação |
|-------|---------------|---------|-----------|
| Qwen não aprende ⌬ via LoRA | Baixa | Alto | Aumentar dataset, r=32 |
| ⌬ conflita com tokenizer BPE do Qwen | Média | Alto | Verificar se ⌬ é token único no vocab Qwen |
| Modelo "esquece" PT-BR após LoRA | Baixa | Médio | Fase C (33% normal) no dataset |
| Codebook desatualizado | Média | Baixo | Regenerar com corpus maior |
| Overfitting com dataset pequeno | Alta | Médio | Usar 5K+ amostras, val split, early stopping |

---

## 📋 CHECKLIST FINAL ANTES DO COLAB

- [x] DNACompressor implementado e testado
- [x] Regra de ouro validada (⌬ só quando comprime)
- [x] Round-trip 100% correto
- [x] Edge cases todos passam
- [x] Smoke test converge (-65% loss)
- [x] Modelo emite ⌬ (mesmo que gibberish)
- [x] Verificar token ⌬ no vocab do Qwen tokenizer (resultado: 2 tokens, resolver com add_special_tokens)
- [x] Testar marcadores alternativos (Δ grego = 1 token, mas mantemos ⌬)
- [x] Descobrir que IDs numéricos longos (⌬F2413) custam caro em BPE
- [x] Pivotar: priorizar ⌬P (parágrafos) e ⌬F com IDs curtos
- [ ] Gerar dataset 5K+ com Alpaca-PT (focar em ⌬F curtos + ⌬P)
- [ ] Adaptar pipeline Colab para dataset ⌬ com add_special_tokens
- [ ] Fazer script de validação pós-treino
- [ ] Preparar script de exportação GGUF
- [ ] Preparar script de exportação GGUF

---

*Documento criado em 07/Abr/2026 — CROM-IA vPureDna Lab*
*Este é um living document. Cada sessão de trabalho deve atualizar as seções relevantes.*
