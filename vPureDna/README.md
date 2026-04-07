# 🧬 vPureDna — Treinamento de Modelo do Zero com DNA Puro

> **Objetivo:** Provar que é possível treinar um modelo de linguagem PEQUENO (50-500M params) do ZERO usando APENAS vocabulário Base-4 (A/T/C/G), sem depender de tokenizer BPE.

**Status:** 🔬 LABORATÓRIO — Fase de concepção  
**Data de criação:** 07 de Abril de 2026  
**Relação com V4.3:** Vertente experimental paralela (não substitui V4.3)

---

## 💡 Por que vPureDna existe?

O V4.3 usa LoRA sobre o Qwen3.5-2B, que HERDA o tokenizer BPE. Isso causa:
- **Overhead em strings curtas:** "a" ou "oi" vira DNA maior que o texto original
- **BPE fragmenta DNA:** "ATCGATCG" é cortado em tokens sem sentido pelo tokenizer
- **O modelo não pensa em DNA** — ele pensa em BPE e tenta traduzir para DNA

A solução radical: um modelo que **nasce pensando em DNA**. Sem BPE. Sem tokenizer legado. Puro Base-4.

---

## 🏗️ Estrutura do Laboratório

```
vPureDna/
├── README.md                  ← Você está aqui
├── DIARIO.md                  ← Log de experimentos e decisões
│
├── 01_encoder/                ← DNA Encoder/Decoder inteligente
│   ├── dna_encoder_v2.py      ← Codebook-aware, greedy longest-match
│   ├── dna_decoder_v2.py      ← Decoder correspondente 
│   └── test_encoder.py        ← Testes: "a", "oi", frases, parágrafos
│
├── 02_dataset/                ← Geração de dados para treino
│   ├── gerar_dataset_puro.py  ← Gerador 100% DNA (sem BPE)
│   ├── validar_dataset.py     ← Métricas: overhead, entropia, cobertura
│   └── corpus/                ← Textos fonte PT-BR
│
├── 03_modelo/                 ← Arquitetura do modelo
│   ├── nano_dna_transformer.py ← Transformer minimalista (50-200M)
│   ├── tokenizer_dna.py       ← Tokenizer custom Base-4
│   └── treinar.py             ← Script de treino PyTorch puro
│
├── 04_avaliacao/              ← Benchmarks e métricas
│   ├── benchmark_overhead.py  ← Bytes DNA vs bytes texto original
│   ├── benchmark_qualidade.py ← BLEU/ROUGE do decode
│   └── benchmark_velocidade.py ← Tokens/s no hardware local
│
└── 05_colab/                  ← Notebooks para GPU cloud
    ├── vPureDna_treino.ipynb   ← Notebook pronto para Colab
    └── requirements.txt        ← Dependências mínimas
```

---

## 🧭 Roadmap (4 Fases)

### Fase 0 — Decisões Fundacionais (Antes de Codar)
- [ ] Escolher chassis: Transformer do zero vs NanoGPT vs GPT-2 small (tokenizer substituído)
- [ ] Definir vocabulário DNA: 4 tokens (A,T,C,G) vs 16 (bigramas) vs 256 (4-gramas)
- [ ] Selecionar corpus: Alpaca-PT vs Wikipedia PT-BR vs Mix
- [ ] Decidir hardware: Colab T4/A100 vs Local (se modelo < 100M)

### Fase 1 — Encoder Inteligente (1-2 dias)
- [ ] Implementar `dna_encoder_v2.py` com codebook hierárquico adaptativo
- [ ] Resolver overhead: "a" ≤ 2 bases, "oi" ≤ 4 bases, parágrafo ≤ 60%
- [ ] Validar com testes automatizados em strings de 1, 5, 20, 100 chars
- [ ] Reusar codebooks existentes (`codebooks/codebook_1x5_dinamico.json`)

### Fase 2 — Modelo Nano + Dataset (1 semana)
- [ ] Gerar dataset 100% DNA (5000-10000 amostras)
- [ ] Implementar `nano_dna_transformer.py` (começar 50M params)
- [ ] Tokenizer custom sem BPE (`tokenizer_dna.py`)
- [ ] Treinar no Colab — Loss alvo < 20

### Fase 3 — Loop Completo (1 semana)
- [ ] Fechar: Texto → DNA → Modelo gera DNA → DNA → Texto
- [ ] Testar edge cases: strings curtas, acentos PT-BR, código, repetições
- [ ] Benchmarks: BLEU > 0.3, overhead < 80%, velocidade > 0.5 t/s

### Fase 4 — Evolução (Contínua)
- [ ] Se funcionar: escalar para 200-500M params
- [ ] Ativar ponteiros @@XX entre contextos
- [ ] Integrar com `crompressor.wasm` para demo no browser
- [ ] Comparar com V4.3 LoRA (qualidade vs eficiência)

---

## 📊 Métricas de Sucesso

| Métrica | Mínimo | Alvo |
|---------|--------|------|
| DNA de "a" (1 char) | ≤ 4 bases | ≤ 2 bases |
| DNA de "oi" (2 chars) | ≤ 6 bases | ≤ 4 bases |
| Overhead (100 chars) | ≤ 80% | ≤ 60% |
| Loss final treino | < 20 | < 10 |
| Decode BLEU (PT-BR) | > 0.3 | > 0.6 |
| Velocidade inference | > 0.5 t/s | > 2 t/s |

---

## 🧬 Estratégia de Abordagem: Híbrido → Puro

A recomendação é **não começar 100% puro**. Evoluir gradualmente:

```
Fase A: HÍBRIDO
  └─ DNA para chunks ≥ 5 chars (codebook)
  └─ BPE/fallback para o resto

Fase B: SEMI-PURO
  └─ DNA para tudo
  └─ Fallback Base-4 ingênuo para raros

Fase C: PURO
  └─ Zero BPE
  └─ Vocabulário 100% DNA + codebook maduro
```

---

## 🔗 Dependências do Projeto Principal

| Recurso | Path no repo |
|---------|-------------|
| Codebooks 1x3/1x5 | `codebooks/` |
| Gerador de codebook | `codebooks/gerar_codebook.py` |
| Gerador dataset DNA | `gerar_dataset_dna.py` |
| Dataset comprimido | `datasets/gerar_dataset_comprimido.py` |
| Lab benchmark DNA | `lab_dna_crom.py` |
| Decoder streaming | `scripts/dna_decoder.py` |
| crompressor binário | `crompressor_bin` |
| Pipeline Colab | `colab/CROM_DNA_LoRA_Training.py` |

---

## ⚠️ Riscos Conhecidos

1. **Modelo puro DNA alucina demais** → Mitigação: começar híbrido
2. **Vocabulário de 4 tokens explode context window** → Mitigação: usar 16-256 tokens
3. **Codebook não cobre acentos PT-BR** → Mitigação: expandir com UTF-8 completo
4. **Treino local impossível** → Mitigação: Colab T4 grátis + modelo < 200M

---

*Este laboratório é uma vertente experimental do CROM-IA. Para continuar o trabalho da V4.3 (LoRA sobre Qwen), veja `v4.3_cognitive_leap/05_CONTINUACAO_V43.md`.*
