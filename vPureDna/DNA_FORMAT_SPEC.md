# 🧬 vPureDna — Especificação do Formato DNA (⌬)

## Princípio Central

> **O DNA (`⌬`) existe para COMPRIMIR e ACELERAR, não para substituir texto curto.**
> Se `⌬ID` é maior que o texto original, o texto original FICA.

```
REGRA DE OURO:
  len("⌬ID") < len(texto_original)  →  usar ⌬ID  (comprime)
  len("⌬ID") ≥ len(texto_original)  →  manter texto  (não comprime)
```

### Exemplos da Regra

| Texto Original | ⌬ID | Tamanho Original | Tamanho ⌬ | Decisão |
|----------------|------|-----------------|-----------|---------|
| "a" | ⌬W1 | 1 | 3 | ❌ Manter "a" |
| "oi" | ⌬W42 | 2 | 4 | ❌ Manter "oi" |
| "de" | ⌬W1 | 2 | 3 | ❌ Manter "de" |
| "para" | ⌬W3 | 4 | 3 | ✅ Usar ⌬W3 |
| "computação" | ⌬W89 | 11 | 4 | ✅ Usar ⌬W89 |
| "inteligência artificial" | ⌬F42 | 25 | 4 | ✅ Usar ⌬F42 |
| "O Brasil é um país..." (40 chars) | ⌬P7 | 40 | 3 | ✅ Usar ⌬P7 |

---

## Hierarquia de Tiers

O sistema usa 3 tiers, cada um com um prefixo semântico após o `⌬`:

```
⌬W  = Palavra  (Word)      — 1 palavra longa (≥ 4 chars)
⌬F  = Frase    (Phrase)     — 2+ palavras recorrentes
⌬P  = Parágrafo (Paragraph) — blocos inteiros de texto/código
```

### Formato: `⌬[TIER][ID_NUMÉRICO]`

| Tier | Formato | Exemplo | Expande para | Tamanho ⌬ |
|------|---------|---------|-------------|-----------|
| W (Word) | `⌬W` + número | `⌬W89` | "computação" | 4 chars |
| F (Phrase) | `⌬F` + número | `⌬F42` | "inteligência artificial" | 4 chars |
| P (Paragraph) | `⌬P` + número | `⌬P7` | Bloco inteiro de texto/código | 3 chars |

### Capacidade

| Tier | Entries Estimadas | ID Max | Tamanho Max do ⌬ | Threshold (texto ≥) |
|------|-------------------|--------|-------------------|---------------------|
| W | ~2.000 palavras | `⌬W1999` | 6 chars | Palavras ≥ 7 chars |
| F | ~11.000 frases | `⌬F10999` | 7 chars | Frases ≥ 8 chars |
| P | ~500 parágrafos | `⌬P499` | 5 chars | Parágrafos ≥ 6 chars |

> **Nota**: Na prática, os IDs mais frequentes são curtos (⌬W1 a ⌬W99, ⌬F1 a ⌬F99),
> então a maioria dos ponteiros é de 3-4 chars. Isso comprime MUITO.

---

## Como o Modelo Usa DNA

O modelo **pensa e fala em linguagem natural**. O DNA é um atalho:

```
SEM DNA (output bruto):
  "A inteligência artificial é o futuro da computação moderna no Brasil"

COM DNA (output comprimido):
  "A ⌬F42 é o ⌬F108 moderna no ⌬W201"
  
  onde: ⌬F42 = "inteligência artificial"
        ⌬F108 = "futuro da computação"
        ⌬W201 = "Brasil"  (só se "Brasil" aparecer muito no contexto)
```

### Benefícios
1. **Velocidade**: Menos tokens para gerar = inferência mais rápida
2. **Compressão**: Resposta ocupa menos espaço
3. **Expansão**: Novos cérebros (codebooks) adicionam novos ⌬IDs sem retreinar

---

## Multi-Brain: Expansão de Inteligência

Cada "cérebro" é um codebook com seus próprios ⌬IDs:

```
Cérebro Base (PT-BR):
  ⌬W1..⌬W2000   → Palavras comuns
  ⌬F1..⌬F11000  → Frases comuns

Cérebro Jurídico (plugin):
  ⌬J1..⌬J500    → Termos jurídicos
  Ex: ⌬J42 = "o réu deverá comparecer em juízo"

Cérebro Código (plugin):
  ⌬C1..⌬C300    → Patterns de código
  Ex: ⌬C7 = "def __init__(self):\n    super().__init__()"

Cérebro Médico (plugin):
  ⌬M1..⌬M800    → Termos médicos
  Ex: ⌬M15 = "diagnóstico diferencial"
```

**Hot-swap**: troca o codebook → novos ⌬IDs disponíveis, mesmo modelo.

---

## Para o vPureDna (Teste)

O teste que queremos validar:

1. Pegar dataset Alpaca-PT
2. Substituir frases longas recorrentes por `⌬FXX` no training data
3. Treinar modelo pequeno (pode ser com tokenizer normal!)
4. Testar se o modelo aprende a emitir `⌬FXX` corretamente
5. Medir: compressão, qualidade, velocidade

**O que NÃO estamos fazendo**: converter TUDO para DNA.
**O que estamos fazendo**: ensinar o modelo a USAR ⌬ como atalho inteligente.
