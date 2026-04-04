# 01. Codebook Regeneration Protocol (V3.6)

## 📌 O Problema da V3.5b
Na versão anterior, injetamos um Codebook gerado artificialmente que acabou contendo "receitas de bolo" e "piadas". Como não refletia a natureza real e repetitiva da Lógica de Programação, a LLM nunca aprendeu a associar o seu uso a código. Para resolvermos a transição, precisamos **extrair Ouro Sintático** (N-Grams perfeitos) usando matemática pura em cima de um dataset focado em código.

## 🛠️ Método Numérico
Vamos mapear um corpus pesado focado em Python/C++/Shell e aplicar uma varredura para extrair `N-Grams` (blocos de texto de 10 a 50 caracteres) que se repetem com altíssima frequência matemática. 

Aplica-se a fórmula de **Entropia de Claude Shannon**:
Blocos que diminuem largamente a entropia do documento (muitos bytes / padrão altamente predizível) são convertidos em Chaves Radix-4 (A, C, T, G).

### Critérios de Sucesso para uma Macro:
1. Tamanho do texto em Bytes: `>= 10`
2. Frequência no Dataset alvo: `>= 500 vezes`
3. A macro **NÃO PODE** ser sub-representada por outra macro maior. (Exemplo falso-positivo: "def " vs "def __init__(self):")

## ⚙️ Exemplo Prático (Geração)
A execução chamará o script `gerador_codebook.py` que mastigará 1GB de Github Code e ejetará no final um `macro_codebook_v4.json` limpíssimo e focado.

**Exemplo do Arquivo Gerado:**
```json
"AATA": {
      "text": "def __init__(self):\n        super().__init__()\n",
      "freq": 9410,
      "bytes": 48
}
```
