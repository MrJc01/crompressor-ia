---
library_name: crompressor-ia
tags:
- qwen
- chat
- pt-br
---

# 🧬 CROM-IA V3.5b: A Evolução Orgânica (O Caso Bicarbonato)

Este modelo marca a transição da pesquisa de Compressão Sub-simbólica (CROM-IA V3.0) para a abstração de múltiplos cérebros (V4.0). 
Ele é um fine-tuning massivo sobre a base `Qwen2.5-1.5B-Instruct` utilizando **117.860 amostras puras de Chat em PT-BR** do dataset Alpaca e GPT-4 Traduzido.

## 🔬 A Descoberta Forense: Sabotados por Bicarbonato de Sódio
A proposta do treinamento V3.5b era misturar 100k conversas humanas com 18k códigos Python altamente comprimidos através de ponteiros `@@DNA` (Dicionário de Complexidade O(1)). O objetivo era curar o "Trauma do Brain-Lock" (onde o modelo falava apenas em código), ensinando-o a diferenciar quando devia falar naturalmente e quando devia injetar ponteiros.

**Porém, descobrimos o maior bug acidental do projeto:**
O Dicionário O(1) usado para varrer as 18.500 amostras de Python durante a criação do dataset local continha apenas *piadas em inglês* e *receitas de bolo* (Ex: `@@AT` = `- 1 colher de chá de bicarbonato de sódio`).
Como não existia bicarbonato de sódio em um script Python nativo, a varredura pré-treino falhou silenciosamente, gerando **0% de compressão** nos dados enviados para o Google Colab.

O resultado? Treinamos a A100 com um dataset 100% puro e não-comprimido. O modelo nunca viu uma tag `@@DNA`.

## 📈 Consequência Positiva (It's not a bug, it's a feature)
Como removemos a anomalia sintática sub-simbólica, este modelo recuperou e **expandiu violentamente a sua fluência humana em PT-BR**, enquanto manteve sua inteligência base para programação (uma vez que treinamos com Learning Rate baixíssimo `1e-5`). 
Ele não alucina, não trava e não excede contexto (desde que usado com uma configuração interativa saudável e `repeat_penalty 1.15`).

## 🔮 O Futuro: Arquitetura Multi-Brain V4.0
Esse "erro perfeito" nos provou matematicamente o conceito empírico de que *Catastrophic Forgetting* pune agressivamente LLMs pequenos quando tentamos forçar domínios antagônicos no mesmo espaço latente.
A partir da V4.0, o **CROM-IA deixará de ser um modelo único** e passará a ser uma matriz Orchestrator usando Modelos Especialistas (Brain Plugins com Adaptadores LoRA dinâmicos e especificação formal de sintaxe).

---

## 🚀 Como Fazer Funcionar (How to Run)

Como o modelo foi intensivamente treinado em cima de **117 mil conversas**, ele é **altamente sensível ao formato do prompt**. Se você não usar o formato correto, ele pode alucinar ou gerar loops infinitos.

**O formato obrigatório para inference (Alpaca-style):**
```text
Abaixo está uma instrução CROM-IA.

### Instruction:
[SUA PERGUNTA OU COMANDO AQUI]

### Input:


### Response:
```

### Exemplo usando `llama.cpp` (Modo CLI Interativo)
Para evitar que o contexto estoure ou que a penalidade de repetição crie loops no código, certifique-se de configurar a repetição para no máximo `1.15` e aumente a janela de contexto `-c`:

```bash
./llama-cli -m Qwen2.5-1.5B-Instruct.Q4_K_M-v3.5b_117k.gguf \
    -c 2048 \
    -n 1024 \
    --temp 0.1 \
    --repeat_penalty 1.15 \
    --in-prefix "\n\n### Instruction:\n" \
    --in-suffix "\n\n### Response:\n" \
    --prompt "Abaixo está uma instrução CROM-IA." \
    --interactive
```

---
**Data de Compilação:** 04 de Abril de 2026
**Treinamento:** NVIDIA A100 (1h27) / Unsloth 
**Loss Final:** 1.173599
