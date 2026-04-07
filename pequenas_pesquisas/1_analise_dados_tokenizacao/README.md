# Pesquisa 1: Análise de Dados e Tokenização

## Objetivo
Identificar a densidade ideal de tokens DNA (@@) por parágrafo para evitar que o modelo perca a capacidade de estruturar frases em Português.

## Experimentos
1. **Teste de Colisão:** Verificar se tokens DNA (Ex: `@@PRT`) colidem com identadores de código reais.
2. **Medição de Entropia:** Comparar texto 100% natural vs texto com 25% DNA.

## Ferramentas
- `transpilador_v42.py`
- `gerador_codebook_v42.py`
