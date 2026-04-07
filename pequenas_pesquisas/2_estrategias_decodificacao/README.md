# Pesquisa 2: Estratégias de Decodificação

## Objetivo
Encontrar o equilíbrio de parâmetros que impeça o modelo sub-1B de entrar em loops infinitos (*fractal redundancy*).

## Parâmetros sob Investigação
- `Temperature`: Testar 0.1 (Estrito) até 0.8 (Criativo).
- `Min P`: Nova técnica para filtrar tokens de baixa probabilidade.
- `Repeat Penalty`: Testar valores entre 1.0 e 1.2.

## Hipótese
A V4.2 falhou porque a Temperatura 0.1 em modelos pequenos força o modelo a escolher sempre o mesmo token de maior probabilidade, gerando loops.
