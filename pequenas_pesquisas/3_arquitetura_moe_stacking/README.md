# Pesquisa 3: Arquitetura MoE e Stacking de LoRAs

## Objetivo
Mudar de Bash Routing (V4.2) para Python Orquestrado (V4.3).

## Testes Reais
1. **Interferência de Pesos:** Carregar 2 LoRAs (Python + PT-BR) e verificar a diluição da semântica.
2. **Switching Hot-Swap:** Testar o tempo de troca de LoRA em runtime via API.

## Design
O "cérebro" não deve mais ser um `if-else` em bash, mas um `SemanticRouter` em Python que decide qual LoRA ativar com base no embedding da pergunta.
