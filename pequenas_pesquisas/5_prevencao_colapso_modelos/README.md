# Pesquisa 5: Prevenção de Colapso de Modelos

## Objetivo
Blindar o modelo contra a degeneração matricial e a repetição infinita.

## Mecanismos
1. **Entropy Monitor:** Se o token seguinte for previsível demais (> 95%), forçar diversidade ou reiniciar o contexto.
2. **Dynamic Penalty:** Aumentar a penalidade de repetição dinamicamente se o script detectar 3 tri-gramas iguais.
3. **Flash Attention Check:** Validar se a ativação de `--flash-attn` reduz o "afogamento" de memória curta.
