# 📓 vPureDna — Diário de Experimentos

## Entrada 1 — 07/Abr/2026 (Criação do Laboratório)

### Contexto
- O V4.3 Cognitive Leap demonstrou que LoRA sobre Qwen3.5-2B converge (Loss 13.28)
- Mas o modelo NÃO emite ponteiros DNA (@@XX) espontaneamente
- O tokenizer BPE herda overhead: "a" vira DNA maior que o texto original
- Decisão: criar vertente experimental para treinar do zero, sem BPE

### Análise da outra IA (insights absorvidos)
1. **Começar híbrido** — DNA para chunks grandes + tokens tradicionais para curtos
2. **Codebook hierárquico 1x5** — letras/bigramas → 2-4 bases, palavras/frases → maiores
3. **Greedy longest-match** — priorizar frases longas no encode
4. **Vocabulário 16-256** — 4 tokens puros pode ser radical demais, usar bigramas/4-gramas
5. **Self-distillation** — modelo aprende com próprias predições validadas

### Decisões Pendentes
- [ ] Chassis do modelo
- [ ] Tamanho do vocabulário
- [ ] Corpus de treino
- [ ] Hardware

### Status: Aguardando decisões fundacionais para iniciar Fase 1
