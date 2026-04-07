# Pesquisa 4: Avaliações SRE e Benchmarks

## Objetivo
Criar uma "Monitoração de Sinais Vitais" para o LLM.

## Métricas (KPIs)
- **T/s (Tokens per Second):** Monitorar a queda de velocidade ao longo do tempo (indicação de estouro de KV-Cache).
- **Hallucination Rate:** Validar respostas contra fatos estruturados.
- **DNA Compression Ratio:** Medir o ganho de velocidade vs perda de precisão.

## Ação
Integrar com o `ROOT_DIAGNOSTICS.md` para criar relatórios post-mortem automáticos se o modelo travar.
