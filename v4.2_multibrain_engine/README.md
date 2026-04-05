# 🧠 CROM-IA V4.2 — Multi-Brain Edge Engine

O Motor CROM-IA V4.2 é uma prova de conceito (PoC) otimizada para borda (Edge CPU) focada em Inferência Dinâmica e Mixture of Experts (MoE) via Bash Routing. Ele permite ligar, desligar e mesclar capacidades semânticas exclusivas empilhando LoRAs diretamente na memória sem necessidade de reinicializar modelos severos.

## 🌟 Arquitetura: Orquestração MoE (Mixture of Experts) Condicional

Em vez de empilhar cegamente todos os cérebros (causando a catástrofe temporal no *Attention Head* observada em modelos sub-1B), o CROM-IA V4.2 isola a demanda lendo a requisição (regex em tempo de execução) para rotear qual região semântica do Llama-cpp ligar.

Nossa matriz validada para CPU/RAM em dispositivos frugais se concentra em hiper-parâmetros base da documentação do QwEN:
- Temperatura `0.3` (Alta restrição de factualidade).
- Penalidade de Repetição `1.1` (Evita loops psiconeurais).
- Roteamento Básico (`1.0` PTBR) vs Roteamento Técnico (`0.80 Python` / `0.50 DPO`).

## ⚙️ Componentes
- **`chat_v42_brain.sh`**: Interface interativa de TUI. Permite ligar cérebros `[1-10]` e alternar RAG ativamente.
- **`benchmark_matrix_v42.sh`**: Suíte de SRE automatizada. Testa 10 prompts contornando o modelo em áreas limiares.
- **Extratores DNA**: Módulo de Descompressão Radix-4 para extração binária reversa.

## 🚀 Como Iniciar

1. Clone o repositório na sua placa raiz.
2. Extraia o `llama.cpp` compilado com flag `LLAMA_FUSE=1` (Memory Lock).
3. Cole os GGUFs na pasta `micro_cerebros`.
4. Aperte os cintos e rode: `./chat_v42_brain.sh`.

---
*Build for Local Intelligence. No Clouds Needed.*
