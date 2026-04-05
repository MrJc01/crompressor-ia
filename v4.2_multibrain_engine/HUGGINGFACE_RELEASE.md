# HUGGINGFACE MODEL CARD RELEASE: CROM-IA V4.2 (0.6B)

Copie e cole o texto abaixo no `README.md` do seu Repositório do Modelo no HuggingFace.

---

### Model Overview
**CROM-IA V4.2 (Multi-Brain Stack)**
O CROM-IA 4.2 é um avanço conceitual estrutural empacotado para o Edge. Este repositório hospeda a coleção de Micro-Cérebros Base, Python e DPO para a rede neural **Qwen 0.6B (Llama L-CPP Q4)**, servindo como uma solução Mixture-of-Experts para instâncias de RAM frugal via Terminal/TUI.

### Stack Weights Experiment (SRE Calibrated):
O modelo colapsará por *Interferência Catastrófica de Atenção* se você injetar LoRAs de forma 1:1 absoluta.
A matriz de orquestração ideal para inference via `llama-cli` é rotear (ligar/desligar) LoRAs usando Regex. Se o disparo da string contiver matemática/Python, use a composição:
- `Base_PTBR_lora.gguf`: 1.0
- `Python_DNA_lora.gguf`: 0.8
- `DPO_Preference_lora.gguf`: 0.5 

### Inference Configuration:
- `Temperature`: 0.3
- `Repeat-Penalty`: 1.10
- `Top-K`: 40

**Warning regarding < 1B Models**:
O motor de geração natural desta versão está amarrado ao limite físico de atenção matemática do Qwen 0.6B 4-bit. Loops recorrentes de repetição podem ocorrer. O foco do CROM-IA 4.2 foi a implantação modular. A coesão semântica longa será abordada no roadmap _Cognitive Leap V4.3_.
