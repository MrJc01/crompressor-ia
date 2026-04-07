# 🧠 CROM-IA V4.3: O Manifesto do Salto Cognitivo (Abril 2026)

## 1. Visão Geral
A V4.3 abandona a fine-tuning supervisionada simples (SFT) para adotar uma arquitetura de **Reinforcement Fine-Tuning (RFT)** baseada nas descobertas de Abril de 2026.

## 2. Pilares de SOTA (State of the Art)

### 2.1. GRPO (Group Relative Policy Optimization)
- **O que é:** Treinamento de raciocínio profundo sem modelo de recompensa massivo.
- **Aplicação no CROM:** O modelo irá gerar múltiplos candidatos de compressão para o símbolo `⌬` e será recompensado pela **menor perda semântica** e **maior taxa de compressão**.

### 2.2. SCONE (Embeddings Contextuais)
- **Técnica:** Desacoplamento da matriz de entrada/saída.
- **Uso do ⌬:** O hexágono químico atuará como um "Ponteiro de Contexto", permitindo que o modelo decodifique o DNA sem aumentar o dicionário de saída.

### 2.3. ALM (Adaptive LoRA Merging)
- **Fusão:** Em vez de rodar o comando `--lora` estaticamente, o CROM-V4.3 usará uma fusão empírica baseada em feedback (ALM) para combinar conhecimentos de Medicina, Código e PT-BR.

## 3. Especificações Técnicas (2026)
- **Chassis:** Llama-4-Scout (3.8B) ou Llama-3.3-8B.
- **Símbolo Soberano:** `⌬` (UTF-8 Hexagon).
- **Token Strategy:** `⌬W_` (Word), `⌬F_` (Phrase), `⌬P_` (Paragraph).

## 4. Estratégia de Dados (SSD 2026)
- **SSD:** *Embarrassingly Simple Self-Distillation*. O modelo aprende com as próprias predições validadas pelo motor de SRE.
