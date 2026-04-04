<div align="center">
  <img src="https://raw.githubusercontent.com/FortAwesome/Font-Awesome/master/svgs/solid/dna.svg" width="80" alt="CROM-IA DNA Logo"/>
  <h1>CROM-IA 🧠🧬</h1>
  <p><strong>A Primeira IA Sub-Simbólica Termodinâmica para Edge Devices (Zero-Swapping)</strong></p>

  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Build Status](https://github.com/MrJc01/crompressor-ia/workflows/CROM-IA%20CI%20PIPELINE/badge.svg)](https://github.com/MrJc01/crompressor-ia/actions)
  [![Hardware](https://img.shields.io/badge/Hardware-Intel_i5__IvyBridge-blue.svg)](https://en.wikipedia.org/wiki/Ivy_Bridge_(microarchitecture))
  
  <br>
<br>
<a href="https://huggingface.co/spaces/CromIA/Rosa-Chat-V3.5"><img src="https://img.shields.io/badge/💬_Testar_Online_Agora!-Painel_Cloud_Gratuito-0e7ce6.svg?style=for-the-badge" alt="HuggingFace Space"></a>
<a href="https://huggingface.co/CromIA/CROM-IA-V3.5-Qwen-1.5B-Organic"><img src="https://img.shields.io/badge/🤗_HuggingFace-Motor_GGUF-ff9d00.svg?style=for-the-badge" alt="HuggingFace Models"></a>
</div>

<br>

**📢 ÚLTIMA ATUALIZAÇÃO (ABRIL 2026) — CROM-IA V3.5b (Cloud Edge Deploy):**
Subimos a interface oficial para o modelo Qwen2.5-1.5B-Organic treinado! Você não precisa baixar nada para testar, basta acessar o link do Painel Online acima.

> **⚠️ Atenção sobre a Compressão DNA:** Embora o painel mostre o `RAG Injector DNA` ativado, a versão V3.5b purificou com sucesso o bug da alucinação (Caso Bicarbonato), mas **ainda não foi treinada para emitir ativamente os ponteiros DNA reais (@@XX)**. A arquitetura web e local já possuí o compressor vivo na camada FUSE/Python esperando o texto do LLM apontar, o que será o foco do treinamento na Arquitetura Multi-Brain O(1) (V4.0).

O **CROM-IA** é uma arquitetura revolucionária que permite a execução de Redes Neurais (LLMs Reais como LLaMA e Qwen) em computadores sem Placa de Vídeo dedicada e com menos de **3GB de memória RAM disponível**.

Diferente de quantizações tradicionais, nossa tecnologia substitui a estrutura alocativa da RAM por uma leitura **fractal O(1)** mapeada diretamente do SSD utilizando compressão termodinâmica (FastCDC) e montadores baseados em **FUSE (Filesystem in Userspace)**.

---

## 🚀 Engenharia Core

As inteligências artificiais dominantes engolem dicionários literais na porta de entrada (os chamados BPE Tokenizers). Nós não. 
O CROM infunde nativamente um modelo Qwen / LLaMA a se comunicar de forma **Sub-Simbólica em Base-4 (Radix-4 / DNA)** (`A, T, C, G`), comprimindo e evitando problemas de Swap e paginação pesada do kernel Linux.

### 🛡️ Pipeline Visível:
1. Poda Mágica anti-ruído baseada na **Entropia de Claude Shannon**.
2. **LoRA Fine-Tuning** convertendo a base do LLaMA em motor DNA.
3. Empacotamento CROM-Compression (Dicionário Codebook HNSW Dinâmico).
4. Motor Nativo C/C++ interceptando os arrays virtuais FUSE O(1).
5. RAG (Busca Documental) 100% matemática (BM25 Native TF-IDF).

---

## 📦 Quick Start

> Você não precisará rodar nada como `root`. Tudo opera perfeitamente seguro em _userspace_.

### 1️⃣ Baixar e Preparar
Inicie o ambiente, valide permissões e ative o motor sub-simbólico:
```bash
git clone https://github.com/MrJc01/crompressor-ia.git
cd crompressor-ia
chmod +x scripts/*.sh

# Cria o container base virtual (.venv) e pacotes C++ nativos
source pesquisa/.venv/bin/activate
```

### 2️⃣ Iniciar o Ecossistema CROM Completo
Use um **único comando** SRE Hardened que orquestra tudo na sua máquina:
```bash
./iniciar_chat_real.sh
```

A mágica toda acontecerá:
1. O _Pre-Flight Check_ SRE protege o seu Kernel contra Swap-thrashing (> 90%).
2. O **Watchdog Health Guard** inicia silenciosamente protegendo seus processos.
3. O Backend injeta o RAG CROM e as LLMs neurais.
4. O WebServer **Streaming** levantará na máquina local.
   👉 Acesso em: [http://localhost:8080](http://localhost:8080)

---

## 🧮 Laboratório de DNA (Demo do Backend)

Em painéis antigos, tudo é `String -> Tensor`. 
No site oficial local do CROM-IA que levantará no link da porta 8080, clique em acessar o **"Laboratório Radix-4"**. Você verá como os pesos fluem dentro do FUSE sob demanda e como os pacotes de comunicação transpilam os pacotes pesados em Base-4 super processáveis.

## 🤝 Colaboradores Open Source

Ansioso para ver o que pesquisadores podem alcançar com essa arquitetura C++ pura + Rust.
Por favor, leia nosso [CONTRIBUTING.md](./CONTRIBUTING.md) antes de criar issues de vazamento de RAM!

<br>
<div align="center">
  <i>Construído com obsessão pela DeepMind Architectures / Google GenAI Hackathons.</i>
</div>
