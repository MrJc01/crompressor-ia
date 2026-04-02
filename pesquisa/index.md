# Índice de Pesquisa e Engenharia: Laboratório CROM-IA

Bem-vindo ao diretório raiz de pesquisa e Provas de Conceito (POCs) do ecossistema CROM-IA.
Este diretório hospeda a borda de desenvolvimento avançado (Edge Labs) onde testamos e validamos viabilidades termodinâmicas, manipulação de arquivos CROM via FUSE (Filesystem in Userspace) e inferência isolada de Redes Neurais com bypass de Memória(Zero-Copy).

## 🚀 Arquitetura Central

A premissa mestre testada neste laboratório é o **FUSE Cascading Bypass**: ao invés de usar o processador e alocar *GigaBytes* maciços de RAM para rodar Large Language Models (LLMs), a rede lê Padrões Cosenoidais (Fractais HNSW) encapsulados em dicionários `.cromdb` instantaneamente, reduzindo gargalos críticos de *Swapping* e de latência.

---

## 📂 Visão Geral das Baterias (As 9 POCs SRE)

Para comprovar sistematicamente essa super-hipótese, este laboratório contém as seguintes sub-provas de conceito unitárias, orquestradas pelo master script de testes `run_all_baterias.sh`.

### Fase 1: Infraestrutura C++ & Mmap
* **`poc_llama_cpp_fuse`**: Prova de bypass de VRAM atracando FUSE no motor LLM real (Ex: `TinyLlama-1.1B`). Provou-se que não usamos 700MB de RAM de Host, mas sim 0MB de RAM usando apontadores FUSE Virtuais Mmap.
* **`poc_kv_cache_to_disk`**: O grande tabu do "Contexto Infinito". Validação gravando Tensores Attention no *SQLite Blob Limitado* da Storage, não sobrecarregando tensores repetidos na memória de vídeo.

### Fase 2: Automação Transformer (Modo Zero-Copy)
* **`poc_safetensors_mmap`**: Empacotamento bypass. Ignorando CPU e roteando a rede neural inteira sobre o Kernel nativo.
* **`poc_tokenless_transformers`**: A CROM-IA não usa "Letras/Sílabas Estáticas" (BPE / SentencePiece), usa os Hashes originados pelo algoritmo iterativo *FastCDC*. Teste O(1) Embedders.
* **`poc_rag_crom`**: Recuperação informacional por dict puro, ignorando a dependência do pesado VectorDB de hardware.

### Fase 3: Teoria da Informação e Swarm P2P
* **`poc_anti_entropy_pruning`**: Usando o isolamento *Claude Shannon* (H > 7.5) para forçar que o treinamento/inferência blinde-se contra lixo de base hexadecimal e randomizado, polpando ciclos neurais iterativos.
* **`poc_genetic_dataset_streaming`**: Ficha de Degradação Sistêmica CROM Node Matrix Loss.
* **`poc_p2p_weight_delta`**: Falso mock *Kademlia* validando que só faremos FastCDC-hashes para enviar diferenças neurais aos peers atrelados no laboratório.
* **`poc_telemetria_sre_ai`**: Sondas *psutil / htop* atestando que os 3GB limitantes (Edge Constraint Hardware) funcionam soltos sob a inferência pesada (Zero-Swap).

---

## 🧬 O DataLoader Nativo (`poc_dataloader`)

Diretório mais importante da pasta de Pesquisa: o **CROM FUSE Cascading PyTorch Dataloader**.
Aqui criamos o *overlay* via Shell que funde (mount) o `SquashFS` com o `dict.cromdb` para virar um arquivo invisível dentro do PyTorch (`train_llm_poc.py`).

Neste laboratório, implementamos matematicamente pela primeira vez o `calc_shannon_entropy` acoplado ao *generator*, descartando atômica e virtualmente os blocos que superassem *7.5 H* e assim isolando o Motor de IA do lixo orgânico (`dataset_raw/`).

---

## 🛠️ Como Usar (Runners)

1. **Auditoria Geral CROM-IA**: Para revisar os ponteiros das 9 Baterias:
```bash
./run_all_baterias.sh
```

2. **Dataloader Injetável Real**: Para experimentar fisicamente o FUSE sendo acoplado e ejetado (*Zero-Footprint* Tear down SRE):
```bash
cd poc_dataloader
./run_llm_training.sh
```

## 🌌 Visualizador Táctico Web
Não incluso neste diretório de infraestrutura pesada, criamos uma contraparte *FrontEnd* Vanilla na pasta raiz gêmea chamada `visualizador-sre`. Projetada sob severo "Estethic Wow Factor" para espelhar as métricas testadas nesta área de *pesquisa/* em interfaces amigáveis operacionais.
