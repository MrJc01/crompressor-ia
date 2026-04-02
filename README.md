# 🧬 CROM-IA (Crompressor Large Language Model)

> _"Evoluindo a Inteligência Artificial através da Compressão Termodinâmica"_

O **Crompressor-IA** é um repositório satélite criado a partir da Engenharia Estrutural do Motor V23 do Crompressor. O objetivo é unificar **Small World Graphs (HNSW)**, **Delta Compression** e **FUSE Cascading** com os gargalos massivos de I/O em Treinamentos de Modelos de Linguagem (PyTorch, JAX).

## 🚀 Como isso funciona?

Diferente de um modelo como LLaMA ou GPT que processa arquivos de texto massivos (.jsonl) extraídos cegamente, o CROM-IA treina lendo a _Matriz de Entropia_ (o `.cromdb` codebook associado aos resíduos).

1. **Acesso O(1) de Dataloaders:** O `CromIterableDataset` do PyTorch não carrega memória desnecessária. Ele usa CGO FUSE para realizar um bypass quântico de informações já processadas.
2. **Codebook Tokens (Embeddings Nativos):** Os IDs LSH do Crompressor servem como fundação vetorial para a IA. Prever o "próximo token" aqui significa prever _Próximos Fractais de Realidade Abstrata_.
3. **KV Cache em Disco:** A retenção de memórias sequenciais ocorre no espaço FUSE, garantindo um "Context Window" virtualmente infinito durante a inferência.

## 📂 Estrutura Físico-Estocástica

- `/docs`: Teses e fundamentação matemática da integração CROM-LLM.
- `/pesquisa`: Projetos de Prova de Conceito (POCs) isolados e pipelines PyTorch experimentais. Abaixo as 9 frentes SRE CROM-IA em atividade:
  - `poc_dataloader` - A raiz de Dataloading FUSE com BPE bypass em Python.
  - `poc_llama_cpp_fuse` - Modelos GGUF e inferência via C++ com CROM.
  - `poc_kv_cache_to_disk` - Inodes FUSE sustentando O(1) de Contexto Infinito em Inferência local LLaMA.
  - `poc_rag_crom` - O Motor Cosenoidal HNSW de CROM atuando puramente como Vectordb.
  - `poc_tokenless_transformers` - Alimentação de FastCDC hashes diretamente como tokens.
  - `poc_p2p_weight_delta` - P2P/Kademlia Edge Swarm para treinar pesos neurais através de transferências micro (.crom).
  - `poc_anti_entropy_pruning` - Escudos de limite de Shannon contra datasets de baixa qualidade (WebCrawling).
  - `poc_safetensors_mmap` - Kernels de C conversando com inodes de SQLite FUSE on-the-fly sem CPU Copy.
  - `poc_genetic_dataset_streaming` - Reconstrução do dicíonário Codebook HNSW em Runtime via Backpropagation Loss.
  - `poc_telemetria_sre_ai` - Dashboards I/O BPFTrace para certificar estabilidade em NVMe/Memória Restrita.

- `crompressor_bin`: O binário compilado que fornece o FUSE mount (SoC CROM).
- `/visualizador-sre`: Frontend Edge Dashboard Premium em arquitetura HTML/CSS/JS nativa, voltado a telemetria, PageFaults Mmap e taxa de bloqueio (Anti-Entropy Pruning) logados do backend em FUSE e AI Bypass. Sem frameworks pesados garantindo zero bloqueio da Edge CPU.

## 🥼 Primeiros Passos (Laboratório)

Para testar o POC inicial Pytorch + FUSE:
```bash
cd pesquisa/poc_dataloader/
chmod +x run_llm_training.sh
./run_llm_training.sh
```
