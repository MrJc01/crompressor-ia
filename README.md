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
- `/pesquisa`: Projetos de Prova de Conceito (POCs) isolados e pipelines PyTorch experimentais.
- `crompressor_bin`: O binário compilado que fornece o FUSE mount (SoC CROM).

## 🥼 Primeiros Passos (Laboratório)

Para testar o POC inicial Pytorch + FUSE:
```bash
cd pesquisa/poc_dataloader/
chmod +x run_llm_training.sh
./run_llm_training.sh
```
