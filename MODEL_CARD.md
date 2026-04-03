---
license: mit
language:
  - pt
tags:
  - dna-encoding
  - radix-4
  - edge-inference
  - zero-swapping
  - fuse-mmap
  - thermodynamic-compression
  - sub-symbolic
datasets:
  - FreedomIntelligence/alpaca-gpt4-portuguese
base_model: Qwen/Qwen2.5-0.5B-Instruct
pipeline_tag: text-generation
---

# CROM-IA V1 DNA 🧬

**A primeira IA Sub-Simbólica Termodinâmica para Edge Devices**

## Descrição

Este modelo é um Fine-Tune LoRA do **Qwen 2.5 0.5B Instruct**, treinado com um dataset especial que codifica instruções em sequências **Radix-4 (DNA: A, T, C, G)**.

O objetivo é permitir inferência de alta performance em hardware extremamente limitado (CPU-only, sem GPU) através de compressão termodinâmica e montagem FUSE com mmap zero-copy.

## Métricas de Performance

| Métrica | Valor |
|---------|-------|
| **Hardware** | Intel i5-3320M (Ivy Bridge, 2012) |
| **RAM Utilizada** | < 700 MB via FUSE mmap |
| **GPU Necessária** | Nenhuma (0 layers offloaded) |
| **Velocidade (Prompt)** | ~37.5 t/s |
| **Velocidade (Geração)** | ~11.4 t/s |
| **Quantização** | Q4_K_M (GGUF) |
| **Tamanho** | ~380 MB |

## Como Usar

### Opção 1: Via CROM-IA Engine (Recomendado)
```bash
git clone https://github.com/MrJc01/crompressor-ia.git
cd crompressor-ia
# Coloque o .gguf na pasta models/
./iniciar_chat_real.sh
```

### Opção 2: Via llama.cpp Direto
```bash
./llama-cli \
    -m qwen2.5-crom-dna.gguf \
    --threads 4 \
    -c 512 -b 256 -n 512 \
    --temp 0.1 \
    --repeat_penalty 1.15 \
    -cnv \
    -p "You are CROM-IA, an AI assistant. Respond in Portuguese."
```

## Arquitetura de Treinamento

1. **Dataset**: Alpaca-GPT4 PT com filtragem de Entropia de Shannon (H > 7.5 descartado)
2. **Método**: LoRA (r=16, alpha=32) sobre Qwen 2.5 0.5B
3. **Codificação DNA**: Bytes UTF-8 → pares de 2 bits → bases nitrogenadas (A=00, T=01, C=10, G=11)
4. **Bypass VRAM**: Modelo servido via FUSE VFS + mmap, RSS constante

## Limitações

- Modelo compacto (0.5B params) — ideal para tarefas concisas e diretas
- Respostas longas podem perder coerência após ~200 tokens
- Otimizado para português brasileiro

## Licença

MIT — Livre para uso pessoal, acadêmico e comercial.

## Links

- **GitHub**: [crompressor-ia](https://github.com/MrJc01/crompressor-ia)
- **Organização**: [CromIA no HuggingFace](https://huggingface.co/CromIA)
