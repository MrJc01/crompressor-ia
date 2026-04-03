#!/bin/bash

# Este script aciona o Cérebro com ativação LoRA focada em Codificação DNA Radix-4

MODELO="/home/j/Área de trabalho/crompressor-ia/models/qwen2.5-crom-dna.gguf"
BINARIO_NATIVO="/home/j/Área de trabalho/crompressor/pesquisa/ia_llm/102-native_llm_humble_pc/bin/llama-cli"

if [ ! -f "$BINARIO_NATIVO" ]; then
    echo "[ERRO] Motor nativo não encontrado!"
    exit 1
fi

echo "=================================================="
echo " 🧬 FASE A: MODO CODIFICADOR (HUMANO -> DNA)"
echo "=================================================="
echo " Pergunte em português e ele cuspirá DNA."
echo "=================================================="

"$BINARIO_NATIVO" \
    -m "$MODELO" \
    --threads 2 \
    -c 1024 \
    -n 256 \
    --temp 0.2 \
    --repeat_penalty 1.18 \
    -cnv \
    -p "Você é uma Célula CROM. Transcreva a resposta como Cadeia Quaternária DNA CROM Base-4 (usar apenas A, T, C, G)."

echo ""
echo "[SISTEMA] Motor C++ Desativado."
