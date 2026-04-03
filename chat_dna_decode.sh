#!/bin/bash

# Este script aciona o Cérebro com ativação LoRA focada em Decodificação DNA Radix-4

MODELO="/home/j/Área de trabalho/crompressor-ia/models/qwen2.5-crom-dna.gguf"
BINARIO_NATIVO="/home/j/Área de trabalho/crompressor/pesquisa/ia_llm/102-native_llm_humble_pc/bin/llama-cli"

if [ ! -f "$BINARIO_NATIVO" ]; then
    echo "[ERRO] Motor nativo não encontrado!"
    exit 1
fi

echo "=================================================="
echo " 🧬 FASE B: MODO DECODIFICADOR (DNA -> HUMANO)"
echo "=================================================="
echo " Digite sequências como: TACGTACGCCGGATCT"
echo "=================================================="

"$BINARIO_NATIVO" \
    -m "$MODELO" \
    --threads 2 \
    -c 1024 \
    -n 256 \
    --temp 0.2 \
    --repeat_penalty 1.18 \
    -cnv \
    -p "Decodifique a sequência biológica DNA CROM Base-4 para linguagem humana em Português."

echo ""
echo "[SISTEMA] Motor C++ Desativado."
