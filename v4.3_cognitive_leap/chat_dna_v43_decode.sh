#!/bin/bash

# =================================================================
# 🧬 CROM-IA v4.3: DECODIFICADOR QWEN 3.5 DNA
# =================================================================
# Este script aciona o novo Chassis 2B para decodificação Radix-4
# =================================================================

MODELO="/home/j/Área de trabalho/crompressor-ia/models/CROM-IA_v4.3_Qwen3.5-2B.gguf"
# Caminho do binário nativo de alto desempenho
BINARIO_NATIVO="/home/j/Área de trabalho/crompressor/pesquisa/ia_llm/102-native_llm_humble_pc/bin/llama-cli"

if [ ! -f "$BINARIO_NATIVO" ]; then
    echo "[ERRO] Motor nativo não encontrado em: $BINARIO_NATIVO"
    exit 1
fi

echo "=================================================="
echo " ⌬  CROM-IA v4.3 : QWEN 3.5 2B DECODER"
echo "=================================================="
echo " Chassis: $(basename "$MODELO")"
echo " Template: ChatML | Contexto: 4k"
echo "=================================================="

"$BINARIO_NATIVO" \
    -m "$MODELO" \
    --threads 4 \
    -c 4096 \
    -n 512 \
    --temp 0.1 \
    --repeat_penalty 1.1 \
    --color \
    -cnv \
    --chat-template qwen \
    -p "<|im_start|>system\nVocê é uma Célula CROM. Decodifique a sequência DNA CROM Base-4 para Português.<|im_end|>\n<|im_start|>user\nDecodifique: "

echo ""
echo "[SISTEMA] Salto Cognitivo Finalizado."
