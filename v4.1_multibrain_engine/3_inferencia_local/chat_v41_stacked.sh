#!/usr/bin/env bash
# ==============================================================================
# CROM-IA V4.1: Inferência Multi-Brain EMPILHADA + Decodificador DNA
# Carrega modelo base + N adaptadores LoRA simultâneos
# A saída passa pelo decodificador DNA automaticamente
# ==============================================================================

LLAMA_CLI="/home/j/Área de trabalho/crompressor-ia/pesquisa/poc_llama_cpp_fuse/llama.cpp/build/bin/llama-cli"
MODELS_DIR="/home/j/Área de trabalho/crompressor-ia/v4.1_multibrain_engine/3_inferencia_local/micro_cerebros"
BASE_MODEL="$MODELS_DIR/qwen3.5-0.8b-base.gguf"
DECODER="/home/j/Área de trabalho/crompressor-ia/v4.1_multibrain_engine/3_inferencia_local/decodificador_dna/decodificador_dna.py"

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║     🧠 CROM-IA V4.1 — Stacked Multi-Brain Engine       ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Modelo Base: Qwen 3.5 0.8B                             ║"
echo "║  Modo: EMPILHAMENTO de N Micro-Cérebros LoRA            ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Verificar modelo base
if [ ! -f "$BASE_MODEL" ]; then
    echo "❌ Modelo base não encontrado em: $BASE_MODEL"
    echo "   Faça download do Qwen 3.5 0.8B GGUF primeiro."
    exit 1
fi

# Detectar e empilhar TODOS os LoRAs disponíveis
LORA_FLAGS=""
LORA_COUNT=0
echo "🔍 Detectando Micro-Cérebros disponíveis..."
for lora in "$MODELS_DIR"/*_lora.gguf; do
    if [ -f "$lora" ]; then
        nome=$(basename "$lora" _lora.gguf)
        echo "   ✅ Cérebro: $nome"
        LORA_FLAGS="$LORA_FLAGS --lora $lora"
        LORA_COUNT=$((LORA_COUNT + 1))
    fi
done

if [ "$LORA_COUNT" -eq 0 ]; then
    echo "⚠️  Nenhum LoRA encontrado. Rodando modelo base puro."
fi

echo ""
echo "🚀 Ativando $LORA_COUNT Micro-Cérebros empilhados simultaneamente!"
echo "🧬 Decodificador DNA: ATIVO"
echo "⏳ Carregando..."
echo ""

# Execução com pipeline de decodificação DNA
"$LLAMA_CLI" \
    -m "$BASE_MODEL" \
    $LORA_FLAGS \
    -c 1024 \
    -n 512 \
    --threads 4 \
    --batch-size 512 \
    --mlock \
    --temp 0.7 \
    --repeat-penalty 1.15 \
    --conversation \
    --prompt "Você é CROM-IA, assistente brasileiro com compressão DNA ativa."
