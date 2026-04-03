#!/bin/bash
# ╔══════════════════════════════════════════════════════════════╗
# ║  🧬 CROM-IA V2: Pipeline DNA Comprimido → Texto Humano     ║
# ║                                                              ║
# ║  Arquitetura:                                                ║
# ║  llama-cli (10 t/s) → DNA tokens → dna_decoder.py (O(1))   ║
# ║  → texto humano expandido (~30 palavras/s com taxa 1:3)     ║
# ╚══════════════════════════════════════════════════════════════╝

set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Binário otimizado para Ivy Bridge (AVX) — dá ~10 t/s
LLAMA_CLI="/home/j/Área de trabalho/crompressor/pesquisa/ia_llm/102-native_llm_humble_pc/bin/llama-cli"

# Modelo DNA V2 treinado com codebook 1:3
MODELO="$BASE_DIR/models/crom-dna-1x3-fixo.gguf"

# Codebook e decoder
CODEBOOK="$BASE_DIR/codebooks/codebook_1x3_fixo.json"
DECODER="$BASE_DIR/scripts/dna_decoder.py"

# Parâmetros
TAXA="${1:-1x3}"
MODO="${2:-fixo}"
THREADS=2
CTX=1024
MAX_TOKENS=128
TEMP=0.2

# Override se passaram parâmetros
if [ "$TAXA" != "1x3" ] || [ "$MODO" != "fixo" ]; then
    MODELO="$BASE_DIR/models/crom-dna-${TAXA}-${MODO}.gguf"
    CODEBOOK="$BASE_DIR/codebooks/codebook_${TAXA}_${MODO}.json"
fi

# Verificações
for FILE in "$LLAMA_CLI" "$MODELO" "$CODEBOOK" "$DECODER"; do
    if [ ! -f "$FILE" ]; then
        echo "❌ Não encontrado: $FILE"
        exit 1
    fi
done

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  🧬 CROM-IA V2: DNA Compression Pipeline (Taxa ${TAXA})   ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Motor     : llama-cli (Ivy Bridge AVX, nativo C++)     ║"
echo "║  Modelo    : $(basename "$MODELO")"
echo "║  Codebook  : $(basename "$CODEBOOK")"
echo "║  Decoder   : dna_decoder.py (hashmap O(1))              ║"
echo "║  Threads   : $THREADS | Ctx: $CTX | Temp: $TEMP         ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  PIPELINE: LLM → DNA tokens → Decoder → Texto Humano   ║"
echo "║  Meta: Cada token DNA = 3 palavras → 30 palavras/s      ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

SYSTEM_PROMPT="Você é um compressor CROM DNA (taxa ${TAXA/x/:}). Comprima a resposta usando códigos do codebook semântico DNA. Use prefixo @@ para palavras sem código. Responda APENAS com códigos DNA."

echo "[!] Digite sua pergunta e aguarde a decodificação supersônica."
echo "================================================================"

"$LLAMA_CLI" \
    -m "$MODELO" \
    --threads $THREADS \
    -c $CTX \
    -n $MAX_TOKENS \
    --temp $TEMP \
    --repeat-penalty 1.18 \
    -cnv \
    -sys "$SYSTEM_PROMPT" \
    2>/dev/null | python3 "$DECODER" --codebook "$CODEBOOK"

echo ""
echo "[SISTEMA] Motor C++ Desativado."
