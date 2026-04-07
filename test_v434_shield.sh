#!/bin/bash
# ==============================================================================
# ⌬ CROM-IA v4.3.4 — SRE TEST (Single Shot)
# ==============================================================================

DIR_BASE="$(dirname "$(realpath "$0")")"
LLAMA_CLI="$DIR_BASE/pesquisa/poc_llama_cpp_fuse/llama.cpp/build/bin/llama-cli"
MODELO_VFS="$DIR_BASE/models/CROM-IA_v4.3_Qwen3.5-2B.gguf"

PROMPT_FILE=$(mktemp /tmp/crom_test_XXXXXX.txt)
cat <<EOF > "$PROMPT_FILE"
### Instruction:
Você é uma Célula CROM. Transcreva a resposta como Cadeia Quaternária DNA CROM Base-4 (usar apenas A, T, C, G).

### Input:
quem e voce?

### Response:
EOF

echo "--- INICIANDO TESTE CONTROLADO (v4.3.4) ---"
python3 "$DIR_BASE/scripts/elite_wrapper_v43.py" \
    "$LLAMA_CLI" \
    -m "$MODELO_VFS" \
    -n 64 \
    --temp 0 \
    -f "$PROMPT_FILE" \
    --log-disable

rm -f "$PROMPT_FILE"
echo -e "\n--- TESTE FINALIZADO ---"
