#!/bin/bash
# ==============================================================================
# ⌬ CROM-IA v4.3.5 — ATOM DECOMPRESSION TEST
# ==============================================================================

DIR_BASE="$(dirname "$(realpath "$0")")"
LLAMA_CLI="$DIR_BASE/pesquisa/poc_llama_cpp_fuse/llama.cpp/build/bin/llama-cli"
MODELO_VFS="$DIR_BASE/models/CROM-IA_v4.3_Qwen3.5-2B.gguf"

PROMPT_FILE=$(mktemp /tmp/crom_atom_test_XXXXXX.txt)
cat <<EOF > "$PROMPT_FILE"
### Instruction:
Você é uma Célula CROM. Responda usando Átomos DNA (⌬) sempre que possível.

### Input:
qual seu status operacional?

### Response:
EOF

echo "--- INICIANDO TESTE DE ÁTOMOS (v4.3.5) ---"
python3 "$DIR_BASE/scripts/elite_wrapper_v43.py" \
    "$LLAMA_CLI" \
    -m "$MODELO_VFS" \
    -n 64 \
    --temp 0 \
    -f "$PROMPT_FILE" \
    --log-disable

rm -f "$PROMPT_FILE"
echo -e "\n--- TESTE FINALIZADO ---"
