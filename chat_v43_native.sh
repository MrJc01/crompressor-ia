#!/bin/bash
# ⌬ CROM-IA v4.7.4 — OPERAÇÃO GAIA-PURGE
DIR_BASE="$(dirname "$(realpath "$0")")"
LLAMA_CLI="$DIR_BASE/pesquisa/poc_llama_cpp_fuse/llama.cpp/build/bin/llama-cli"
MODELO="$DIR_BASE/models/CROM-IA_v4.3_Qwen3.5-2B.gguf"
WRAPPER="$DIR_BASE/scripts/elite_wrapper_v43.py"

# Fix mktemp: XXXXXX deve ser o sufixo no Linux
PROMPT_FILE=$(mktemp /tmp/crom_purge_XXXXXX)

cat <<EOF > "$PROMPT_FILE"
<|im_start|>system
SISTEMA: CROM-IA KERNEL. MODO: ATOMS_ONLY.
RESPOSTA: ⌬[CÓDIGO_TRADUZIDO].
ONE-SHOT:
User: quem e voce?
Assistant: ⌬P_ARQUITETURA<|im_end|>
<|im_start|>user
EOF

echo -e "\033[0;32m🔥 KERNEL V4.7.4 ATIVO. SINAL PURO.\033[0m\n"

python3 "$WRAPPER" "$LLAMA_CLI" \
    -m "$MODELO" \
    -n 256 \
    --threads 4 \
    --flash-attn on \
    --log-disable \
    --temp 0 \
    --top-k 1 \
    -r "<|im_end|>" \
    -f "$PROMPT_FILE" \
    --conversation
