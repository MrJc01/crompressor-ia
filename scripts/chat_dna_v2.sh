#!/bin/bash
# ╔══════════════════════════════════════════════════════════════╗
# ║  🧬 CROM-IA V2 SRE: Chat DNA Batch Blindado (Final)          ║
# ║  Usa `script` para capturar output TTY direto do llama-cli   ║
# ║  Extrai resposta DNA com sed, decodifica com Python O(1)     ║
# ╚══════════════════════════════════════════════════════════════╝

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA_CLI="/home/j/Área de trabalho/crompressor/pesquisa/ia_llm/102-native_llm_humble_pc/bin/llama-cli"

TAXA="${1:-1x5}"
MODO="${2:-fixo}"

MODELO="$BASE_DIR/models/crom-dna-${TAXA}-${MODO}.gguf"
CODEBOOK="$BASE_DIR/codebooks/codebook_${TAXA}_${MODO}.json"
DECODER="$BASE_DIR/scripts/dna_decoder.py"

# Verificações
for FILE in "$LLAMA_CLI" "$MODELO" "$CODEBOOK" "$DECODER"; do
    if [ ! -f "$FILE" ]; then
        echo "❌ Não encontrado: $FILE"
        exit 1
    fi
done

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  🧬 CROM-IA Chat DNA (Taxa ${TAXA}) — Motor AVX Puro     ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Modelo   : crom-dna-${TAXA}-${MODO}.gguf"
echo "║  Codebook : codebook_${TAXA}_${MODO}.json"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "[!] Digite 'sair' para encerrar."
echo ""

SYSTEM_PROMPT="Você é um compressor CROM DNA (taxa ${TAXA/x/:}). Comprima a resposta usando códigos do codebook semântico DNA. Use prefixo @@ para palavras sem código. Responda APENAS com códigos DNA."

TMP_TTY="/tmp/crom_tty_$$.txt"

while true; do
    echo -ne "\033[0;36m👤 Você:\033[0m "
    read -r PERGUNTA

    if [ -z "$PERGUNTA" ]; then continue; fi
    if [ "$PERGUNTA" = "sair" ] || [ "$PERGUNTA" = "exit" ]; then
        echo "Saindo..."
        rm -f "$TMP_TTY"
        break
    fi

    # Roda llama-cli em single-turn, capturando TTY via `script`
    # -> `script -qec` cria um pseudo-terminal e grava TUDO num arquivo.
    # -> `--single-turn` faz a Llama responder UMA vez e sair sozinha.
    # -> `--no-display-prompt` não re-ecoa a pergunta na saída.
    # -> `--no-perf` esconde métricas de velocidade.
    # -> `2>/dev/null` mata os logs de backend do stderr.
    script -qec "\"$LLAMA_CLI\" \
        -m \"$MODELO\" \
        -c 1024 -t 4 -n 128 \
        --temp 0.2 --repeat-penalty 1.18 \
        --single-turn --no-display-prompt --no-perf \
        -sys \"$SYSTEM_PROMPT\" \
        -p \"$PERGUNTA\" \
        2>/dev/null" "$TMP_TTY" > /dev/null 2>&1

    # Extrai SOMENTE a linha de resposta DNA (após "> prompt", pula linha vazia, pega resposta)
    DNA_RAW=$(sed -n '/^> /{n;n;p;q}' "$TMP_TTY" | tr -d '\r')

    if [ -z "$DNA_RAW" ]; then
        echo -e "\033[0;31m[!] Sem resposta do motor.\033[0m"
        continue
    fi

    # Decodifica DNA → Texto Humano via Python O(1) decoder
    TEXTO_HUMANO=$(echo "$DNA_RAW" | python3 "$DECODER" --codebook "$CODEBOOK" --quiet 2>/dev/null)

    echo -e "\033[0;32m🤖 CROM:\033[0m $TEXTO_HUMANO"
    echo ""
done
