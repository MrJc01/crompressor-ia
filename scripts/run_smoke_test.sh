#!/bin/bash
# CROM-IA Teste de Fumaça (Pre-Flight Check)

echo "=================================================="
echo " 🌫️  CROM-IA : SRE SMOKE TEST"
echo "=================================================="

FAILS=0

function check() {
    if $1; then
        echo " ✔️  [PASS] $2"
    else
        echo " ❌ [FAIL] $2"
        FAILS=$((FAILS + 1))
    fi
}

check "[ -x ./crompressor_bin ]" "Binário CROM-IA existe e é executável"
check "ls models/*.gguf >/dev/null 2>&1" "Modelos GGUF encontrados no diretório"
check "[ -x ./pesquisa/ia_llm/102-native_llm_humble_pc/bin/llama-cli ]" "Binário Nativo C++ (llama-cli) encontrado"
check "[ -d ./pesquisa/.venv ]" "VirtualEnv Python configurada"
check "[ -f ./visualizador-sre/server.py ]" "Servidor Python API encontrado"

SWAP_TOTAL=$(free -m | grep Swap | awk '{print $2}')
SWAP_USED=$(free -m | grep Swap | awk '{print $3}')
if [ "$SWAP_TOTAL" -gt 0 ]; then
    SWAP_PCT=$(( 100 * SWAP_USED / SWAP_TOTAL ))
    check "[ \"$SWAP_PCT\" -lt 85 ]" "Memória Swap saudável ($SWAP_PCT% em uso)"
else
    check "true" "Swap não alocado (bypass verification)"
fi

echo "=================================================="
if [ $FAILS -eq 0 ]; then
    echo " 🚀 TODOS OS SISTEMAS VERDES! Pronto para decolagem livre de FUSE."
    exit 0
else
    echo " 💥 $FAILS testes falharam. Hardening SRE impediu inicialização."
    exit 1
fi
