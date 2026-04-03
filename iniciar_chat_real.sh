#!/bin/bash
# Orquestrador Inicializador CROM-IA (SRE Hardened)

DIR_BASE="$(dirname "$(realpath "$0")")"
VENV_PYTHON="$DIR_BASE/pesquisa/.venv/bin/python3"
PORT_API=5000
PORT_UI=8080

echo "=================================================="
echo " 🛡️  CROM-IA : ORQUESTRADOR HARDENED "
echo "=================================================="

# 1. Verificacao de Swap (SRE Táticas)
SWAP_TOTAL=$(free -m | grep Swap | awk '{print $2}')
SWAP_USED=$(free -m | grep Swap | awk '{print $3}')
if [ "$SWAP_TOTAL" -gt 0 ]; then
    SWAP_PCT=$(( 100 * SWAP_USED / SWAP_TOTAL ))
    if [ "$SWAP_PCT" -gt 95 ]; then
        echo "[FALHA SRE] O Sistema está usando ${SWAP_PCT}% do Swap."
        echo "A inferência causaria Thrashing pesado. Abortando. Libere memória primeiro."
        exit 1
    fi
fi

# 2. Verificar dependências Python do Backend
if [ ! -f "$VENV_PYTHON" ]; then
    echo "[ERRO] VirtualEnvironment Python não encontrado em pesquisa/.venv/"
    exit 1
fi

# 3. Cleanup: Evita instâncias fantasmas na porta local
echo "[1/4] Higienizando ambiente anterior..."
fuser -k -9 $PORT_API/tcp 2>/dev/null || true
fuser -k -9 $PORT_UI/tcp 2>/dev/null || true
pkill -f "health_guard.sh" 2>/dev/null || true
fusermount -u "$DIR_BASE/mnt_crom" 2>/dev/null || true

# 4. Iniciar Sub-rotinas do Sistema Distribuído Background
echo "[2/4] Armando Escudos (Health Guard Daemon)..."
bash "$DIR_BASE/scripts/health_guard.sh" &
GUARD_PID=$!

echo "[3/4] Levantando Servidor Frontend (Porta $PORT_UI)..."
cd "$DIR_BASE"
python3 -m http.server $PORT_UI --directory visualizador-sre > /dev/null 2>&1 &
UI_PID=$!

# Função de Graceful Teardown (C-C catch)
function teardown() {
    echo -e "\n[TEARDOWN] Desligando rede neural e daemons..."
    kill -TERM "$GUARD_PID" 2>/dev/null
    kill -TERM "$UI_PID" 2>/dev/null
    fusermount -u "$DIR_BASE/mnt_crom" 2>/dev/null || true
    echo "[OK] Ambiente finalizado com segurança forense (Zero Footprint)."
    exit 0
}
trap teardown SIGINT SIGTERM

echo "=================================================="
echo " 🟢 TUDO ONLINE: Acesse http://localhost:$PORT_UI"
echo " 📡 Backend rodando em :$PORT_API"
echo " Pressione Ctrl+C para derrubar tudo."
echo "=================================================="

echo "[4/4] Injetando Kernel de Inferência Python..."
cd "$DIR_BASE/visualizador-sre"
"$VENV_PYTHON" server.py
