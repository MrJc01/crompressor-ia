#!/bin/bash
# ========================================
# CROM-IA: Inicializador do Chat Real (LLM)
# ========================================
set -e

VENV="/home/j/Área de trabalho/crompressor-ia/pesquisa/.venv"
MODELS_DIR="/home/j/Área de trabalho/crompressor-ia/models"
MODEL_FILE="$MODELS_DIR/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
SERVER_DIR="/home/j/Área de trabalho/crompressor-ia/visualizador-sre"

echo "========================================="
echo "   CROM-IA: Inicializador Chat Real LLM"
echo "========================================="

# 1. Ativar virtualenv
source "$VENV/bin/activate"
echo "[OK] Virtualenv ativado"

# 2. Baixar modelo se nao existir
if [ ! -f "$MODEL_FILE" ]; then
    echo "[DL] Baixando TinyLlama-1.1B Chat Q4_K_M (~640MB)..."
    mkdir -p "$MODELS_DIR"
    python3 -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id='TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF',
    filename='tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
    local_dir='$MODELS_DIR',
)
print(f'[OK] Modelo baixado: {path}')
"
else
    echo "[OK] Modelo encontrado: $MODEL_FILE"
    echo "     Tamanho: $(du -h "$MODEL_FILE" | cut -f1)"
fi

# 3. Levantar servidor LLM (porta 5000) em background
echo ""
echo "[START] Levantando servidor LLM na porta 5000..."
python3 "$SERVER_DIR/server.py" &
LLM_PID=$!
echo "[OK] Servidor LLM PID: $LLM_PID"

# 4. Levantar servidor frontend (porta 8080) em background
echo "[START] Levantando frontend na porta 8080..."
cd "$SERVER_DIR"
python3 -m http.server 8080 &
HTTP_PID=$!
echo "[OK] Frontend PID: $HTTP_PID"

echo ""
echo "========================================="
echo "   CROM-IA ONLINE"
echo "========================================="
echo "   Chat:     http://localhost:8080/chat.html"
echo "   Portal:   http://localhost:8080"
echo "   API:      http://localhost:5000/api/status"
echo "========================================="
echo "   Ctrl+C para encerrar tudo"
echo "========================================="

# Trap para matar ambos os processos no Ctrl+C
trap "echo '[TEARDOWN] Encerrando...'; kill $LLM_PID $HTTP_PID 2>/dev/null; exit 0" SIGINT SIGTERM

# Aguardar
wait
