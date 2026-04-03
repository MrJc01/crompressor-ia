#!/bin/bash
# CROM-IA OOM/Swap Watchdog SRE Daemon

LOG_FILE="../logs/health_guard.log"
mkdir -p ../logs

echo "[$(date)] SRE Health Guard Ativado." | tee -a "$LOG_FILE"

while true; do
    # Ler percentual de Swap usado (via free -m)
    SWAP_INFO=$(free -m | grep Swap)
    SWAP_TOTAL=$(echo "$SWAP_INFO" | awk '{print $2}')
    SWAP_USED=$(echo "$SWAP_INFO" | awk '{print $3}')
    
    if [ "$SWAP_TOTAL" -gt 0 ]; then
        SWAP_PCT=$(( 100 * SWAP_USED / SWAP_TOTAL ))
    else
        SWAP_PCT=0
    fi

    # Analisar RSS (Memoria Residente) do servidor Python
    PYTHON_PID=$(pgrep -f "server.py" | head -n 1)
    
    if [ ! -z "$PYTHON_PID" ]; then
        RSS_KB=$(ps -o rss= -p "$PYTHON_PID")
        RSS_MB=$(( RSS_KB / 1024 ))
        
        # Guard: Se RSS Vazar acima de 2.5GB, abortar o motor pra salvar o Kernel
        if [ "$RSS_MB" -gt 2500 ]; then
            echo "[$(date)] [FATAL] Memory Leak Critico detectado no python (RSS: ${RSS_MB}MB)! Disparando SIGKILL. 🚀💥" | tee -a "$LOG_FILE"
            kill -9 "$PYTHON_PID"
        fi
    fi

    # Guard: Se SWAP passar de 96%, estamos prestes a travar = Abortar
    if [ "$SWAP_PCT" -gt 96 ]; then
        echo "[$(date)] [FATAL] Swap critical mass (96%). Destruindo montagens para resgatar Edge Device! 💥" | tee -a "$LOG_FILE"
        fusermount -u ../mnt_crom 2>/dev/null || true
        kill -9 "$PYTHON_PID" 2>/dev/null || true
        exit 1
    fi

    sleep 5
done
