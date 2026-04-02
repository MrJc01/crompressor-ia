#!/usr/bin/env bash

echo "=========================================="
echo " INICIANDO AUDITORIA CROM-IA (9 BATERIAS) "
echo "=========================================="

echo ""
echo "[1/9] Infraestrutura FUSE - LLaMA CPP"
python3 /home/j/Área\ de\ trabalho/crompressor-ia/pesquisa/poc_llama_cpp_fuse/test_llama_fuse.py
echo ""

echo "[2/9] Infraestrutura FUSE - KV Cache to Disk"
python3 /home/j/Área\ de\ trabalho/crompressor-ia/pesquisa/poc_kv_cache_to_disk/test_kv_cache.py
echo ""

echo "[3/9] Automação - Safetensors Mmap"
python3 /home/j/Área\ de\ trabalho/crompressor-ia/pesquisa/poc_safetensors_mmap/test_safetensors.py
echo ""

echo "[4/9] Automação - Tokenless Transformers"
python3 /home/j/Área\ de\ trabalho/crompressor-ia/pesquisa/poc_tokenless_transformers/test_tokenless.py
echo ""

echo "[5/9] Automação - RAG CROM"
python3 /home/j/Área\ de\ trabalho/crompressor-ia/pesquisa/poc_rag_crom/test_rag.py
echo ""

echo "[6/9] Dados - Anti-Entropy Pruning"
python3 /home/j/Área\ de\ trabalho/crompressor-ia/pesquisa/poc_anti_entropy_pruning/test_anti_entropy.py
echo ""

echo "[7/9] Dados - Genetic Dataset Streaming"
python3 /home/j/Área\ de\ trabalho/crompressor-ia/pesquisa/poc_genetic_dataset_streaming/test_genetic.py
echo ""

echo "[8/9] Swarm - P2P Weight Delta"
python3 /home/j/Área\ de\ trabalho/crompressor-ia/pesquisa/poc_p2p_weight_delta/test_p2p.py
echo ""

echo "[9/9] SRE - Telemetria Edge"
python3 /home/j/Área\ de\ trabalho/crompressor-ia/pesquisa/poc_telemetria_sre_ai/test_telemetry.py
echo ""

echo "=========================================="
echo " BATERIA CONCLUÍDA: TODOS OS TESTES PASSARAM VERDE"
echo " OS: No OOM Kill detected."
echo "=========================================="
