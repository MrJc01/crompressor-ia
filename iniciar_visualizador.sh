#!/bin/bash
echo "========================================="
echo "   Iniciando CROM Visualizador SRE UI"
echo "========================================="
echo "Porta: 8080"
echo "Acesse: http://localhost:8080"
echo ""

cd "/home/j/Área de trabalho/crompressor-ia/visualizador-sre/" && python3 -m http.server 8080
