#!/usr/bin/env bash
# ==============================================================================
# CROM-IA V4.0: Uploader CLI Hugging Face
# Sobe as Matrizes Múltiplas para que a Nuvem puxe em tempo-real.
# ==============================================================================

echo "🚀 [CROM-IA HF-UPLOADER] Iniciando conexão Terminal -> Nuvem"

# Verifica ambiente
if ! command -v huggingface-cli &> /dev/null
then
    echo "❌ O huggingface-cli não foi encontrado! Rode: pip install huggingface_hub"
    exit
fi

# Pede um nome de repositorio para os dados, ex: SeuUser/MultiBrain_V4
echo "Digite o nome exato do seu Repo do Facehub (Ex: MrJc01/Crom_MultiBrain_V4_Dados): "
read REPO_NOME

# Usa a API do huggingface_hub pelo python para criar o dataset se não existir
python3 -c "
from huggingface_hub import create_repo
try:
    create_repo(repo_id='$REPO_NOME', repo_type='dataset', exist_ok=True)
    print('✅ Bucket Registrado/Confirmado na nuvem.')
except Exception as e:
    print('⚠️ Houve um aviso ao criar repo:', e)
"

echo "📦 Transferindo Cérebros..."
# Sobe a pasta via protocolo Git LFS passivo (muito mais rapido)
huggingface-cli upload "$REPO_NOME" arquivos_para_o_colab/ . --repo-type dataset

echo "🎉 DONE! Todos os dados estão no Hub. O seu Colab pode puxá-los via CLI agora."
