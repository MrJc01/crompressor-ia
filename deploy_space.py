#!/usr/bin/env python3
import os
from huggingface_hub import HfApi

# Configuração
SPACE_ID = "CromIA/Rosa-Chat-V3.5"
FOLDER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "huggingface_space")

print(f"🚀 Iniciando deploy Serverless no HuggingFace Spaces ({SPACE_ID})...")
api = HfApi()

try:
    print("Verificando se o Space já existe...")
    api.create_repo(
        repo_id=SPACE_ID,
        repo_type="space",
        space_sdk="gradio",
        exist_ok=True
    )
    print("✅ Space validado/criado com sucesso!")
except Exception as e:
    print(f"⚠️ Aviso na criação de status (pode ser problema de permissão ou já existir): {e}")

try:
    print("Subindo arquivos da Injeção Web...")
    api.upload_folder(
        folder_path=FOLDER_PATH,
        repo_id=SPACE_ID,
        repo_type="space"
    )
    print("✅ Upload finalizado!")
    print(f"🌍 ACESSE SEU APP EM: https://huggingface.co/spaces/{SPACE_ID}")
except Exception as e:
    print(f"❌ Erro crítico no upload: {e}")
