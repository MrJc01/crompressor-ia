#!/usr/bin/env python3
"""
CROM-IA: Script de upload do modelo para HuggingFace Hub.
Uso: python3 scripts/upload_huggingface.py
"""
import sys
import os

# Garantir que o venv é usado
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pesquisa', '.venv', 'lib', 'python3.12', 'site-packages'))

from huggingface_hub import HfApi, login

print("=" * 50)
print(" 🤗 CROM-IA: Upload para HuggingFace Hub")
print("=" * 50)

# 1. Login
print("\n[1/3] Cole seu Access Token do HuggingFace abaixo:")
print("      (Gere em: https://huggingface.co/settings/tokens)")
print("      (A senha fica invisível, apenas cole e aperte Enter)\n")
login()

# 2. Configuração do repositório
print("\n[2/3] Configuração do repositório:")
repo_id = input("  → Digite o nome completo (org/modelo): ").strip()

if not repo_id or "/" not in repo_id:
    print("[ERRO] Formato inválido. Use: MinhaOrganizacao/MeuModelo")
    sys.exit(1)

# 3. Upload
modelo_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'qwen2.5-crom-dna.gguf')

if not os.path.exists(modelo_path):
    print(f"[ERRO] Modelo não encontrado em: {modelo_path}")
    sys.exit(1)

size_mb = os.path.getsize(modelo_path) / (1024 * 1024)
print(f"\n[3/3] Iniciando upload de qwen2.5-crom-dna.gguf ({size_mb:.0f} MB)")
print(f"      Destino: https://huggingface.co/{repo_id}")
print("      (Isso pode levar alguns minutos dependendo da sua internet...)\n")

api = HfApi()

# Criar o repo se não existir
try:
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
except Exception as e:
    print(f"[AVISO] Repo já existe ou erro: {e}")

# Upload do arquivo
api.upload_file(
    path_or_fileobj=modelo_path,
    path_in_repo="qwen2.5-crom-dna.gguf",
    repo_id=repo_id,
    repo_type="model",
)

url = f"https://huggingface.co/{repo_id}"
print(f"\n{'=' * 50}")
print(f" ✅ SUCESSO! Modelo publicado em:")
print(f"    {url}")
print(f"{'=' * 50}")
print(f"\n  Agora copie esse link e cole no chat do Antigravity")
print(f"  para eu atualizar o README.md do seu GitHub!\n")
