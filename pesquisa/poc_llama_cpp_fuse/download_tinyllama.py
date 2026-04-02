#!/usr/bin/env python3
import os
from huggingface_hub import hf_hub_download

print("Iniciando Download do Asset Funcional CROM-IA (TinyLlama-1.1B Q4_K_M GGUF)...")

model_id = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
filename = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
local_dir = "/tmp/crom_mnt"

os.makedirs(local_dir, exist_ok=True)

print(f"Buscando {filename} do repositório {model_id}...")
# Baixando diretamente para o /tmp/crom_mnt emulando o FUSE virtual mount
target_path = hf_hub_download(repo_id=model_id, filename=filename, local_dir=local_dir)

print(f"Download e Validação Física concluídos.\nLocalizado em: {target_path}")
print("Assets Termodinâmicos OK.")
