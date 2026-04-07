import os
import mmap

# Caminho da ponte de memória compartilhada
BRIDGE_PATH = "/home/j/Área de trabalho/crompressor-ia/mnt_crom/bridge_lsh_map.mmap"
# Tamanho reservado: 64MB para o Mapa LSH (ajustável conforme necessidade do motor Go)
BRIDGE_SIZE = 64 * 1024 * 1024 

def setup_bridge():
    print(f"🏗️  Inicializando Ponte MMap em: {BRIDGE_PATH}")
    
    # Garante que o diretório existe
    os.makedirs(os.path.dirname(BRIDGE_PATH), exist_ok=True)
    
    try:
        with open(BRIDGE_PATH, "wb") as f:
            # Preenche com zeros para alocação física
            f.write(b'\x00' * BRIDGE_SIZE)
        
        print(f"✅ Arquivo de 64MB criado e zerado.")
        print(f"🔗 O motor Go (crompressor) e a IA (visualizador-sre) agora podem mapear este arquivo.")
        
    except Exception as e:
        print(f"❌ Erro ao criar ponte: {e}")

if __name__ == "__main__":
    setup_bridge()
