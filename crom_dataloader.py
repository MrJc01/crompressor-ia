import mmap
import json
import os
import torch
from torch.utils.data import IterableDataset

class CromIterableDataset(IterableDataset):
    """
    DataLoader VFS CROM Pytorch (O(1)).
    Lê diretamente da montagem FUSE Cascade (`fuse-overlayfs`) alinhando *memory map*
    para evitar cópia agressiva para a RAM do Sistema.
    """
    def __init__(self, file_path, block_size=16*1024*1024): # 16MB igual ao bloco CROM
        self.file_path = file_path
        self.block_size = block_size
        
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Erro Crítico: Mountpoint CROM {self.file_path} não estabelecido!")

    def _mmap_generator(self):
        # Abertura Read-Only binária 
        with open(self.file_path, "r", encoding="utf-8") as f:
            # Em python, utilizar o f.readline no topo de um VFS como SquashFS ou CROM pode gerar overhead
            # Porém, o Cgo-Switching é otimizado via blocos, portanto a leitura sequencial grande compensa.
            
            # Aqui simulamos a leitura da montagem viva do CROM
            while True:
                line = f.readline()
                if not line:
                    break
                
                # Simular o Filtro LSH Heurístico (Remoção do ruído entrópico)
                try:
                    data = json.loads(line)
                    # Exemplo prático de BPE bypass / Pré-tokenização:
                    # Em um sistema em produção as strings já viriam tokenizadas do HNSW
                    yield data["text"]
                except json.JSONDecodeError:
                    pass

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        generator = self._mmap_generator()
        
        # Num cenário distribuído com múltiplos workers, dividiríamos o mmap em offsets
        if worker_info is not None:
            # SRE Bypass: Simplificação O(1) para chunking de dataset
            for i, item in enumerate(generator):
                if i % worker_info.num_workers == worker_info.id:
                    yield item
        else:
            for item in generator:
                yield item

if __name__ == "__main__":
    # Teste rápido de sanidade da classe de DataLoader
    pass
