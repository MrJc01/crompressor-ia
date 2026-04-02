import mmap
import json
import os
import torch
import random
import math
from collections import Counter
from torch.utils.data import IterableDataset

def calc_shannon_entropy(text):
    if not text:
        return 0.0
    b = text.encode('utf-8')
    freqs = Counter(b)
    size = len(b)
    return -sum((c / size) * math.log2(c / size) for c in freqs.values())

class CromIterableDataset(IterableDataset):
    """
    DataLoader VFS CROM Pytorch (O(1)).
    Lê diretamente da montagem FUSE Cascade `fuse-overlayfs`.
    
    NOVA VERSÃO CROM-IA: 
    O Dataloader não se preocupa mais em resgatar as "palavras brutas".
    Nós injetamos na Rede Neural a infraestrutura Cosenoidal (Codebook IDs Puros).
    """
    def __init__(self, file_path, block_size=16*1024*1024):
        self.file_path = file_path
        self.block_size = block_size
        
        # Emulando a "Tabela Dinâmica de Fractais" do .cromdb
        self.crom_internal_vocab_size = 4096 
        
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Erro Crítico: Mountpoint CROM {self.file_path} não estabelecido!")

    def _mmap_generator(self):
        with open(self.file_path, "r", encoding="utf-8") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                
                try:
                    data = json.loads(line)
                    raw_text = data["text"]
                    
                    # [Simulador CROM-IA HNSW FUSE]
                    # Em um mount CROM real via C/C++ Binding direto (CGO bypass),
                    # nós nem decodificaríamos JSON string e sim leríamos .cromdb_id puro do block offset
                    
                    # Extraindo IDs Semântico-Fractais via string hash (simulando CROM O(1) FastCDC)
                    words = raw_text.split()
                    
                    entropy = calc_shannon_entropy(raw_text)
                    if entropy > 7.5:
                        # Simulando o salto da string caótica - Anti-Entropy Pruning
                        print(f"\\r[SRE-DLT] Block Pruned! HNSW Evidenciou Entropia Caótica (H={entropy:.2f} > 7.5)", end="", flush=True)
                        continue
                    
                    # Em vez de SentencePiece/BPE LLM clássico, o CodebookID já sai pronto
                    codebook_id_sequence = [hash(w) % self.crom_internal_vocab_size for w in words]
                    
                    yield codebook_id_sequence
                except json.JSONDecodeError:
                    pass

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        generator = self._mmap_generator()
        
        if worker_info is not None:
            for i, item in enumerate(generator):
                if i % worker_info.num_workers == worker_info.id:
                    yield item
        else:
            for item in generator:
                yield item

if __name__ == "__main__":
    pass
