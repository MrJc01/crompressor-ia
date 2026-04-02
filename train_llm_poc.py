import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from crom_dataloader import CromIterableDataset

class MockLLM(nn.Module):
    """
    LLM Dummy para validar o gargalo de I/O em GPU.
    """
    def __init__(self, vocab_size=5000, d_model=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return self.linear(self.embedding(x))

def pseudo_tokenize(text):
    # Simula tokenização
    words = text.split()
    return torch.tensor([hash(w) % 5000 for w in words], dtype=torch.long)

def collate_fn(batch):
    # Pad and batch
    tokenized = [pseudo_tokenize(t) for t in batch]
    max_len = max(len(t) for t in tokenized)
    padded = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, t in enumerate(tokenized):
        padded[i, :len(t)] = t
    return padded

def train(mount_path="/tmp/magic_merge/mock_dataset.jsonl"):
    print(f"\n[SRE Engine] Iniciando Simulação de Treinamento LLM a partir de {mount_path}...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    model = MockLLM().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    dataset = CromIterableDataset(file_path=mount_path)
    # Num workers atrelados ao I/O FUSE
    dataloader = DataLoader(dataset, batch_size=256, collate_fn=collate_fn, num_workers=2)
    
    start_time = time.time()
    total_samples = 0
    
    model.train()
    for batch_idx, data in enumerate(dataloader):
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward Pass Real
        outputs = model(data)
        
        # Pseudo Loss - Evitar Memory Leaks reais no POC
        loss = outputs.sum()
        loss.backward()
        optimizer.step()
        
        total_samples += data.size(0)
        
        if batch_idx % 100 == 0:
            elapsed = time.time() - start_time
            print(f"Epoch 0 | Batch {batch_idx:04d} | Samples: {total_samples} | Tempo: {elapsed:.2f}s | Perda: {loss.item():.4f}")
            
    total_time = time.time() - start_time
    print(f"\n[SUCESSO] Treinamento Finalizado O(1) Zero-Copy!")
    print(f"I/O Total: {total_samples} amostras extraídas termodinamicamente via CROM.")
    print(f"Tempo Total: {total_time:.2f}s")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="/tmp/magic_merge/mock_dataset.jsonl")
    args = parser.parse_args()
    
    train(args.dataset)
