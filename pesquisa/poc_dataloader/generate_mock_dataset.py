import json
import random
import os


def generate():
    output_file = "mock_dataset.jsonl"
    print(f"Bateria SRE: Gerando {output_file}...")
    
    vocab = [
        "A", "inteligência", "artificial", "é", "um", "campo", "da", "computação",
        "que", "busca", "criar", "sistemas", "capazes", "de", "realizar", "tarefas",
        "O", "compressor", "quântico", "alavanca", "redução", "de", "entropia",
        "redes", "neurais", "profundas", "são", "utilizadas", "para", "processamento",
        "de", "linguagem", "natural", "e", "visão", "computacional",
        "singularidade", "velocidade", "mmap", "fuse", "dataset", "epoch"
    ]
    
    num_lines = 150000
    
    with open(output_file, "w", encoding="utf-8") as f:
        for _ in range(num_lines):
            # Ocasionalmente gerar ruído aleatório para emular entropia de Shannon > 7.5
            if random.random() < 0.05:
                text = "".join(random.choices("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()_+", k=100))
            else:
                length = random.randint(10, 50)
                text = " ".join(random.choices(vocab, k=length))
            
            record = {"text": text, "source": "simulated_wiki", "id": random.randint(1000, 9999999)}
            f.write(json.dumps(record) + "\n")
            
    size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"\n[SUCESSO] Dataset Massivo Concluído! Tamanho: {size_mb:.2f} MB")

if __name__ == "__main__":
    generate()
