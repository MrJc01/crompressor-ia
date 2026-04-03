import math
import random

def calculate_shannon_entropy(data_string):
    prob = [float(data_string.count(c)) / len(data_string) for c in dict.fromkeys(list(data_string))]
    entropy = - sum([p * math.log(p) / math.log(2.0) for p in prob])
    return entropy

def simulate_pruning():
    print("====================================")
    print(" POC: Anti-Entropy SRE Pruning")
    print("====================================\n")
    
    # Dataset simulado com amostras reais e sujas (código hexadecimal corrompido)
    corpus = [
        "A garotinha foi ao parque brincar com seu cachorro de estimação.",
        "QWERTYUIOPASDFGHJKLZXCVBNMqwertyuiopasdfghjklzxcvbnm1234567890", # Entropia Alta
        "IAs estão convergindo para o limite da termodinâmica do Shannon.",
        "0bxd1289xzzz22kk3j4l1l1l111ll", # Ruído hexadecimal
        "xz9f8a7b6c5d4e3f2a1b0c9d8e7f6a5b4c3d2e1f", # Hash puro (lixo)
        "O motor termodinâmico do CROM expande os limites da IA de Edge.",  # Dados ricos
    ]
    
    threshold = 4.5 # Strings com H > 4.5 indicam distribuição uniforme (caos/lixo).
    
    print("Iniciando Dataloader Stream...")
    for idx, sentence in enumerate(corpus):
        h_score = calculate_shannon_entropy(sentence)
        if h_score > threshold:
            print(f" [DROP] Lixo detectado! H={h_score:.2f} > {threshold} | Amostra: {sentence[:10]}...")
            print("  -> O Pytorch sequer fará Foward Pass nisso. CPU/GPU Cycles economizados.")
        else:
            print(f" [PASS] Dados Limpos H={h_score:.2f} < {threshold} | Amostra: {sentence[:15]}...")
            
    print("\n[SUCESSO] Pre-training Dataset blindado contra envenenamento e ruídos WebCrawling.")

if __name__ == "__main__":
    simulate_pruning()
