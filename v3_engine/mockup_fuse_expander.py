#!/usr/bin/env python3
"""
🧬 CROM-IA V3: Simulador FUSE RAM-Expander
Simula como o C++ irá injetar as respostas RAG Dimensionais instantaneamente
ao interceptar os Memory Pointers no fluxo do LLM.
"""

import sys
import time
import json
import os

def carregar_codebook():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(base_dir, "macro_codebook_v3.json")
    
    if not os.path.exists(input_file):
        print(f"❌ Arquivo {input_file} não encontrado!")
        sys.exit(1)
        
    with open(input_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def v3_stream_decoder(stream_texto, codebook):
    entries = codebook["entries"]
    escape_prefix = codebook["escape_prefix"]
    
    # Montamos as chaves ordenadas por tamanho decrescente para não sobrescrever menores
    ponteiros = sorted(list(entries.keys()), key=len, reverse=True)
    
    print("\n[V3 RAM EXPANDER] Interceptando Stream do LLM...\n")
    print("-" * 60)
    
    buffer_palavra = ""
    em_escape = False
    
    for char in stream_texto:
        # Simulando velocidade de LLM (ex: 20 t/s)
        time.sleep(0.01)
        
        if char.isspace() or char in ".,!?":
            # Processar buffer ao encontrar pausa
            if buffer_palavra.startswith(escape_prefix):
                # Escape: imprime a palavra crua
                sys.stdout.write(buffer_palavra[len(escape_prefix):])
            elif buffer_palavra in entries:
                # HIT: Expansão Instantânea O(1)
                expanded = entries[buffer_palavra]["text"]
                # Imprime rapidamente (Simulando a Memória RAM jogando o bloco)
                sys.stdout.write("\033[92m" + expanded + "\033[0m")
            else:
                # Fallback, não era ponteiro nem escape (em produção o LLM não falha a sintaxe)
                sys.stdout.write(buffer_palavra)
            
            sys.stdout.write(char)
            sys.stdout.flush()
            buffer_palavra = ""
        else:
            buffer_palavra += char

    # Último buffer se a string não terminou com espaço
    if buffer_palavra:
        if buffer_palavra.startswith(escape_prefix):
            sys.stdout.write(buffer_palavra[len(escape_prefix):])
        elif buffer_palavra in entries:
            expanded = entries[buffer_palavra]["text"]
            sys.stdout.write("\033[92m" + expanded + "\033[0m")
        else:
            sys.stdout.write(buffer_palavra)
    
    print("\n" + "-" * 60)

def main():
    codebook = carregar_codebook()
    
    # Texto Artificial vindo do suposto LLM CROM-IA V3
    fluxo_simulado = (
        "@@Bem-vindo! @@Para @@entender @@a @@química, @@precisamos @@brincar: "
        "AA @@E @@para @@fazer @@o @@experimento, @@voce @@precisa @@de: AT @@misturado "
        "@@com TA @@Em @@nossa @@pesquisa, @@usamos @@medidas. AC "
        "AG @@Espero @@que @@ajude!"
    )
    
    print(f"📦 Payload Bruto do LLM: {fluxo_simulado}")
    print(f"📊 Tamanho do Payload  : {len(fluxo_simulado)} chars")
    
    start_time = time.time()
    v3_stream_decoder(fluxo_simulado, codebook)
    elapsed = time.time() - start_time
    
    print(f"⏱️ Tempo de Stream Expandido: {elapsed:.2f}s")
    print(f"⚡ A expansão dos códigos DNA (em verde) ocorreu sem custo computacional para o LLM.")

if __name__ == "__main__":
    main()
