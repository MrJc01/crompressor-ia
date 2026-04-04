import json
import traceback
import sys
import os

try:
    from datasets import load_dataset
except ImportError:
    print("A biblioteca 'datasets' não está instalada. Execute: pip install datasets")
    sys.exit(1)

def run_download():
    output_path = "data/raw_corpus/python_extremado_corpus.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print("Iniciando ingestão com instruções (Alpaca format)...")
    
    try:
        dataset = load_dataset(
            "iamtarun/python_code_instructions_18k_alpaca", 
            streaming=True, 
            split="train"
        )
        
        limite_docs = 20000 
        documentos_extraidos = 0
        
        with open(output_path, "w", encoding="utf-8") as f:
            for row in dataset:
                code_str = row.get('output', '')
                if len(code_str) < 20: continue 
                
                # Salvamos a instrução original também para o treinamento LoRA
                dados = {
                    "id": documentos_extraidos,
                    "instruction": row.get('instruction', 'Generate Python code.'),
                    "prompt": row.get('input', ''),
                    "content": code_str
                }
                f.write(json.dumps(dados) + "\n")
                documentos_extraidos += 1
                if documentos_extraidos >= limite_docs: break
                    
        print(f"\nSucesso! Arquivo finalizado em: {output_path} ({documentos_extraidos} docs)")
        
    except Exception as e:
        print("Falha Crítica na Operação de Ingestão:")
        traceback.print_exc()

if __name__ == "__main__":
    run_download()
