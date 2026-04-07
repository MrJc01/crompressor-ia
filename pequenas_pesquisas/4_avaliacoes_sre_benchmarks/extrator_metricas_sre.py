import re
import os

def parse_log(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
        
    # Extracting Tokens per second from llama-cli summary
    # Format: "llama_print_timings: prompt eval time = ..."
    # We want the 'eval time' and 'token count'
    
    tps_match = re.search(r"eval time = .*? (\d+\.\d+) tokens per second", content)
    tps = tps_match.group(1) if tps_match else "N/A"
    
    # Extract the actual generation (after the prompt)
    # Llama-cli outputs the prompt then the completion.
    
    # Simple check for N-gram repetition (4 words)
    words = content.split()
    repetition_found = False
    for i in range(len(words) - 8):
        window = words[i:i+4]
        next_window = words[i+4:i+8]
        if window == next_window:
            repetition_found = True
            break
            
    return {
        "file": os.path.basename(filepath),
        "tps": tps,
        "loop_detected": repetition_found
    }

if __name__ == "__main__":
    LOG_DIR = "/home/j/Área de trabalho/crompressor-ia/pequenas_pesquisas/2_estrategias_decodificacao/logs"
    print(f"{'FILE':<20} | {'TPS':<10} | {'LOOP?':<10}")
    print("-" * 45)
    
    if os.path.exists(LOG_DIR):
        for log_file in sorted(os.listdir(LOG_DIR)):
            if log_file.endswith(".txt"):
                metrics = parse_log(os.path.join(LOG_DIR, log_file))
                print(f"{metrics['file']:<20} | {metrics['tps']:<10} | {str(metrics['loop_detected']):<10}")
    else:
        print(f"⚠️ Diretório de logs {LOG_DIR} não encontrado. Execute o benchmark do Módulo 2 primeiro.")
