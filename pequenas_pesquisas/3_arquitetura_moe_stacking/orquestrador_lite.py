import subprocess
import os

class CROMOrchestrator:
    def __init__(self, binary_path, model_path, lora_dir):
        self.binary = binary_path
        self.model = model_path
        self.lora_dir = lora_dir
        
    def _get_lora_path(self, name):
        return os.path.join(self.lora_dir, f"{name}_lora.gguf")

    def run_query(self, query, active_loras=None):
        if active_loras is None:
            active_loras = ["Base_PTBR"]
            
        # Using a list for cmd handles spaces correctly in subprocess.run
        cmd = [self.binary, "-m", self.model, "-p", query, "-n", "64", "--temp", "0.7"]
        
        for lora in active_loras:
            lora_path = self._get_lora_path(lora)
            if os.path.exists(lora_path):
                cmd.extend(["--lora", lora_path])
            else:
                print(f"⚠️ LoRA {lora} não encontrada em {lora_path}")
                
        print(f"--- Executando com Cérebros: {active_loras} ---")
        try:
            # shell=False (default) is safer for paths with spaces
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            return result.stdout
        except subprocess.TimeoutExpired:
            return "❌ Timeout na inferência."
        except Exception as e:
            return f"❌ Erro: {str(e)}"

if __name__ == "__main__":
    BIN = "/home/j/Área de trabalho/crompressor-ia/pesquisa/poc_llama_cpp_fuse/llama.cpp/build/bin/llama-cli"
    MODEL = "/home/j/Área de trabalho/crompressor-ia/v4.2_multibrain_engine/3_inferencia_local/micro_cerebros/qwen3-0.6b.Q4_K_M.gguf"
    LORAS = "/home/j/Área de trabalho/crompressor-ia/v4.2_multibrain_engine/3_inferencia_local/micro_cerebros"
    
    orc = CROMOrchestrator(BIN, MODEL, LORAS)
    
    # Teste 1: Base PT-BR
    print("\n[Teste 1: Base PT-BR]")
    print(orc.run_query("Olá, você é uma IA?"))
    
    # Teste 2: Python + Base
    print("\n[Teste 2: Python + Base]")
    print(orc.run_query("Como fazer um loop em Python?", active_loras=["Base_PTBR", "Python_DNA"]))
