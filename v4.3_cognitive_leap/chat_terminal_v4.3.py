#!/usr/bin/env python3
import os
import sys
import time
import psutil
from llama_cpp import Llama

# ── Cores e Estética SRE (ANSI) ─────────────────────────────────────────────
CYAN = "\033[0;36m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
RED = "\033[0;31m"
WHITE = "\033[1;37m"
DIM = "\033[2m"
BOLD = "\033[1m"
NC = "\033[0m"

# ── Configurações do Chassis (VFS/Crompressor Optimized) ──────────────────────
MODEL_PATH = "/home/j/Área de trabalho/crompressor-ia/mnt_crom/CROM-IA_v4.3_Qwen3.5-2B.gguf"
# Alterado para a raiz para evitar erro de Read-only FS no ponto de montagem FUSE
BRIDGE_PATH = "/home/j/Área de trabalho/crompressor-ia/bridge_lsh_map.mmap"
N_CTX = 2048
N_THREADS = 4  # Aumentado para 4 threads (Total de CPUs lógicas) para compensar latência VFS
N_BATCH = 512  # Aumentado para 512 para processamento de prompt mais rápido

def clear_screen():
    os.system('clear' if os.name == 'posix' else 'cls')

def draw_header():
    clear_screen()
    print(f"{CYAN}╔══════════════════════════════════════════════════════════════╗{NC}")
    print(f"{CYAN}║{NC}{BOLD}       🧠 CROM-IA V4.3 — MONITOR DE CHAT TERMINAL          {NC}{CYAN}║{NC}")
    print(f"{CYAN}╠══════════════════════════════════════════════════════════════╣{NC}")
    print(f"{CYAN}║{NC}  Chassis: {WHITE}Qwen 3.5 2B (DNA-Optimized){NC}                      {CYAN}║{NC}")
    print(f"{CYAN}╚══════════════════════════════════════════════════════════════╝{NC}")
    print("")

def draw_preflight():
    print(f"{WHITE}── Status de Pré-Voo (SRE Hardening) ───────────────────────{NC}")
    if os.path.exists(MODEL_PATH):
        tamanho = os.path.getsize(MODEL_PATH) / (1024**2)
        print(f"   {GREEN}✅{NC} Modelo: {os.path.basename(MODEL_PATH)} {DIM}({tamanho:.1f} MB){NC}")
    else:
        print(f"   {RED}❌ Chassis não encontrado em: {MODEL_PATH}{NC}")
        sys.exit(1)
        
    rss_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    print(f"   {GREEN}✅{NC} Memória Inicial (RSS): {CYAN}{rss_mb:.1f} MB{NC}")
    print(f"   {GREEN}✅{NC} Camada de Atenção: {CYAN}{N_CTX} tokens{NC}")
    print(f"   {GREEN}✅{NC} Threads de Processamento: {CYAN}{N_THREADS}{NC}")
    print("")

class CromRetriever:
    """Motor de recuperação DNA via Shared Memory (MMap)."""
    def __init__(self, bridge_path):
        self.bridge_path = bridge_path
        self.dna_active = False
        if os.path.exists(bridge_path):
            self.dna_active = True

    def retrieve_context(self, query):
        """Busca átomos de DNA relevantes nos arquivos do projeto e injeta o símbolo soberano."""
        if not self.dna_active:
            return ""
        
        # Lista de arquivos para minerar contexto DNA (Prioridade SOTA)
        search_paths = [
            "/home/j/Área de trabalho/crompressor-ia/v4.3_cognitive_leap/ARCHITECTURE_PROPOSAL.md",
            "/home/j/Área de trabalho/crompressor-ia/v4.3_cognitive_leap/ROOT_DIAGNOSTICS.md"
        ]
        
        relevant_atoms = []
        keywords = [kw.lower() for kw in query.split() if len(kw) > 3]
        
        for path in search_paths:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    # Simulação de busca LSH/Semântica para esta fase de integração
                    for line in content.split('\n'):
                        if any(kw in line.lower() for kw in keywords):
                            relevant_atoms.append(line.strip())
                            if len(relevant_atoms) >= 5: break

        if relevant_atoms:
            # Formatação com o Símbolo Soberano ⌬ para o prompt
            header = f"\n{YELLOW}⌬ [SOLO SEMÂNTICO - ATOMS DETECTED]{NC}\n"
            atoms_list = "\n".join([f"  └─ {a}" for a in relevant_atoms])
            return f"{header}{atoms_list}\n"
            
        return f"\n{DIM}[DNA STATE: Estável - Sem atoms detectados]{NC}"

def main():
    draw_header()
    draw_preflight()
    
    print(f"{DIM}Carregando engine neural...{NC}")
    try:
        # Configurações de Invocação (SRE Optimized)
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=N_CTX,
            n_threads=N_THREADS,
            n_batch=N_BATCH,
            use_mlock=True,
            use_mmap=True,
            verbose=False,
            chat_format="chatml" 
        )
        print(f"{GREEN}[OK] Salto Cognitivo Inicializado.{NC}\n")
    except Exception as e:
        print(f"{RED}[ERRO FATAL] Falha no boot do modelo: {e}{NC}")
        print(f"{YELLOW}Dica: Se falhar por 'mlock', tente fechar o Chrome ou aplicativos pesados.{NC}")
        sys.exit(1)

    print(f"{WHITE}Digite {BOLD}'sair'{NC}{WHITE} para encerrar ou {BOLD}Ctrl+C{NC}{WHITE}.{NC}")
    print(f"{WHITE}────────────────────────────────────────────────────────────{NC}")

    messages = [
        {"role": "system", "content": "Você é CROM-IA v4.3, uma inteligência de compressão DNA. Responda de forma lógica, técnica e em Português."}
    ]

    retriever = CromRetriever(BRIDGE_PATH)

    while True:
        try:
            user_input = input(f"\n{CYAN}👤 VOCÊ >{NC} ").strip()
            
            if not user_input:
                continue
            
            # 🧬 DNA Cognitive Loop
            dna_context = retriever.retrieve_context(user_input)
            
            if user_input.lower() in ["sair", "exit", "quit", "0"]:
                print(f"\n{DIM}Sistemas neurais desligados. Até breve.{NC}")
                break

            # Injeção do Contexto DNA
            prompt_final = f"{dna_context}\n\nUsuário: {user_input}"
            messages.append({"role": "user", "content": prompt_final})
            
            # Limitar histórico para não estourar o contexto (mantém system + ultimos 4 turnos)
            if len(messages) > 6:
                messages = [messages[0]] + messages[-4:]

            print(f"\n{GREEN}⌬  CROM-IA v4.3 >{NC} ", end="", flush=True)

            t0 = time.time()
            tokens_count = 0
            full_response = ""
            
            stream = llm.create_chat_completion(
                messages=messages,
                max_tokens=512,
                temperature=0.3, # Baixa temperatura para precisão DNA
                repeat_penalty=1.1,
                stream=True
            )

            for chunk in stream:
                if "content" in chunk["choices"][0]["delta"]:
                    text = chunk["choices"][0]["delta"]["content"]
                    print(text, end="", flush=True)
                    full_response += text
                    tokens_count += 1

            latency = time.time() - t0
            tps = tokens_count / latency if latency > 0 else 0
            rss_final = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

            print(f"\n\n{DIM}📊 [SRE] {tokens_count} tokens | {tps:.1f} t/s | Latência: {latency:.2f}s | RSS: {rss_final:.1f}MB{NC}")
            
            messages.append({"role": "assistant", "content": full_response})

        except KeyboardInterrupt:
            print(f"\n\n{YELLOW}[!] Operação interrompida pelo usuário.{NC}")
        except Exception as e:
            print(f"\n\n{RED}[ERRO] {e}{NC}")

if __name__ == "__main__":
    main()
