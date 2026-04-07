import os
import sys
import pty
import select
import subprocess

# ⌬ CROM-IA — Kernel de Tradução v4.7.3 (OPERAÇÃO PURGE)
# ==============================================================================

# Dicionário Arcaico de Átomos
ATOM_MAP = {
    "⌬P_ARQUITETURA": "CROM-IA v4.3: Arquitetura Cognitiva de Salto Genético (Qwen-2B)",
    "⌬dna1": "DNA COMPRIMIDO: Segmento de Resumo Arcaico de Alta Densidade",
    "⌬A_STATUS": "STATUS ATUAL: Kernel Gaia Estabilizado. Shield Ativo. ⌬ v4.7.3 operante.",
    "⌬E_ERRO": "ERRO DE KERNEL: Incoerência detectada. Reiniciando fluxo...",
    "⌬H_USER": "USUÁRIO: DNA Humano identificado. Processando requisição...",
}

def transform_output(buffer):
    """Filtro Stealth: Silencia o prompt e ruídos binários. Apenas Átomos (⌬) saem."""
    output = ""
    active_decoding = False
    atom_mode = False
    atom_buffer = ""
    
    full_text = buffer.decode('utf-8', errors='ignore')
    
    # 🔍 Detecção de Início de Resposta (ChatML)
    if "<|im_start|>assistant" in full_text:
        # A partir daqui, começaremos a processar o output real
        # Pegamos apenas o que vem DEPOIS da tag assistant
        parts = full_text.split("<|im_start|>assistant")
        text_to_process = parts[-1]
        active_decoding = True
    else:
        # ⚠️ SRE Debug: Se o processo morreu mas não mandou a tag,
        # o buffer contém o erro crítico do llama-cli.
        return b"" # Buffer vazio por enquanto

    # ⎔ Processamento Atômico (Somente após active_decoding)
    for char in text_to_process:
        if char == "⌬":
            atom_mode = True
            atom_buffer = "⌬"
            continue
        
        if atom_mode:
            if char.isalnum() or char == "_":
                atom_buffer += char
                continue
            else:
                # Fim do Átomo - Validação e Tradução
                if len(atom_buffer) > 1:
                    traducao = ATOM_MAP.get(atom_buffer, f"Conceito: {atom_buffer}")
                    output += f"\033[1;36m{atom_buffer}\033[0m \033[3m({traducao})\033[0m "
                atom_mode = False
                atom_buffer = ""
                if char in "\n\r\t ": continue

        # Filtrar TODO o DNA Binário (ATCG) e Ruído de Texto
        # Apenas novas linhas e espaços são permitidos fora de átomos
        if char in "\n\r":
            output += char
            
    return output.encode('utf-8')

def main():
    if len(sys.argv) < 2:
        print("Uso: python3 elite_wrapper_v43.py [comando...]")
        sys.exit(1)

    cmd = sys.argv[1:]
    
    # Criar PTY para emular terminal interativo
    pid, fd = pty.fork()
    
    if pid == 0:  # FILHO: Roda o llama-cli
        os.execvp(cmd[0], cmd)
    else:         # PAI: Intercepta e filtra
        try:
            full_buffer = b""
            while True:
                # Ler do PTY
                r, w, e = select.select([fd, sys.stdin], [], [])
                
                if fd in r:
                    data = os.read(fd, 1024)
                    if not data: break
                    full_buffer += data
                    
                    # ⌬ Transmissão Stealth
                    transformed = transform_output(full_buffer)
                    if transformed:
                        # Limpa a tela e mostra apenas a resposta filtrada
                        # Para evitar duplicidade e ruído do llama-cli
                        sys.stdout.write(transformed.decode('utf-8'))
                        sys.stdout.flush()
                        full_buffer = b"<|im_start|>assistant" # Reset 'fake' do buffer para evitar repetição

                if sys.stdin in r:
                    user_input = sys.stdin.readline()
                    os.write(fd, user_input.encode('utf-8'))

        except (OSError, BrokenPipeError):
            pass
        finally:
            # Se saímos do loop e nada foi impresso, o llama-cli falhou
            # Vamos mostrar o erro para o usuário não ficar no escuro
            if len(full_buffer.strip()) > 50 and b"<|im_start|>assistant" not in full_buffer:
                print(f"\n\033[1;31m[ERRO CRÍTICO DO KERNEL]:\033[0m\n{full_buffer.decode('utf-8', errors='ignore')}")
            
            print("\n\n\033[1;30m[Sistemas neurais desconectados. Operação Purge v4.7.3 concluída.]\033[0m")
            os.close(fd)

if __name__ == "__main__":
    main()
