#!/usr/bin/env python3
import sys
import os
import re

# ⌬ Configurações
DNA_FILE = "/home/j/Área de trabalho/crompressor-ia/v4.3_cognitive_leap/DNA_ATOMS.md"
RE_TOKEN = re.compile(r'(⌬[WFP]_[A-Z0-9_]*|⌬dna[0-9]*)')

def load_atoms():
    atoms = {}
    if os.path.exists(DNA_FILE):
        with open(DNA_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if ":" in line and (line.strip().startswith("- ⌬") or line.strip().startswith("- @dna")):
                    parts = line.split(":", 1)
                    token = parts[0].strip("- ").strip()
                    meaning = parts[1].strip()
                    atoms[token] = meaning
    return atoms

def main():
    atoms = load_atoms()
    
    # Desativa buffering do stdout para garantir streaming em tempo real
    # sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
    
    buffer = ""
    try:
        while True:
            char = sys.stdin.read(1)
            if not char:
                break
            
            buffer += char
            
            # Se encontrar um possível início de token, continua acumulando
            if char in "⌬@":
                continue
            
            # Se o buffer parece conter um token completo ou espaço/newline, tenta traduzir
            if char.isspace() or char in ".,!?;:":
                # Verifica tokens no buffer
                matches = list(RE_TOKEN.finditer(buffer))
                if matches:
                    new_buffer = ""
                    last_idx = 0
                    for match in matches:
                        token = match.group(0)
                        new_buffer += buffer[last_idx:match.start()]
                        new_buffer += atoms.get(token, token)
                        last_idx = match.end()
                    new_buffer += buffer[last_idx:]
                    sys.stdout.write(new_buffer)
                    buffer = ""
                else:
                    sys.stdout.write(buffer)
                    buffer = ""
            
            # Limita tamanho do buffer para evitar delay infinito
            if len(buffer) > 50:
                sys.stdout.write(buffer)
                buffer = ""
                
            sys.stdout.flush()
            
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
