#!/usr/bin/env python3
import sys
import re
import os

DNA_FILE = "/home/j/Área de trabalho/crompressor-ia/v4.3_cognitive_leap/DNA_ATOMS.md"

def load_atoms():
    # Estrutura: { tipo: [lista_de_linhas] }
    atoms = {"W": [], "F": [], "P": [], "dna": []}
    current_type = None
    
    if os.path.exists(DNA_FILE):
        with open(DNA_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if "## ⌬W_" in line: current_type = "W"
                elif "## ⌬F_" in line: current_type = "F"
                elif "## ⌬P_" in line: current_type = "P"
                elif "## ⌬dna" in line: current_type = "dna"
                elif line.startswith("- ⌬") or line.startswith("- @dna"):
                    if current_type:
                        atoms[current_type].append(line)
    return atoms

def retrieve_context(query, atoms):
    # Simula busca simples por palavras-chave na query
    query_terms = query.lower().split()
    
    # Prioridade: dna > W > F > P (Máximo 8 átomos no total)
    result = []
    
    # 1. Pega até 3 atalhos dna (Sempre úteis para compressão)
    result.extend(atoms["dna"][:3])
    
    # 2. Busca termos específicos nos outros tipos
    for t in ["W", "F", "P"]:
        for a in atoms[t]:
            if any(term in a.lower() for term in query_terms):
                if a not in result and len(result) < 8:
                    result.append(a)
                    
    # 3. Preenchimento (Fall-through) se ainda houver espaço
    for t in ["W", "F", "P"]:
        for a in atoms[t]:
            if a not in result and len(result) < 6:
                result.append(a)
                
    return result

if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else ""
    atoms_db = load_atoms()
    context_atoms = retrieve_context(query, atoms_db)
    
    if context_atoms:
        for atom in context_atoms:
            # Imprime apenas o átomo puro para injeção limpa
            print(atom)
