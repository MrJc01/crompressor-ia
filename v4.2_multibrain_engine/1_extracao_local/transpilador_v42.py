#!/usr/bin/env python3
"""
CROM-IA V4.2 — Transpilador DNA (Taxa 25%, corrigido)
=====================================================
Mudança vs V4.1: TAXA_MUTACAO = 0.25 (era 0.75)
Formato: Chat Template ChatML com system prompt DNA explícito
"""

import json
import random
import re
import os
import sys

# ══════════════════════════════════════════════════════════════
# CONFIGURAÇÃO V4.2 — DNA CONSERVADOR
# ══════════════════════════════════════════════════════════════
TAXA_MUTACAO = 0.25  # 25% mutante (V4.1 era 75% — causou catastrophic forgetting!)

# Codebook DNA padrão (será carregado externamente em produção)
CODEBOOK_PADRAO = {
    # Python
    "import": "@@IMP", "def": "@@DEF", "return": "@@RET",
    "print": "@@PRT", "class": "@@CLS", "self": "@@SLF",
    "function": "@@FNC", "variable": "@@VAR", "string": "@@STR",
    "list": "@@LST", "dict": "@@DCT", "tuple": "@@TPL",
    "for": "@@FOR", "while": "@@WHL", "if": "@@IFF",
    "else": "@@ELS", "elif": "@@ELF", "try": "@@TRY",
    "except": "@@EXC", "finally": "@@FNL", "with": "@@WTH",
    "lambda": "@@LMB", "yield": "@@YLD", "async": "@@ASY",
    "await": "@@AWT", "True": "@@TRU", "False": "@@FAL",
    "None": "@@NON", "and": "@@AND", "or": "@@ORR",
    # Medicina
    "paciente": "@@PAC", "diagnóstico": "@@DGN", "tratamento": "@@TRT",
    "sintoma": "@@SNT", "doença": "@@DOE", "medicamento": "@@MED",
    "exame": "@@EXM", "cirurgia": "@@CIR", "hospital": "@@HSP",
    "médico": "@@MDC", "enfermeiro": "@@ENF", "receita": "@@RCT",
    "febre": "@@FBR", "dor": "@@DOR", "sangue": "@@SNG",
    "coração": "@@CRC", "pulmão": "@@PLM", "fígado": "@@FGD",
    "rim": "@@RIM", "cérebro": "@@CRB", "osso": "@@OSS",
    # Geral PT-BR
    "porque": "@@PQE", "quando": "@@QND", "como": "@@CMO",
    "onde": "@@OND", "sempre": "@@SMP", "também": "@@TBM",
    "muito": "@@MTO", "pouco": "@@PCO", "grande": "@@GRD",
    "pequeno": "@@PQN", "exemplo": "@@EXP", "resultado": "@@RES",
}


def carregar_codebook(path_codebook):
    """Carrega codebook DNA de arquivo JSON externo."""
    if path_codebook and os.path.exists(path_codebook):
        with open(path_codebook, "r") as f:
            data = json.load(f)
        if "codebook" in data and isinstance(data["codebook"], dict):
            return data["codebook"]
        return data
    return CODEBOOK_PADRAO


def aplicar_mutacao_dna(texto, codebook):
    """Substitui palavras-chave por tokens DNA comprimidos."""
    resultado = texto
    for palavra, token_dna in codebook.items():
        pattern = re.compile(re.escape(palavra), re.IGNORECASE)
        resultado = pattern.sub(token_dna, resultado)
    return resultado


def formatar_chat_template(instruction, output, usar_dna=False, codebook=None):
    """V4.2: ChatML com system prompt DNA. DNA a 25% (conservador)."""
    if usar_dna and codebook:
        output_final = aplicar_mutacao_dna(output, codebook)
        system_msg = "Você é CROM-IA. Use tokens @@DNA quando apropriado para comprimir respostas."
    else:
        output_final = output
        system_msg = "Você é CROM-IA, um assistente inteligente brasileiro."

    texto_completo = (
        f"<|im_start|>system\n{system_msg}<|im_end|>\n"
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n{output_final}<|im_end|>"
    )

    return {"text": texto_completo}


def transpilar_dataset_v42(path_dataset_original, path_saida, codebook, taxa_mutacao=0.25):
    """Transpilação V4.2: Taxa 25% (conservadora), Chat Template."""
    total = 0
    mutados = 0

    with open(path_dataset_original, "r") as fin, open(path_saida, "w") as fout:
        for line in fin:
            try:
                entry = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            # Detectar formato (ChatML ou instrução/output)
            if "text" in entry and "<|im_start|>" in entry.get("text", ""):
                # Já é ChatML — extrair instruction e output
                text = entry["text"]
                user_match = re.search(r'<\|im_start\|>user\n(.*?)<\|im_end\|>', text, re.DOTALL)
                asst_match = re.search(r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', text, re.DOTALL)
                instruction = user_match.group(1) if user_match else ""
                output = asst_match.group(1) if asst_match else ""
            else:
                instruction = entry.get("instruction", entry.get("input", ""))
                output = entry.get("output", entry.get("response", entry.get("text", "")))

            if not output:
                continue

            usar_dna = random.random() < taxa_mutacao
            resultado = formatar_chat_template(instruction, output, usar_dna, codebook)

            fout.write(json.dumps(resultado, ensure_ascii=False) + "\n")
            total += 1
            if usar_dna:
                mutados += 1

    taxa_real = (mutados / total * 100) if total > 0 else 0
    print(f"✅ Transpilação V4.2 concluída!")
    print(f"   Total: {total} | Mutados: {mutados} ({taxa_real:.1f}%)")
    print(f"   Taxa alvo: {taxa_mutacao*100:.0f}% | Taxa real: {taxa_real:.1f}%")
    print(f"   Saída: {path_saida}")
    return total, mutados


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("CROM-IA V4.2 — Transpilador DNA (25% conservador)")
        print(f"Uso: python3 {sys.argv[0]} <dataset.jsonl> <saida.jsonl> [codebook.json] [taxa_mutacao]")
        print(f"Exemplo: python3 {sys.argv[0]} python_15k.jsonl python_DNA25.jsonl codebook.json 0.25")
        sys.exit(1)

    path_in = sys.argv[1]
    path_out = sys.argv[2]
    path_cb = sys.argv[3] if len(sys.argv) > 3 else None
    taxa = float(sys.argv[4]) if len(sys.argv) > 4 else TAXA_MUTACAO

    codebook = carregar_codebook(path_cb)
    transpilar_dataset_v42(path_in, path_out, codebook, taxa)
