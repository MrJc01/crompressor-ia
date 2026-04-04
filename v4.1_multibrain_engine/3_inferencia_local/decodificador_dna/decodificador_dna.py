#!/usr/bin/env python3
"""
CROM-IA V4.1 — Decodificador DNA Pós-Processamento
Intercepta saída do modelo e traduz @@TOKENS de volta para palavras reais.
"""

import re
import json
import sys
import os

# Codebook reverso (Token DNA → Palavra real)
CODEBOOK_REVERSO = {
    # Python
    "@@IMP": "import", "@@DEF": "def", "@@RET": "return",
    "@@PRT": "print", "@@CLS": "class", "@@SLF": "self",
    "@@FNC": "function", "@@VAR": "variable", "@@STR": "string",
    "@@LST": "list", "@@DCT": "dict", "@@TPL": "tuple",
    "@@FOR": "for", "@@WHL": "while", "@@IFF": "if",
    "@@ELS": "else", "@@ELF": "elif", "@@TRY": "try",
    "@@EXC": "except", "@@FNL": "finally", "@@WTH": "with",
    "@@LMB": "lambda", "@@YLD": "yield", "@@ASY": "async",
    "@@AWT": "await", "@@TRU": "True", "@@FAL": "False",
    "@@NON": "None", "@@AND": "and", "@@ORR": "or",
    # Medicina
    "@@PAC": "paciente", "@@DGN": "diagnóstico", "@@TRT": "tratamento",
    "@@SNT": "sintoma", "@@DOE": "doença", "@@MED": "medicamento",
    "@@EXM": "exame", "@@CIR": "cirurgia", "@@HSP": "hospital",
    "@@MDC": "médico", "@@ENF": "enfermeiro", "@@RCT": "receita",
    "@@FBR": "febre", "@@DOR": "dor", "@@SNG": "sangue",
    "@@CRC": "coração", "@@PLM": "pulmão", "@@FGD": "fígado",
    "@@RIM": "rim", "@@CRB": "cérebro", "@@OSS": "osso",
    # Geral
    "@@PQE": "porque", "@@QND": "quando", "@@CMO": "como",
    "@@OND": "onde", "@@SMP": "sempre", "@@TBM": "também",
    "@@MTO": "muito", "@@PCO": "pouco", "@@GRD": "grande",
    "@@PQN": "pequeno", "@@EXP": "exemplo", "@@RES": "resultado",
}

# Regex para encontrar tokens @@XXX
PATTERN_DNA = re.compile(r'@@[A-Z]{2,4}')


def decodificar_dna(texto, codebook_reverso=None):
    """
    Intercepta tokens @@DNA no texto e substitui pela palavra real.
    Retorna: (texto_decodificado, contagem_tokens_traduzidos)
    """
    if codebook_reverso is None:
        codebook_reverso = CODEBOOK_REVERSO

    contagem = 0

    def substituir(match):
        nonlocal contagem
        token = match.group(0)
        if token in codebook_reverso:
            contagem += 1
            return codebook_reverso[token]
        return token  # Token desconhecido, manter original

    resultado = PATTERN_DNA.sub(substituir, texto)
    return resultado, contagem


def carregar_codebook_reverso(path_codebook):
    """Carrega codebook e inverte (palavra→token vira token→palavra)."""
    if path_codebook and os.path.exists(path_codebook):
        with open(path_codebook, "r") as f:
            original = json.load(f)
        return {v: k for k, v in original.items()}
    return CODEBOOK_REVERSO


def modo_pipe():
    """
    Modo pipeline: lê stdin linha por linha, decodifica DNA, imprime.
    Uso: ./llama-cli -m modelo.gguf ... | python3 decodificador_dna.py
    """
    for line in sys.stdin:
        decoded, count = decodificar_dna(line.rstrip())
        if count > 0:
            print(f"{decoded}  [🧬 {count} tokens DNA decodificados]")
        else:
            print(decoded)
        sys.stdout.flush()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Modo teste direto
        texto = " ".join(sys.argv[1:])
        resultado, n = decodificar_dna(texto)
        print(f"Original:    {texto}")
        print(f"Decodificado: {resultado}")
        print(f"Tokens DNA:  {n}")
    else:
        # Modo pipeline (stdin)
        modo_pipe()
