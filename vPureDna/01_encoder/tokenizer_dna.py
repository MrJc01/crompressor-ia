#!/usr/bin/env python3
"""
🧬 vPureDna — DNA Compressor (⌬)

Regra de ouro: ⌬ID só aparece quando é MENOR que o texto original.
"oi" fica "oi". "inteligência artificial" vira ⌬F42.

O modelo pensa em texto natural. ⌬ é um atalho inteligente.
"""

import os
import json
import re

DNA_MARKER = "⌬"

_DEFAULT_CB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "codebooks", "codebook_1x5_dinamico_expandido.json"
)


class DNACompressor:
    """
    Comprime texto substituindo frases longas por ⌬IDs.
    Descomprime expandindo ⌬IDs de volta para texto.

    Tiers:
      ⌬W = Palavra (1 token, ≥ threshold)
      ⌬F = Frase (2+ tokens recorrentes)
      ⌬P = Parágrafo (blocos inteiros)
    """

    def __init__(self, codebook_path=None):
        if codebook_path is None:
            codebook_path = _DEFAULT_CB

        with open(codebook_path, 'r', encoding='utf-8') as f:
            cb = json.load(f)

        raw_entries = cb.get("entries", {})
        raw_reverse = cb.get("reverse_map", {})

        # Reorganizar em tiers: W (1 palavra), F (2+ palavras)
        self.w_entries = {}  # id_num → texto
        self.f_entries = {}  # id_num → texto
        self.w_reverse = {}  # texto → ⌬WXX
        self.f_reverse = {}  # texto → ⌬FXX

        w_id = 1
        f_id = 1

        # Ordenar por frequência (mais frequente = ID menor = comprime mais)
        sorted_entries = sorted(
            raw_entries.items(),
            key=lambda x: x[1].get("freq", 0),
            reverse=True
        )

        for _code, val in sorted_entries:
            text = val.get("text", "")
            n_words = len(text.split())

            if n_words == 1:
                dna_id = f"{DNA_MARKER}W{w_id}"
                # Regra de ouro: ⌬WXX só se for menor que o texto
                if len(dna_id) < len(text):
                    self.w_entries[w_id] = text
                    self.w_reverse[text] = dna_id
                    w_id += 1
            else:
                dna_id = f"{DNA_MARKER}F{f_id}"
                # Frases de 2+ palavras quase sempre comprimem
                if len(dna_id) < len(text):
                    self.f_entries[f_id] = text
                    self.f_reverse[text] = dna_id
                    f_id += 1

        # Construir reverse lookup otimizado para greedy match
        self._all_reverse = {}
        self._all_reverse.update(self.f_reverse)  # frases primeiro (maior match)
        self._all_reverse.update(self.w_reverse)

        self.total_w = len(self.w_entries)
        self.total_f = len(self.f_entries)

    @staticmethod
    def _tokenize(text):
        return re.findall(r'[\w]+|[.,!?;:\-\(\)\[\]\{\}\"\'\/\\#$%&*+=<>~`^|]', text.lower())

    def compress(self, text):
        """
        Texto → texto com ⌬IDs onde comprime.
        Palavras curtas ficam intactas.
        """
        tokens = self._tokenize(text)
        result = []
        i = 0

        while i < len(tokens):
            matched = False

            # Greedy: tenta match do maior fragmento primeiro (até 5 palavras)
            max_n = min(5, len(tokens) - i)
            for n in range(max_n, 0, -1):
                fragment = ' '.join(tokens[i:i+n])
                if fragment in self._all_reverse:
                    dna_id = self._all_reverse[fragment]
                    result.append(dna_id)
                    i += n
                    matched = True
                    break

            if not matched:
                result.append(tokens[i])  # manter texto original
                i += 1

        return ' '.join(result)

    def decompress(self, text):
        """
        Texto com ⌬IDs → texto humano.
        """
        parts = text.split()
        result = []

        for part in parts:
            if part.startswith(f"{DNA_MARKER}W"):
                try:
                    wid = int(part[2:])
                    result.append(self.w_entries.get(wid, part))
                except ValueError:
                    result.append(part)
            elif part.startswith(f"{DNA_MARKER}F"):
                try:
                    fid = int(part[2:])
                    result.append(self.f_entries.get(fid, part))
                except ValueError:
                    result.append(part)
            elif part.startswith(f"{DNA_MARKER}P"):
                result.append(part)  # TODO: paragraphs
            else:
                result.append(part)

        return ' '.join(result)

    def stats(self, text):
        """Métricas de compressão."""
        compressed = self.compress(text)
        decompressed = self.decompress(compressed)

        tokens_orig = self._tokenize(text)
        tokens_comp = compressed.split()

        n_dna = sum(1 for t in tokens_comp if t.startswith(DNA_MARKER))
        n_plain = len(tokens_comp) - n_dna

        return {
            "original": text,
            "compressed": compressed,
            "decompressed": decompressed,
            "tokens_original": len(tokens_orig),
            "tokens_compressed": len(tokens_comp),
            "dna_markers": n_dna,
            "plain_tokens": n_plain,
            "compression_ratio": len(tokens_orig) / max(len(tokens_comp), 1),
            "chars_saved": len(text) - len(compressed),
            "chars_original": len(text),
            "chars_compressed": len(compressed),
            "roundtrip_ok": decompressed.strip() == ' '.join(tokens_orig),
        }


if __name__ == "__main__":
    print("🧬 vPureDna — DNA Compressor (⌬)")
    print("=" * 60)

    comp = DNACompressor()
    print(f"  ⌬W entries: {comp.total_w} (palavras longas que comprimem)")
    print(f"  ⌬F entries: {comp.total_f} (frases que comprimem)")
    print()

    tests = [
        "a",
        "oi",
        "de que para",
        "Olá mundo",
        "inteligência artificial é o futuro da computação",
        "O Brasil é um país com rica diversidade cultural e natural",
        "A inteligência artificial pode ser usada para resolver problemas de computação",
        "def hello(): print('Hello World')",
    ]

    for text in tests:
        s = comp.stats(text)
        print(f'  ORIGINAL:    "{text}"')
        print(f'  COMPRIMIDO:  "{s["compressed"]}"')
        print(f'  EXPANDIDO:   "{s["decompressed"]}"')
        print(f'  Tokens: {s["tokens_original"]} → {s["tokens_compressed"]} | '
              f'⌬ markers: {s["dna_markers"]} | Ratio: {s["compression_ratio"]:.1f}x | '
              f'Roundtrip: {"✅" if s["roundtrip_ok"] else "❌"}')
        print()
