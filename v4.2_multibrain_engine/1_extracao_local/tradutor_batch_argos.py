#!/usr/bin/env python3
"""
CROM-IA V4.2 — Tradutor Batch EN→PT com Argos Translate
========================================================
Traduz datasets do inglês para português offline.
Features:
  - Checkpoint a cada 500 frases (resume se interromper)
  - Progress bar simples
  - Formata em ChatML
"""

import json
import os
import sys
import time

# ======= SEGURANÇA SRE (FAIL FAST) =======
# Foi detectado que rodar a tradução local trava a máquina via CPU lockup.
# O ArgosTranslate carrega um modelo OpenNMT pesado. Deve ser rodado NO COLAB (A100).
if not os.path.exists('/content'):
    print("\n" + "="*60)
    print("🚨 ALERTA SRE: RISCO DE TRAVAMENTO LOCAL 🚨")
    print("="*60)
    print("A tradução iterativa de 10.000 amostras através de modelos OpenNMT irá sufocar")
    print("a CPU desta máquina e causar um lockup do SO (RAM/Swap leak).")
    print("\n✅ AÇÃO CORRETIVA: Este script deve ser acoplado no Jupyter Notebook do Colab")
    print("e executado na Nuvem junto às rotinas de preparação do Treinamento.")
    print("="*60 + "\n")
    sys.exit(1)
# ==========================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "datasets_hibridos")
CHECKPOINT_INTERVAL = 500


def instalar_modelo_argos():
    """Instala o modelo en→pt do Argos Translate."""
    try:
        import argostranslate.package
        import argostranslate.translate
    except ImportError:
        print("❌ argostranslate não instalado!")
        print("   pip install argostranslate")
        sys.exit(1)

    # Verificar se já tem o modelo en→pt
    installed = argostranslate.translate.get_installed_languages()
    lang_codes = [l.code for l in installed]

    if 'en' in lang_codes and 'pt' in lang_codes:
        print("✅ Modelo en→pt já instalado")
        return

    print("📥 Baixando modelo en→pt do Argos (~50MB)...")
    argostranslate.package.update_package_index()
    available = argostranslate.package.get_available_packages()
    pkg = next((p for p in available if p.from_code == 'en' and p.to_code == 'pt'), None)
    if pkg:
        pkg.install()
        print("✅ Modelo instalado!")
    else:
        print("❌ Modelo en→pt não encontrado no Argos")
        sys.exit(1)


def traduzir_texto(texto, tradutor):
    """Traduz um texto curto de EN para PT."""
    if not texto or not isinstance(texto, str):
        return texto

    # Argos funciona melhor com textos curtos
    # Dividir em parágrafos para manter qualidade
    paragrafos = texto.split('\n')
    traduzidos = []

    for paragrafo in paragrafos:
        paragrafo = paragrafo.strip()
        if not paragrafo:
            traduzidos.append('')
            continue

        # Se parece código, não traduzir
        if any(kw in paragrafo for kw in ['def ', 'class ', 'import ', 'return ',
                                            'if __name__', '```', '   ', '{', '}']):
            traduzidos.append(paragrafo)
            continue

        try:
            trad = tradutor.translate(paragrafo)
            traduzidos.append(trad)
        except Exception:
            traduzidos.append(paragrafo)  # Fallback: manter original

    return '\n'.join(traduzidos)


def formatar_chatml(instruction, output):
    """Formata em ChatML."""
    system_msg = "Você é CROM-IA, um assistente inteligente brasileiro."
    texto = (
        f"<|im_start|>system\n{system_msg}<|im_end|>\n"
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n{output}<|im_end|>"
    )
    return {"text": texto}


def progress_bar(current, total, prefix='', length=40):
    """Progress bar simples no terminal."""
    percent = current / max(total, 1) * 100
    filled = int(length * current / max(total, 1))
    bar = '█' * filled + '░' * (length - filled)
    sys.stdout.write(f'\r   {prefix} |{bar}| {current}/{total} ({percent:.1f}%)')
    sys.stdout.flush()


def traduzir_dataset(path_input, path_output=None, max_amostras=10000):
    """Traduz dataset EN→PT com checkpoint e resumo."""
    import argostranslate.translate

    if not path_output:
        base = os.path.splitext(os.path.basename(path_input))[0]
        path_output = os.path.join(OUTPUT_DIR, f"{base}_ptbr.jsonl")

    checkpoint_file = path_output + ".checkpoint"

    # Obter tradutor
    tradutor = argostranslate.translate.get_translation_from_codes('en', 'pt')
    if not tradutor:
        print("❌ Tradutor en→pt não disponível")
        sys.exit(1)

    # Verificar checkpoint
    start_line = 0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            start_line = int(f.read().strip())
        print(f"📋 Resumindo do checkpoint: linha {start_line}")

    # Contar total
    with open(path_input, 'r') as f:
        total_linhas = sum(1 for _ in f)
    total_linhas = min(total_linhas, max_amostras)

    print(f"\n{'='*60}")
    print(f"🌐 CROM-IA V4.2 — Tradução EN→PT")
    print(f"   Input: {os.path.basename(path_input)}")
    print(f"   Total: {total_linhas} amostras")
    print(f"   Início: linha {start_line}")
    print(f"{'='*60}\n")

    total_traduzidas = start_line
    mode = 'a' if start_line > 0 else 'w'
    t_start = time.time()

    with open(path_input, 'r', encoding='utf-8') as fin, \
         open(path_output, mode, encoding='utf-8') as fout:

        for i, line in enumerate(fin):
            if i < start_line:
                continue
            if total_traduzidas >= max_amostras:
                break

            try:
                entry = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            instruction = entry.get('instruction', entry.get('input', ''))
            output = entry.get('output', entry.get('response', ''))

            if not instruction or not output:
                continue

            # Traduzir
            instruction_pt = traduzir_texto(instruction, tradutor)
            output_pt = traduzir_texto(output, tradutor)

            # Formatar e salvar
            formatted = formatar_chatml(instruction_pt, output_pt)
            fout.write(json.dumps(formatted, ensure_ascii=False) + '\n')

            total_traduzidas += 1
            progress_bar(total_traduzidas, total_linhas, 'Traduzindo')

            # Checkpoint
            if total_traduzidas % CHECKPOINT_INTERVAL == 0:
                with open(checkpoint_file, 'w') as cf:
                    cf.write(str(total_traduzidas))
                fout.flush()

    elapsed = time.time() - t_start
    rate = (total_traduzidas - start_line) / max(elapsed, 1)

    print(f"\n\n✅ Tradução concluída!")
    print(f"   Total traduzidas: {total_traduzidas}")
    print(f"   Tempo: {elapsed/60:.1f} minutos")
    print(f"   Velocidade: {rate:.1f} amostras/segundo")
    print(f"   Saída: {path_output}")

    # Limpar checkpoint
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    return path_output


def main():
    import argparse

    parser = argparse.ArgumentParser(description="CROM-IA V4.2 — Tradutor Batch EN→PT")
    parser.add_argument('--input', default=None,
                        help='Dataset EN (.jsonl). Default: openhermes_10k_en.jsonl')
    parser.add_argument('--output', default=None, help='Saída PT (.jsonl)')
    parser.add_argument('--max', type=int, default=10000, help='Max amostras (default: 10000)')
    parser.add_argument('--install-only', action='store_true',
                        help='Apenas instalar o modelo en→pt')

    args = parser.parse_args()

    # Instalar modelo
    instalar_modelo_argos()

    if args.install_only:
        print("✅ Modelo instalado. Pronto para traduzir.")
        return

    # Default input
    if not args.input:
        args.input = os.path.join(OUTPUT_DIR, "openhermes_10k_en.jsonl")

    if not os.path.exists(args.input):
        print(f"❌ Arquivo não encontrado: {args.input}")
        print(f"   Execute primeiro: python3 download_datasets_v42.py")
        sys.exit(1)

    traduzir_dataset(args.input, args.output, args.max)


if __name__ == "__main__":
    main()
