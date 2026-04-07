#!/usr/bin/env python3
"""
🧬 vPureDna v5 — Chat Terminal com Compressão DNA Real (⌬)

Pipeline completo:
  1. User digita texto normal
  2. DNACompressor comprime → ⌬ tokens
  3. Modelo GGUF processa (entende ⌬ nativamente)
  4. Resposta com ⌬ → DNACompressor expande → texto humano

Uso:
  python3 chat_vpuredna_v5.py
  python3 chat_vpuredna_v5.py --model Q4  (usa Q4_K_M, mais leve)
  python3 chat_vpuredna_v5.py --raw       (sem compressão DNA)
"""

import os
import sys
import subprocess
import argparse
import json
import time
import signal

# Adicionar path do encoder
DIR_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(DIR_BASE, "vPureDna", "01_encoder"))

from tokenizer_dna import DNACompressor

# ============================================================
# CONFIGURAÇÃO
# ============================================================
LLAMA_CLI = os.path.join(DIR_BASE, "pesquisa", "poc_llama_cpp_fuse",
                         "llama.cpp", "build", "bin", "llama-cli")

MODELS = {
    "F16": os.path.join(DIR_BASE, "models", "vpuredna_v5", "vpuredna_v5.gguf"),
    "Q4": os.path.join(DIR_BASE, "models", "vpuredna_v5", "vpuredna_v5_Q4KM.gguf"),
}

# Cores ANSI
C = {
    "CYAN":    "\033[0;36m",
    "GREEN":   "\033[0;32m",
    "YELLOW":  "\033[1;33m",
    "RED":     "\033[0;31m",
    "MAGENTA": "\033[0;35m",
    "BOLD":    "\033[1m",
    "DIM":     "\033[2m",
    "NC":      "\033[0m",
}


def print_banner():
    print(f"""
{C['CYAN']}╔══════════════════════════════════════════════════════════════╗
║{C['NC']}  🧬 {C['BOLD']}vPureDna v5{C['NC']} — Chat com Compressão DNA Real (⌬)         {C['CYAN']}║
╠══════════════════════════════════════════════════════════════╣
║{C['NC']}  Pipeline: Texto → ⌬Compress → LLM → ⌬Expand → Texto       {C['CYAN']}║
╚══════════════════════════════════════════════════════════════╝{C['NC']}
""")


def print_stats(comp, text_in, text_compressed, text_out, text_expanded, gen_time):
    """Exibe métricas de compressão em tempo real."""
    stats_in = comp.stats(text_in) if text_in else None
    stats_out = comp.stats(text_out) if text_out else None

    dna_in = sum(1 for t in text_compressed.split() if t.startswith("⌬")) if text_compressed else 0
    dna_out = sum(1 for t in text_out.split() if t.startswith("⌬")) if text_out else 0

    chars_saved_in = len(text_in) - len(text_compressed) if text_in and text_compressed else 0
    chars_saved_out = len(text_out) - len(text_expanded) if text_out and text_expanded else 0

    ratio_in = len(text_in) / max(len(text_compressed), 1) if text_in and text_compressed else 1.0
    ratio_out = len(text_expanded) / max(len(text_out), 1) if text_expanded and text_out else 1.0

    print(f"\n{C['DIM']}┌─────────────────────────────────────────────────┐")
    print(f"│  ⌬ DNA Compression Stats                        │")
    print(f"├─────────────────────────────────────────────────┤")
    print(f"│  Input:  {dna_in:>3} ⌬ tokens | ratio {ratio_in:.1f}x | {chars_saved_in:>+4} chars  │")
    print(f"│  Output: {dna_out:>3} ⌬ tokens | expanded {ratio_out:.1f}x            │")
    print(f"│  Time:   {gen_time:.1f}s                                   │")
    print(f"└─────────────────────────────────────────────────┘{C['NC']}")


class VPureDnaChat:
    """Chat interativo com compressão DNA em tempo real."""

    def __init__(self, model_key="Q4", use_dna=True, threads=2, ctx=2048,
                 temp=0.3, repeat_penalty=1.1, ngl=0):
        self.use_dna = use_dna
        self.threads = threads
        self.ctx = ctx
        self.temp = temp
        self.repeat_penalty = repeat_penalty
        self.ngl = ngl

        # Carregar compressor DNA
        print(f"  {C['YELLOW']}[1/3]{C['NC']} Carregando DNA Compressor (⌬)...")
        self.comp = DNACompressor()
        print(f"        ⌬W: {self.comp.total_w} | ⌬F: {self.comp.total_f}")

        # Verificar modelo
        model_path = MODELS.get(model_key)
        if not model_path or not os.path.exists(model_path):
            print(f"  {C['RED']}[ERRO]{C['NC']} Modelo não encontrado: {model_path}")
            print(f"  Modelos disponíveis: {list(MODELS.keys())}")
            sys.exit(1)

        self.model_path = model_path
        model_size = os.path.getsize(model_path) / 1e9
        print(f"  {C['YELLOW']}[2/3]{C['NC']} Modelo: {os.path.basename(model_path)} ({model_size:.1f} GB)")

        # Verificar llama-cli
        if not os.path.exists(LLAMA_CLI):
            print(f"  {C['RED']}[ERRO]{C['NC']} llama-cli não encontrado: {LLAMA_CLI}")
            sys.exit(1)
        print(f"  {C['YELLOW']}[3/3]{C['NC']} Engine: llama.cpp ✅")

        self.history = []
        self.session_stats = {
            "total_chars_saved": 0,
            "total_dna_tokens": 0,
            "total_queries": 0,
        }

    def _build_prompt(self, user_input):
        """Constrói prompt ChatML com DNA."""
        system_msg = (
            "Você é CROM-IA, um assistente inteligente em Português. "
            "Você entende e usa marcadores ⌬ (DNA tokens) para compressão semântica. "
            "⌬W = palavra comprimida, ⌬F = frase comprimida. "
            "Responda de forma clara e completa."
        )

        prompt = f"<|im_start|>system\n{system_msg}<|im_end|>\n"

        # Histórico (últimas 3 trocas)
        for h in self.history[-3:]:
            prompt += f"<|im_start|>user\n{h['user']}<|im_end|>\n"
            prompt += f"<|im_start|>assistant\n{h['assistant']}<|im_end|>\n"

        prompt += f"<|im_start|>user\n{user_input}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"

        return prompt

    def _call_llama(self, prompt, max_tokens=256):
        """Chama llama-cli e retorna resposta (one-shot, sem interativo)."""
        cmd = [
            LLAMA_CLI,
            "-m", self.model_path,
            "--threads", str(self.threads),
            "-c", str(self.ctx),
            "-n", str(max_tokens),
            "--temp", str(self.temp),
            "--repeat-penalty", str(self.repeat_penalty),
            "-p", prompt,
            "--no-display-prompt",
            "--log-disable",
            "--no-cnv",
        ]

        if self.ngl > 0:
            cmd.extend(["-ngl", str(self.ngl)])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )
            output = result.stdout.strip()

            # Remover bloco <think>...</think> se presente
            import re
            output = re.sub(r'<think>.*?</think>', '', output, flags=re.DOTALL).strip()

            # Limpar artefatos de geração
            if "<|im_end|>" in output:
                output = output.split("<|im_end|>")[0].strip()
            if "<|im_start|>" in output:
                output = output.split("<|im_start|>")[0].strip()

            return output

        except subprocess.TimeoutExpired:
            return "[ERRO: Timeout na geração]"
        except Exception as e:
            return f"[ERRO: {e}]"

    def chat(self, user_input):
        """Processa uma mensagem com pipeline DNA completo."""
        # 1. Comprimir input com DNA
        if self.use_dna:
            compressed_input = self.comp.compress(user_input)
        else:
            compressed_input = user_input

        # 2. Construir prompt
        prompt = self._build_prompt(compressed_input)

        # 3. Gerar resposta
        t0 = time.time()
        raw_response = self._call_llama(prompt)
        gen_time = time.time() - t0

        # 4. Expandir DNA na resposta
        if self.use_dna:
            expanded_response = self.comp.decompress(raw_response)
        else:
            expanded_response = raw_response

        # 5. Atualizar histórico
        self.history.append({
            "user": compressed_input,
            "assistant": raw_response,
        })

        # 6. Stats
        self.session_stats["total_queries"] += 1
        chars_saved = len(user_input) - len(compressed_input)
        self.session_stats["total_chars_saved"] += max(0, chars_saved)
        dna_count = sum(1 for t in compressed_input.split() if t.startswith("⌬"))
        self.session_stats["total_dna_tokens"] += dna_count

        return {
            "input_original": user_input,
            "input_compressed": compressed_input,
            "output_raw": raw_response,
            "output_expanded": expanded_response,
            "gen_time": gen_time,
            "dna_tokens_in": dna_count,
            "chars_saved": chars_saved,
        }

    def run_interactive(self):
        """Loop interativo de chat."""
        print_banner()

        dna_status = f"{C['GREEN']}ON{C['NC']}" if self.use_dna else f"{C['RED']}OFF{C['NC']}"
        print(f"  DNA Compression: {dna_status}")
        print(f"  Model: {os.path.basename(self.model_path)}")
        print(f"  Threads: {self.threads} | Ctx: {self.ctx} | Temp: {self.temp}")
        print()
        print(f"  Comandos: {C['DIM']}/quit /stats /dna /raw /clear /bench{C['NC']}")
        print(f"  {'─' * 55}")

        while True:
            try:
                user_input = input(f"\n{C['GREEN']}  ⌬ Você ▸{C['NC']} ").strip()
            except (EOFError, KeyboardInterrupt):
                print(f"\n\n{C['CYAN']}  👋 Sessão encerrada.{C['NC']}")
                self._print_session_stats()
                break

            if not user_input:
                continue

            # Comandos especiais
            if user_input.lower() in ("/quit", "/exit", "/q"):
                print(f"\n{C['CYAN']}  👋 Sessão encerrada.{C['NC']}")
                self._print_session_stats()
                break

            elif user_input.lower() == "/stats":
                self._print_session_stats()
                continue

            elif user_input.lower() == "/dna":
                self.use_dna = True
                print(f"  {C['GREEN']}✅ DNA Compression: ON{C['NC']}")
                continue

            elif user_input.lower() == "/raw":
                self.use_dna = False
                print(f"  {C['YELLOW']}⚠️ DNA Compression: OFF (modo raw){C['NC']}")
                continue

            elif user_input.lower() == "/clear":
                self.history = []
                print(f"  {C['GREEN']}✅ Histórico limpo.{C['NC']}")
                continue

            elif user_input.lower() == "/bench":
                self._run_benchmark()
                continue

            # Chat normal
            if self.use_dna and user_input != user_input:
                pass  # placeholder

            # Mostrar compressão em tempo real
            if self.use_dna:
                compressed = self.comp.compress(user_input)
                if compressed != user_input.lower():
                    dna_count = sum(1 for t in compressed.split() if t.startswith("⌬"))
                    if dna_count > 0:
                        print(f"  {C['DIM']}⌬ compressed: {compressed[:80]}{'...' if len(compressed) > 80 else ''}{C['NC']}")

            print(f"  {C['DIM']}⏳ Gerando...{C['NC']}", end="", flush=True)

            result = self.chat(user_input)

            # Limpar "Gerando..."
            print(f"\r  {' ' * 20}\r", end="")

            # Mostrar resposta
            display = result["output_expanded"] if self.use_dna else result["output_raw"]
            print(f"  {C['MAGENTA']}🧬 CROM ▸{C['NC']} {display}")

            # Mostrar stats inline
            if self.use_dna:
                dna_out = sum(1 for t in result["output_raw"].split() if t.startswith("⌬"))
                if result["dna_tokens_in"] > 0 or dna_out > 0:
                    print(f"  {C['DIM']}[⌬in:{result['dna_tokens_in']} ⌬out:{dna_out} saved:{result['chars_saved']}ch time:{result['gen_time']:.1f}s]{C['NC']}")

    def _print_session_stats(self):
        """Exibe estatísticas da sessão."""
        s = self.session_stats
        print(f"\n{C['CYAN']}  ┌─────────── Sessão vPureDna v5 ───────────┐")
        print(f"  │  Queries:        {s['total_queries']:>6}                 │")
        print(f"  │  ⌬ tokens total: {s['total_dna_tokens']:>6}                 │")
        print(f"  │  Chars salvos:   {s['total_chars_saved']:>6}                 │")
        print(f"  └────────────────────────────────────────┘{C['NC']}")

    def _run_benchmark(self):
        """Benchmark rápido de compressão DNA."""
        print(f"\n  {C['YELLOW']}⚡ Benchmark DNA Compressor...{C['NC']}")

        tests = [
            "O que é inteligência artificial?",
            "A inteligência artificial pode ser usada para resolver problemas de computação",
            "O Brasil é um país com rica diversidade cultural e natural",
            "Explique como funciona a compressão de dados em sistemas de informação",
        ]

        total_original = 0
        total_compressed = 0

        for text in tests:
            stats = self.comp.stats(text)
            total_original += stats["chars_original"]
            total_compressed += stats["chars_compressed"]

            ratio_str = f"{stats['compression_ratio']:.1f}x"
            rt_str = "✅" if stats["roundtrip_ok"] else "❌"
            print(f"  {C['DIM']}{text[:50]:50s} → {ratio_str:>5} | ⌬:{stats['dna_markers']:>2} | RT:{rt_str}{C['NC']}")

        overall_ratio = total_original / max(total_compressed, 1)
        print(f"\n  {C['GREEN']}Overall: {overall_ratio:.2f}x compressão | "
              f"{total_original - total_compressed} chars salvos{C['NC']}")


def main():
    parser = argparse.ArgumentParser(description="🧬 vPureDna v5 — Chat DNA")
    parser.add_argument("--model", choices=["Q4", "F16"], default="Q4",
                        help="Modelo GGUF (Q4=leve 1.1GB, F16=full 3.3GB)")
    parser.add_argument("--raw", action="store_true",
                        help="Desabilitar compressão DNA")
    parser.add_argument("--threads", type=int, default=2,
                        help="Threads CPU (default: 2)")
    parser.add_argument("--ctx", type=int, default=2048,
                        help="Context window (default: 2048)")
    parser.add_argument("--temp", type=float, default=0.3,
                        help="Temperature (default: 0.3)")
    parser.add_argument("--ngl", type=int, default=0,
                        help="GPU layers (0=CPU only)")
    args = parser.parse_args()

    chat = VPureDnaChat(
        model_key=args.model,
        use_dna=not args.raw,
        threads=args.threads,
        ctx=args.ctx,
        temp=args.temp,
        ngl=args.ngl,
    )

    chat.run_interactive()


if __name__ == "__main__":
    main()
