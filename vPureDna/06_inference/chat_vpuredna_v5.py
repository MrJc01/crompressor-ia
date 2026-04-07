#!/usr/bin/env python3
"""
vPureDna v5.1 — Chat Terminal com llama-cli interativo via PTY

Arquitectura:
  - llama-cli roda em modo INTERATIVO (conversation mode)
  - Python controla via pexpect (pseudo-terminal)
  - KV cache persistente entre turnos = respostas rápidas
  - DNA Compressor mede compressão (métricas)

Uso:
  python3 chat_vpuredna_v5.py
  python3 chat_vpuredna_v5.py --model Q4
  python3 chat_vpuredna_v5.py --raw
"""

import os
import sys
import argparse
import re
import time
import signal
import pexpect

DIR_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(DIR_BASE, "vPureDna", "01_encoder"))

from tokenizer_dna import DNACompressor

LLAMA_CLI = os.path.join(DIR_BASE, "pesquisa", "poc_llama_cpp_fuse",
                         "llama.cpp", "build", "bin", "llama-cli")

MODELS = {
    "F16": os.path.join(DIR_BASE, "models", "vpuredna_v5", "vpuredna_v5.gguf"),
    "Q4":  os.path.join(DIR_BASE, "models", "vpuredna_v5", "vpuredna_v5_Q4KM.gguf"),
}

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
║{C['NC']}  🧬 {C['BOLD']}vPureDna v5.1{C['NC']} — Chat DNA + llama-cli Interativo       {C['CYAN']}║
╠══════════════════════════════════════════════════════════════╣
║{C['NC']}  Engine: llama-cli (PTY) | KV cache persistente             {C['CYAN']}║
╚══════════════════════════════════════════════════════════════╝{C['NC']}
""")


class LlamaInteractive:
    """Controla llama-cli em modo interativo via pexpect."""

    # Regex para detectar o prompt '> ' do llama-cli
    PROMPT_PATTERN = r'\n> $'
    # Regex para detectar stats line
    STATS_PATTERN = r'\[ Prompt: [\d,.]+ t/s \| Generation: [\d,.]+ t/s \]'

    def __init__(self, model_path, threads=4, ctx=2048, temp=0.3,
                 repeat_penalty=1.1, ngl=0, system_prompt=None):
        self.model_path = model_path
        self.process = None
        self.ready = False

        args = [
            LLAMA_CLI,
            '-m', model_path,
            '--threads', str(threads),
            '-c', str(ctx),
            '-n', '256',
            '--temp', str(temp),
            '--repeat-penalty', str(repeat_penalty),
            '--log-disable',
        ]

        if ngl > 0:
            args.extend(['-ngl', str(ngl)])

        if system_prompt:
            args.extend(['-sys', system_prompt])

        model_size = os.path.getsize(model_path) / 1e9
        print(f"     Modelo: {os.path.basename(model_path)} ({model_size:.1f} GB)")
        print(f"     Engine: llama-cli (interativo, PTY)")

        # Iniciar processo via pexpect (args como lista para paths com espaços)
        self.process = pexpect.spawn(args[0], args=args[1:], encoding='utf-8',
                                     timeout=120, maxread=65536,
                                     env=os.environ.copy())
        self.process.setecho(False)

        # Aguardar o primeiro prompt '> '
        print(f"     Carregando modelo", end="", flush=True)
        try:
            self.process.expect(r'> $', timeout=120)
            self.ready = True
            print(f" ✅")
        except (pexpect.TIMEOUT, pexpect.EOF):
            print(f" ❌ TIMEOUT")
            self.ready = False

    def send(self, message, timeout=120):
        """Envia mensagem e retorna resposta."""
        if not self.ready or not self.process or not self.process.isalive():
            return "[ERRO: llama-cli não está rodando]"

        # Enviar mensagem
        self.process.sendline(message)

        # Aguardar resposta completa (até o próximo prompt '> ')
        try:
            self.process.expect(r'> $', timeout=timeout)
            raw = self.process.before
        except pexpect.TIMEOUT:
            # Pegar o que teve até agora
            raw = self.process.before or ""
            raw += " [TIMEOUT]"
        except pexpect.EOF:
            raw = self.process.before or "[ERRO: processo encerrou]"
            self.ready = False

        # Limpar output
        return self._clean_output(raw, message)

    def _clean_output(self, raw, original_input):
        """Limpa o output bruto do llama-cli."""
        text = raw

        # Remover a linha de input ecoada
        lines = text.split('\n')
        clean_lines = []
        skip_input = True
        for line in lines:
            stripped = line.strip()
            # Pular linha vazia no início e o echo do input
            if skip_input and (not stripped or stripped == original_input):
                if stripped == original_input:
                    skip_input = False
                continue
            skip_input = False
            # Pular stats line
            if re.search(self.STATS_PATTERN, stripped):
                continue
            clean_lines.append(line)

        text = '\n'.join(clean_lines).strip()

        # Remover ANSI codes
        text = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', text)
        text = re.sub(r'\r', '', text)

        # Remover <think>...</think> do Qwen3
        think_match = re.search(r'<think>(.*?)</think>(.*)', text, flags=re.DOTALL)
        if think_match:
            after = think_match.group(2).strip()
            inside = think_match.group(1).strip()
            text = after if after else inside
        else:
            think_open = re.search(r'<think>(.*)', text, flags=re.DOTALL)
            if think_open:
                content = think_open.group(1).strip()
                if content:
                    text = content

        # Limpar artefatos
        text = re.sub(r'<\|im_end\|>', '', text)
        text = re.sub(r'<\|im_start\|>.*', '', text)

        return text.strip()

    def stop(self):
        """Para o llama-cli."""
        if self.process and self.process.isalive():
            try:
                self.process.sendline('/exit')
                self.process.expect(pexpect.EOF, timeout=10)
            except Exception:
                pass
            try:
                self.process.close(force=True)
            except Exception:
                pass


class VPureDnaChat:
    """Chat interativo com DNA compression e llama-cli."""

    SYSTEM_PROMPT = (
        "Você é CROM-IA, um assistente inteligente criado pelo projeto Crompressor. "
        "Responda sempre em Português do Brasil, de forma clara, objetiva e completa. "
        "Seja conciso e direto. Não repita a pergunta do usuário."
    )

    def __init__(self, model_key="Q4", use_dna=True, threads=4, ctx=2048,
                 temp=0.3, repeat_penalty=1.1, ngl=0):
        self.use_dna = use_dna
        self.temp = temp

        # 1. DNA Compressor
        print(f"  {C['YELLOW']}[1/2]{C['NC']} Carregando DNA Compressor (⌬)...")
        self.comp = DNACompressor()
        print(f"        ⌬W: {self.comp.total_w} | ⌬F: {self.comp.total_f}")

        # 2. llama-cli interativo
        print(f"  {C['YELLOW']}[2/2]{C['NC']} Iniciando llama-cli...")

        model_path = MODELS.get(model_key)
        if not model_path or not os.path.exists(model_path):
            print(f"  {C['RED']}[ERRO]{C['NC']} Modelo não encontrado: {model_path}")
            sys.exit(1)

        if not os.path.exists(LLAMA_CLI):
            print(f"  {C['RED']}[ERRO]{C['NC']} llama-cli não encontrado: {LLAMA_CLI}")
            sys.exit(1)

        self.llm = LlamaInteractive(
            model_path=model_path,
            threads=threads,
            ctx=ctx,
            temp=temp,
            repeat_penalty=repeat_penalty,
            ngl=ngl,
            system_prompt=self.SYSTEM_PROMPT,
        )

        if not self.llm.ready:
            print(f"  {C['RED']}[ERRO]{C['NC']} llama-cli não iniciou!")
            sys.exit(1)

        self.session_stats = {
            "total_chars_saved": 0,
            "total_dna_tokens": 0,
            "total_queries": 0,
            "total_gen_time": 0.0,
        }

    def chat(self, user_input):
        """Processa mensagem."""
        # Medir compressão DNA
        if self.use_dna:
            compressed = self.comp.compress(user_input)
        else:
            compressed = user_input

        # Chamar llama-cli
        t0 = time.time()
        response = self.llm.send(user_input, timeout=120)
        gen_time = time.time() - t0

        # Stats
        self.session_stats["total_queries"] += 1
        self.session_stats["total_gen_time"] += gen_time
        chars_saved = len(user_input) - len(compressed)
        self.session_stats["total_chars_saved"] += max(0, chars_saved)
        dna_count = sum(1 for t in compressed.split() if t.startswith("⌬"))
        self.session_stats["total_dna_tokens"] += dna_count

        return {
            "response": response,
            "gen_time": gen_time,
            "dna_tokens_in": dna_count,
            "chars_saved": chars_saved,
            "compressed": compressed,
        }

    def run_interactive(self):
        """Loop interativo."""
        print_banner()

        dna_status = f"{C['GREEN']}ON{C['NC']}" if self.use_dna else f"{C['RED']}OFF{C['NC']}"
        print(f"  DNA Compression: {dna_status}")
        print(f"  Temp: {self.temp}")
        print()
        print(f"  Comandos: {C['DIM']}/quit /stats /dna /raw /clear /bench{C['NC']}")
        print(f"  {'─' * 55}")

        while True:
            try:
                user_input = input(f"\n{C['GREEN']}  ⌬ Você ▸{C['NC']} ").strip()
            except (EOFError, KeyboardInterrupt):
                print(f"\n\n{C['CYAN']}  👋 Sessão encerrada.{C['NC']}")
                self._print_session_stats()
                self.llm.stop()
                break

            if not user_input:
                continue

            if user_input.lower() in ("/quit", "/exit", "/q"):
                print(f"\n{C['CYAN']}  👋 Sessão encerrada.{C['NC']}")
                self._print_session_stats()
                self.llm.stop()
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
                print(f"  {C['YELLOW']}⚠️  DNA Compression: OFF{C['NC']}")
                continue

            elif user_input.lower() == "/clear":
                self.llm.send("/clear")
                print(f"  {C['GREEN']}✅ Histórico limpo.{C['NC']}")
                continue

            elif user_input.lower() == "/bench":
                self._run_benchmark()
                continue

            # Mostrar compressão
            if self.use_dna:
                compressed = self.comp.compress(user_input)
                dna_count = sum(1 for t in compressed.split() if t.startswith("⌬"))
                if dna_count > 0:
                    ratio = len(user_input) / max(len(compressed), 1)
                    print(f"  {C['DIM']}⌬ [{dna_count} tokens, {ratio:.1f}x]{C['NC']}")

            print(f"  {C['DIM']}⏳ ...{C['NC']}", end="", flush=True)

            result = self.chat(user_input)

            print(f"\r  {' ' * 15}\r", end="")

            response = result["response"]
            if not response or response.startswith("[ERRO"):
                print(f"  {C['RED']}🧬 CROM ▸{C['NC']} {response or '[Sem resposta]'}")
            else:
                print(f"  {C['MAGENTA']}🧬 CROM ▸{C['NC']} {response}")

            print(f"  {C['DIM']}[{result['gen_time']:.1f}s | ⌬:{result['dna_tokens_in']}]{C['NC']}")

    def _print_session_stats(self):
        s = self.session_stats
        avg = s["total_gen_time"] / max(s["total_queries"], 1)
        print(f"\n{C['CYAN']}  ┌─────────── Sessão vPureDna v5.1 ─────────┐")
        print(f"  │  Queries:        {s['total_queries']:>6}                 │")
        print(f"  │  ⌬ tokens total: {s['total_dna_tokens']:>6}                 │")
        print(f"  │  Chars salvos:   {s['total_chars_saved']:>6}                 │")
        print(f"  │  Tempo médio:    {avg:>5.1f}s                 │")
        print(f"  └────────────────────────────────────────┘{C['NC']}")

    def _run_benchmark(self):
        print(f"\n  {C['YELLOW']}⚡ Benchmark DNA Compressor...{C['NC']}")
        tests = [
            "O que é inteligência artificial?",
            "A inteligência artificial pode ser usada para resolver problemas de computação",
            "O Brasil é um país com rica diversidade cultural e natural",
            "Explique como funciona a compressão de dados em sistemas de informação",
        ]
        total_o = total_c = 0
        for text in tests:
            s = self.comp.stats(text)
            total_o += s["chars_original"]
            total_c += s["chars_compressed"]
            rt = "✅" if s["roundtrip_ok"] else "❌"
            print(f"  {C['DIM']}{text[:50]:50s} → {s['compression_ratio']:.1f}x | ⌬:{s['dna_markers']:>2} | RT:{rt}{C['NC']}")
        ratio = total_o / max(total_c, 1)
        print(f"\n  {C['GREEN']}Overall: {ratio:.2f}x | {total_o - total_c} chars salvos{C['NC']}")


def main():
    parser = argparse.ArgumentParser(description="vPureDna v5.1 — Chat DNA")
    parser.add_argument("--model", choices=["Q4", "F16"], default="Q4")
    parser.add_argument("--raw", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--ctx", type=int, default=2048)
    parser.add_argument("--temp", type=float, default=0.3)
    parser.add_argument("--ngl", type=int, default=0)
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
