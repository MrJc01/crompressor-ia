#!/usr/bin/env python3
"""
🧬 vPureDna v5 — Server Manager (llama-server)

Gerencia o llama-server como daemon HTTP.
Substitui o hack script -qc que causava SIGSEGV.

API: POST /v1/chat/completions (OpenAI-compatible)
"""

import os
import sys
import time
import signal
import atexit
import subprocess
import urllib.request
import urllib.error
import json

DIR_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LLAMA_SERVER = os.path.join(DIR_BASE, "pesquisa", "poc_llama_cpp_fuse",
                            "llama.cpp", "build", "bin", "llama-server")

MODELS = {
    "F16": os.path.join(DIR_BASE, "models", "vpuredna_v5", "vpuredna_v5.gguf"),
    "Q4": os.path.join(DIR_BASE, "models", "vpuredna_v5", "vpuredna_v5_Q4KM.gguf"),
}

# Cores
C = {
    "CYAN":   "\033[0;36m",
    "GREEN":  "\033[0;32m",
    "YELLOW": "\033[1;33m",
    "RED":    "\033[0;31m",
    "DIM":    "\033[2m",
    "NC":     "\033[0m",
}


class LlamaServerManager:
    """Gerencia o llama-server como processo daemon."""

    def __init__(self, model_key="Q4", host="127.0.0.1", port=8081,
                 threads=2, ctx=2048, ngl=0):
        self.host = host
        self.port = port
        self.threads = threads
        self.ctx = ctx
        self.ngl = ngl
        self.process = None
        self.base_url = f"http://{host}:{port}"

        # Verificar binário
        if not os.path.exists(LLAMA_SERVER):
            raise FileNotFoundError(f"llama-server não encontrado: {LLAMA_SERVER}")

        # Verificar modelo
        model_path = MODELS.get(model_key)
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo '{model_key}' não encontrado: {model_path}")

        self.model_path = model_path
        self.model_key = model_key

        # Cleanup automático
        atexit.register(self.stop)

    def start(self, timeout=60):
        """Inicia llama-server e aguarda ficar pronto."""
        # Verificar se já tem algo na porta
        if self._is_healthy():
            print(f"  {C['GREEN']}✅{C['NC']} Server já rodando em {self.base_url}")
            return True

        model_size = os.path.getsize(self.model_path) / 1e9
        print(f"  {C['YELLOW']}⏳{C['NC']} Iniciando llama-server...")
        print(f"     Modelo: {os.path.basename(self.model_path)} ({model_size:.1f} GB)")
        print(f"     URL: {self.base_url}")

        cmd = [
            LLAMA_SERVER,
            "-m", self.model_path,
            "--host", self.host,
            "--port", str(self.port),
            "--threads", str(self.threads),
            "-c", str(self.ctx),
            "--log-disable",
            "--chat-template", "chatml",  # Força ChatML puro, SEM <think>
        ]

        if self.ngl > 0:
            cmd.extend(["-ngl", str(self.ngl)])

        # Iniciar como daemon
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid,  # Novo session group
        )

        # Aguardar health check
        t0 = time.time()
        dots = 0
        while time.time() - t0 < timeout:
            if self.process.poll() is not None:
                print(f"\n  {C['RED']}❌{C['NC']} Server morreu (exit={self.process.returncode})")
                return False

            if self._is_healthy():
                elapsed = time.time() - t0
                print(f"\n  {C['GREEN']}✅{C['NC']} Server pronto! ({elapsed:.1f}s)")
                return True

            # Progress dots
            dots += 1
            if dots % 2 == 0:
                print(".", end="", flush=True)
            time.sleep(0.5)

        print(f"\n  {C['RED']}❌{C['NC']} Timeout ({timeout}s) — server não respondeu")
        self.stop()
        return False

    def stop(self):
        """Para o server graciosamente."""
        if self.process and self.process.poll() is None:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=5)
            except Exception:
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                except Exception:
                    pass
            self.process = None

    def _is_healthy(self):
        """Verifica se o server está respondendo."""
        try:
            req = urllib.request.Request(f"{self.base_url}/health", method="GET")
            with urllib.request.urlopen(req, timeout=2) as resp:
                data = json.loads(resp.read())
                return data.get("status") == "ok"
        except Exception:
            return False

    def chat_completion(self, messages, max_tokens=256, temperature=0.3,
                        repeat_penalty=1.1, stop=None, thinking_budget=None):
        """
        POST /v1/chat/completions

        Args:
            messages: Lista de {"role": "...", "content": "..."}
            max_tokens: Máximo de tokens na resposta
            temperature: Temperatura de amostragem
            repeat_penalty: Penalidade de repetição
            stop: Lista de stop tokens
            thinking_budget: 0 = desabilita <think> do Qwen3 (mais rápido)

        Returns:
            str: Conteúdo da resposta do assistente
        """
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "repeat_penalty": repeat_penalty,
            "stream": False,
        }

        if stop:
            payload["stop"] = stop

        if thinking_budget is not None:
            payload["thinking_budget"] = thinking_budget

        data = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                result = json.loads(resp.read())

            choices = result.get("choices", [])
            if choices:
                content = choices[0].get("message", {}).get("content", "")
                return content.strip()

            return "[ERRO: Resposta vazia do server]"

        except urllib.error.URLError as e:
            return f"[ERRO: Server não respondeu — {e}]"
        except Exception as e:
            return f"[ERRO: {e}]"

    def get_stats(self):
        """Retorna métricas do server."""
        try:
            req = urllib.request.Request(f"{self.base_url}/health", method="GET")
            with urllib.request.urlopen(req, timeout=2) as resp:
                return json.loads(resp.read())
        except Exception:
            return {}


if __name__ == "__main__":
    print("🧬 vPureDna — Server Manager Test")
    print("=" * 50)

    mgr = LlamaServerManager(model_key="Q4", threads=2)

    if mgr.start(timeout=60):
        print("\n  Testando chat...")
        resp = mgr.chat_completion([
            {"role": "system", "content": "Responda em Português."},
            {"role": "user", "content": "oi"},
        ], max_tokens=32)
        print(f"  Resposta: {resp}")
        mgr.stop()
    else:
        print("  FALHA ao iniciar server")
