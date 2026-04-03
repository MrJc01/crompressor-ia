#!/usr/bin/env python3
"""CROM-IA LLM Server - Backend Real de Inferencia Edge."""
import json, time, os, sys, psutil
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
STOP_TOKEN = "<" + "|" + "user" + "|" + ">"
END_TOKEN = "</" + "s>"
llm = None

SYSTEM_PROMPT = (
    "Voce e o CROM Assistant, uma inteligencia artificial avancada que roda "
    "localmente na maquina do usuario usando compressao termodinamica FUSE mmap. "
    "Voce e extremamente inteligente, amigavel e responde sempre em portugues brasileiro. "
    "Voce tem conhecimento profundo sobre: compressao fractal, sistemas FUSE, "
    "entropia de Shannon, treinamento neural sem GPU, edge computing, "
    "e o projeto Crompressor. Responda de forma clara, detalhada e completa. "
    "Se o usuario disser algo casual como eae ou oi, responda amigavelmente. "
    "Se o usuario disser seu nome, lembre dele na conversa."
)

conversation_history = []

def load_model():
    global llm
    from llama_cpp import Llama
    print("=" * 60)
    print("  CROM-IA LLM Server - Carregando modelo real")
    print("=" * 60)
    print(f"  Modelo: {os.path.basename(MODEL_PATH)}")
    sz = os.path.getsize(MODEL_PATH) / 1024 / 1024
    print(f"  Tamanho: {sz:.0f} MB | Mmap: ON | GPU: 0")
    print("=" * 60)
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=2048, n_threads=4, n_gpu_layers=0,
        use_mmap=True, use_mlock=False, verbose=False,
    )
    proc = psutil.Process(os.getpid())
    rss = proc.memory_info().rss / 1024 / 1024
    print(f"  [OK] Carregado! RSS: {rss:.1f} MB")
    print(f"  [URL] http://localhost:5000")
    print("=" * 60)

def build_prompt(history):
    sys_tag = "<" + "|system|>"
    user_tag = "<" + "|user|>"
    asst_tag = "<" + "|assistant|>"
    parts = [f"{sys_tag}\n{SYSTEM_PROMPT}{END_TOKEN}"]
    for msg in history[-20:]:
        tag = user_tag if msg["role"] == "user" else asst_tag
        parts.append(f"{tag}\n{msg['content']}{END_TOKEN}")
    parts.append(asst_tag)
    return "\n".join(parts)

class ChatHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write(f"[CROM-API] {args[0]}\n")

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def do_GET(self):
        p = urlparse(self.path).path
        if p == "/api/status":
            proc = psutil.Process(os.getpid())
            rss = proc.memory_info().rss / 1024 / 1024
            self._json(200, {"status":"online","model":os.path.basename(MODEL_PATH),"rss_mb":round(rss,2),"mmap":True})
        else:
            self._json(404, {"error":"GET /api/status ou POST /api/chat"})

    def do_POST(self):
        p = urlparse(self.path).path
        if p == "/api/chat":
            self._chat()
        elif p == "/api/reset":
            conversation_history.clear()
            self._json(200, {"status":"cleared"})
        else:
            self._json(404, {"error":"not found"})

    def _chat(self):
        global conversation_history
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            data = json.loads(body.decode("utf-8"))
            user_msg = data.get("message", "").strip()
            if not user_msg:
                self._json(400, {"error":"message vazio"})
                return

            conversation_history.append({"role":"user","content":user_msg})
            prompt = build_prompt(conversation_history)

            t0 = time.time()
            result = llm(
                prompt,
                max_tokens=512,
                temperature=0.7,
                top_p=0.9,
                stop=[END_TOKEN, STOP_TOKEN],
                echo=False,
            )
            elapsed = (time.time() - t0) * 1000

            response_text = result["choices"][0]["text"].strip()
            conversation_history.append({"role":"assistant","content":response_text})

            proc = psutil.Process(os.getpid())
            rss = proc.memory_info().rss / 1024 / 1024

            self._json(200, {
                "response": response_text,
                "latency_ms": round(elapsed, 1),
                "tokens": result["usage"]["completion_tokens"],
                "rss_mb": round(rss, 2),
            })
            print(f"[CROM] Inferencia: {elapsed:.0f}ms | Tokens: {result['usage']['completion_tokens']} | RSS: {rss:.1f}MB")

        except Exception as e:
            print(f"[ERRO] {e}")
            self._json(500, {"error": str(e)})

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json(self, code, data):
        self.send_response(code)
        self._cors()
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"[ERRO] Modelo nao encontrado: {MODEL_PATH}")
        print("Execute: iniciar_chat_real.sh para baixar o modelo primeiro.")
        sys.exit(1)
    load_model()
    server = HTTPServer(("0.0.0.0", 5000), ChatHandler)
    print("\n[CROM-IA] Servidor pronto. Ctrl+C para encerrar.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[CROM-IA] Servidor encerrado.")
        server.server_close()
