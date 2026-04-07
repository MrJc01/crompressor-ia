#!/usr/bin/env python3
"""
🧬 vPureDna — Treinar modelo DNA (⌬) do zero
Smoke test: char-level transformer, ~5M params, 300+ steps.
Se o loss converge → vale treinar modelo maior no Colab.
"""

import os
import sys
import json
import math
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

VPURE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# =========================================
# Mini Transformer (char-level)
# =========================================

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embd
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                             .view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        hd = C // self.n_head
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)
        k = k.view(B, T, self.n_head, hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, hd).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hd))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.GELU(),
            nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, n_layer, n_head, n_embd, block_size, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.wte.weight = self.head.weight  # weight tying
        self.apply(self._init_weights)
        n_params = sum(p.numel() for p in self.parameters())
        print(f"  🧠 MiniTransformer: {n_params/1e6:.2f}M params | vocab={vocab_size} | ctx={block_size}")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None: torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new, temperature=0.8, top_k=40):
        for _ in range(max_new):
            idx_c = idx[:, -self.block_size:]
            logits, _ = self(idx_c)
            logits = logits[:, -1, :] / temperature
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, 1)), dim=1)
        return idx


# =========================================
# Char-level tokenizer (simples)
# =========================================

class CharTokenizer:
    def __init__(self, text):
        chars = sorted(set(text))
        self.char_to_id = {c: i for i, c in enumerate(chars)}
        self.id_to_char = {i: c for c, i in self.char_to_id.items()}
        self.vocab_size = len(chars)

    def encode(self, text):
        return [self.char_to_id.get(c, 0) for c in text]

    def decode(self, ids):
        return ''.join(self.id_to_char.get(i, '?') for i in ids)


# =========================================
# Main
# =========================================

def main():
    parser = argparse.ArgumentParser(description="🧬 vPureDna Smoke Test Trainer")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--dataset", type=str, default=None)
    args = parser.parse_args()

    if args.dataset is None:
        args.dataset = os.path.join(VPURE_DIR, "02_dataset", "dataset_vpuredna.jsonl")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print(" 🧬 vPureDna — SMOKE TEST (o modelo aprende ⌬?)")
    print("=" * 60)
    print(f"  Device: {device} | Steps: {args.max_steps} | Block: {args.block_size}")
    print()

    # 1. Carregar dataset e construir corpus de texto
    print("[1] Carregando dataset...")
    with open(args.dataset, encoding='utf-8') as f:
        data = [json.loads(l) for l in f]

    # Concatenar tudo como texto contínuo (estilo NanoGPT)
    corpus_parts = []
    for d in data:
        corpus_parts.append(f"### Instrução: {d['instruction']}\n### Input: {d['input']}\n### Output: {d['output']}\n\n")
    corpus = ''.join(corpus_parts)
    print(f"  Corpus: {len(corpus)} chars | {len(data)} amostras")

    # Contar ⌬ no corpus
    n_dna = corpus.count('⌬')
    print(f"  Marcadores ⌬ no corpus: {n_dna}")

    # 2. Tokenizer char-level
    print("\n[2] Construindo tokenizer char-level...")
    tok = CharTokenizer(corpus)
    print(f"  Vocab: {tok.vocab_size} chars")

    # Verificar que ⌬ está no vocab
    if '⌬' in tok.char_to_id:
        print(f"  ⌬ = token ID {tok.char_to_id['⌬']} ✅")
    else:
        print(f"  ⌬ NÃO está no vocab! ❌")
        return

    # 3. Codificar
    all_ids = tok.encode(corpus)
    data_np = np.array(all_ids, dtype=np.int64)
    split = int(len(data_np) * 0.9)
    train_data = data_np[:split]
    val_data = data_np[split:]
    print(f"  Train: {len(train_data)} tokens | Val: {len(val_data)} tokens")

    # 4. Criar modelo
    print(f"\n[3] Criando modelo...")
    model = MiniTransformer(
        vocab_size=tok.vocab_size,
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
        block_size=args.block_size, dropout=0.1,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # 5. Training loop
    print(f"\n[4] Treinando ({args.max_steps} steps)...")
    print(f"{'─' * 60}")
    print(f"  {'Step':>6} | {'Train Loss':>11} | {'Val Loss':>9} | {'t/s':>6}")
    print(f"{'─' * 60}")

    def get_batch(data_arr):
        ix = torch.randint(len(data_arr) - args.block_size - 1, (args.batch_size,))
        x = torch.stack([torch.from_numpy(data_arr[i:i+args.block_size].copy()) for i in ix]).to(device)
        y = torch.stack([torch.from_numpy(data_arr[i+1:i+1+args.block_size].copy()) for i in ix]).to(device)
        return x, y

    @torch.no_grad()
    def eval_loss():
        model.eval()
        losses = []
        for _ in range(10):
            X, Y = get_batch(val_data)
            _, loss = model(X, Y)
            losses.append(loss.item())
        model.train()
        return sum(losses) / len(losses)

    log = []
    best_val = float('inf')
    t0 = time.time()

    for step in range(args.max_steps):
        X, Y = get_batch(train_data)
        _, loss = model(X, Y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % args.eval_interval == 0 or step == args.max_steps - 1:
            val_loss = eval_loss()
            elapsed = time.time() - t0
            tps = (step + 1) * args.batch_size * args.block_size / max(elapsed, 1)
            print(f"  {step:>6} | {loss.item():>11.4f} | {val_loss:>9.4f} | {tps:>6.0f}")
            log.append({"step": step, "train": loss.item(), "val": val_loss})
            if val_loss < best_val:
                best_val = val_loss
                ckpt = os.path.join(VPURE_DIR, "03_modelo", "checkpoints")
                os.makedirs(ckpt, exist_ok=True)
                torch.save({"model": model.state_dict(), "tok_chars": tok.char_to_id}, 
                           os.path.join(ckpt, "best.pt"))

    elapsed = time.time() - t0

    # 6. Gerar amostra
    print(f"\n{'─' * 60}")
    print(f"\n[5] Gerando amostras de teste...")

    prompts = [
        "### Instrução: Responda usando marcadores ⌬ para frases conhecidas.\n### Input: O que é inteligência artificial?\n### Output:",
        "### Instrução: Expanda os marcadores ⌬ para texto completo em Português.\n### Input: ⌬F186",
        "### Instrução: Responda em Português.\n### Input: Como funciona a compressão de dados?\n### Output:",
    ]

    model.eval()
    for p in prompts:
        ids = tok.encode(p)
        x = torch.tensor([ids], dtype=torch.long, device=device)
        out = model.generate(x, max_new=100, temperature=0.8)
        generated = tok.decode(out[0].tolist())[len(p):]
        print(f"\n  Prompt: {p[:60]}...")
        print(f"  Gerado: {generated[:120]}")

    # 7. Relatório
    convergiu = log[-1]["train"] < log[0]["train"] * 0.5
    dna_na_saida = any('⌬' in tok.decode(
        model.generate(
            torch.tensor([tok.encode("### Output:")], dtype=torch.long, device=device),
            max_new=50, temperature=0.8
        )[0].tolist()
    ) for _ in range(3))

    print(f"\n{'=' * 60}")
    print(f" 📊 RELATÓRIO vPureDna SMOKE TEST")
    print(f"{'=' * 60}")
    print(f"  Tempo:           {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Loss inicial:    {log[0]['train']:.4f}")
    print(f"  Loss final:      {log[-1]['train']:.4f}")
    print(f"  Best val loss:   {best_val:.4f}")
    print(f"  Convergiu (>50%): {'✅ SIM' if convergiu else '❌ NÃO'}")
    print(f"  Emite ⌬?:        {'✅ SIM' if dna_na_saida else '⚠️ Precisa mais treino'}")
    print(f"\n  VEREDICTO: {'🔥 VALE treinar no Colab!' if convergiu else '⚠️ Investigar dataset/modelo'}")
    print(f"{'=' * 60}")

    # Salvar log
    log_path = os.path.join(VPURE_DIR, "03_modelo", "checkpoints", "smoke_test_log.json")
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2)


if __name__ == "__main__":
    main()
