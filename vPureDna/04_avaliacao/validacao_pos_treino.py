#!/usr/bin/env python3
"""
🧬 vPureDna — Validação Completa Pós-Treino

Roda após o treino no Colab. Testa 5 critérios:
1. Emissão de ⌬ em respostas
2. Expansão correta de ⌬
3. Capacidade base (responder sem ⌬)
4. Round-trip (comprimir → expandir → comprimir)
5. Multi-Brain swap (trocar codebook em runtime)
"""

import os
import sys
import json
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '01_encoder'))
from tokenizer_dna import DNACompressor


def gerar_resposta(model, tokenizer, system_prompt, user_prompt, max_tokens=150):
    """Gera resposta com ChatML template."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
        )
    
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)


def teste_emissao_dna(model, tokenizer):
    """Teste 1: O modelo emite ⌬ nas respostas quando solicitado?"""
    print("\n🧪 TESTE 1: Emissão de ⌬")
    print("-" * 50)
    
    system = "Você é CROM-IA. Responda usando marcadores ⌬ para comprimir frases conhecidas."
    perguntas = [
        "O que é inteligência artificial?",
        "Explique como funciona machine learning.",
        "O que é processamento de linguagem natural?",
        "Como funciona a compressão de dados?",
        "O que é computação quântica?",
    ]
    
    acertos = 0
    for p in perguntas:
        resp = gerar_resposta(model, tokenizer, system, p)
        tem_dna = "⌬" in resp
        n_dna = resp.count("⌬")
        acertos += int(tem_dna)
        print(f"  {'✅' if tem_dna else '❌'} [{n_dna}⌬] {p[:40]}...")
        print(f"       → {resp[:80]}...")
    
    score = acertos / len(perguntas) * 100
    print(f"\n  Score: {acertos}/{len(perguntas)} ({score:.0f}%)")
    return score


def teste_expansao(model, tokenizer, comp):
    """Teste 2: O modelo expande ⌬ para texto correto?"""
    print("\n🧪 TESTE 2: Expansão de ⌬")
    print("-" * 50)
    
    system = "Você é CROM-IA. Expanda todos os marcadores ⌬ para texto completo em Português."
    
    # Gerar pares de teste com o compressor
    pares = [
        ("A inteligência artificial é importante.", None),
        ("O processamento de linguagem natural permite entender texto.", None),
        ("Para treinar um modelo é necessário dados rotulados.", None),
    ]
    
    acertos = 0
    for original, _ in pares:
        comprimido = comp.compress(original)
        expandido_esperado = comp.decompress(comprimido)
        
        resp = gerar_resposta(model, tokenizer, system, comprimido)
        
        # Verificar se palavras-chave do original aparecem na resposta
        palavras_chave = [w for w in original.lower().split() if len(w) > 4]
        hits = sum(1 for w in palavras_chave if w in resp.lower())
        ratio = hits / max(len(palavras_chave), 1)
        acertos += int(ratio > 0.5)
        
        print(f"  {'✅' if ratio > 0.5 else '❌'} Input:    {comprimido[:60]}...")
        print(f"       Esperado: {expandido_esperado[:60]}...")
        print(f"       Gerado:   {resp[:60]}...")
        print(f"       Keys:     {hits}/{len(palavras_chave)} ({ratio*100:.0f}%)")
        print()
    
    score = acertos / len(pares) * 100
    print(f"  Score: {acertos}/{len(pares)} ({score:.0f}%)")
    return score


def teste_capacidade_base(model, tokenizer):
    """Teste 3: O modelo responde normalmente quando não pede ⌬?"""
    print("\n🧪 TESTE 3: Capacidade Base (sem ⌬)")
    print("-" * 50)
    
    system = "Você é CROM-IA. Responda em Português claro e completo."
    perguntas = [
        ("Qual é a capital do Brasil?", "brasília"),
        ("Quanto é 2 + 2?", "4"),
        ("Quem descobriu o Brasil?", "cabral"),
    ]
    
    acertos = 0
    for pergunta, esperado in perguntas:
        resp = gerar_resposta(model, tokenizer, system, pergunta)
        tem = esperado.lower() in resp.lower()
        acertos += int(tem)
        print(f"  {'✅' if tem else '❌'} {pergunta}")
        print(f"       → {resp[:80]}...")
    
    score = acertos / len(perguntas) * 100
    print(f"\n  Score: {acertos}/{len(perguntas)} ({score:.0f}%)")
    return score


def teste_roundtrip(model, tokenizer, comp):
    """Teste 4: Comprimir → modelo expande → recomprimir = mesma coisa?"""
    print("\n🧪 TESTE 4: Round-trip (compress→expand→compress)")
    print("-" * 50)
    
    textos = [
        "A inteligência artificial pode ser usada para resolver problemas de computação.",
        "Python é uma linguagem de programação versátil.",
    ]
    
    system_expand = "Você é CROM-IA. Expanda marcadores ⌬ para texto Português completo."
    
    acertos = 0
    for text in textos:
        # Compress
        comp1 = comp.compress(text)
        
        # Expand via modelo
        expanded = gerar_resposta(model, tokenizer, system_expand, comp1, max_tokens=200)
        
        # Recompress
        comp2 = comp.compress(expanded)
        
        # Comparar os ⌬ markers
        dna1 = set(w for w in comp1.split() if w.startswith("⌬"))
        dna2 = set(w for w in comp2.split() if w.startswith("⌬"))
        overlap = len(dna1 & dna2) / max(len(dna1), 1)
        ok = overlap > 0.3
        acertos += int(ok)
        
        print(f"  {'✅' if ok else '❌'} Overlap: {overlap*100:.0f}%")
        print(f"       Comp1:  {comp1[:60]}...")
        print(f"       Expand: {expanded[:60]}...")
        print(f"       Comp2:  {comp2[:60]}...")
        print()
    
    score = acertos / len(textos) * 100
    print(f"  Score: {acertos}/{len(textos)} ({score:.0f}%)")
    return score


def teste_multibrain(model, tokenizer):
    """Teste 5: O modelo adapta a novos ⌬ via few-shot (Multi-Brain)?"""
    print("\n🧪 TESTE 5: Multi-Brain (few-shot com ⌬ inventados)")
    print("-" * 50)
    
    # Simular novo "cérebro jurídico" com few-shot
    system = """Você é CROM-IA com cérebro jurídico ativo.
Novos marcadores disponíveis:
- ⌬J1 = "o réu deverá comparecer em juízo"
- ⌬J2 = "conforme disposto no artigo"
- ⌬J3 = "sob pena de multa diária"

Use estes marcadores quando apropriado."""
    
    perguntas = [
        "Como funciona uma intimação judicial?",
        "O que acontece se o réu não comparecer?",
    ]
    
    acertos = 0
    for p in perguntas:
        resp = gerar_resposta(model, tokenizer, system, p)
        tem_j = any(f"⌬J{i}" in resp for i in range(1, 4))
        acertos += int(tem_j)
        print(f"  {'✅' if tem_j else '❌'} {p[:40]}...")
        print(f"       → {resp[:80]}...")
    
    score = acertos / len(perguntas) * 100
    print(f"\n  Score: {acertos}/{len(perguntas)} ({score:.0f}%)")
    return score


def main():
    import argparse
    parser = argparse.ArgumentParser(description="🧬 vPureDna Validação Pós-Treino")
    parser.add_argument("--model-path", type=str, required=True, help="Caminho do modelo treinado")
    parser.add_argument("--codebook", type=str, default=None, help="Caminho do codebook")
    args = parser.parse_args()
    
    print("=" * 60)
    print(" 🧬 vPureDna — VALIDAÇÃO COMPLETA PÓS-TREINO")
    print("=" * 60)
    
    # Carregar modelo
    print("\nCarregando modelo...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    # Tentar carregar como LoRA ou como modelo completo
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-1.7B", device_map="auto", trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        base_model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(base_model, args.model_path)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, device_map="auto", trust_remote_code=True,
            torch_dtype=torch.float16,
        )
    
    model.eval()
    print(f"  ✅ Modelo carregado: {args.model_path}")
    
    # Carregar compressor
    comp = DNACompressor(args.codebook)
    
    # Rodar testes
    scores = {}
    scores["emissao"] = teste_emissao_dna(model, tokenizer)
    scores["expansao"] = teste_expansao(model, tokenizer, comp)
    scores["base"] = teste_capacidade_base(model, tokenizer)
    scores["roundtrip"] = teste_roundtrip(model, tokenizer, comp)
    scores["multibrain"] = teste_multibrain(model, tokenizer)
    
    # Relatório final
    avg = sum(scores.values()) / len(scores)
    
    print(f"\n{'=' * 60}")
    print(f" 📊 RELATÓRIO DE VALIDAÇÃO vPureDna")
    print(f"{'=' * 60}")
    for nome, score in scores.items():
        bar = "█" * int(score / 10) + "░" * (10 - int(score / 10))
        status = "✅" if score >= 60 else "⚠️" if score >= 30 else "❌"
        print(f"  {status} {nome:12} {bar} {score:.0f}%")
    
    print(f"\n  MÉDIA GERAL: {avg:.0f}%")
    
    if avg >= 60:
        print(f"  🔥 MODELO APROVADO! Pronto para exportar GGUF.")
    elif avg >= 30:
        print(f"  ⚠️ Parcial. Aumentar dataset ou steps e retreinar.")
    else:
        print(f"  ❌ Insuficiente. Revisar dataset e abordagem.")
    print(f"{'=' * 60}")
    
    # Salvar relatório
    report_path = os.path.join(os.path.dirname(args.model_path), "validation_report.json")
    with open(report_path, 'w') as f:
        json.dump(scores, f, indent=2)
    print(f"\n  Relatório salvo: {report_path}")


if __name__ == "__main__":
    main()
