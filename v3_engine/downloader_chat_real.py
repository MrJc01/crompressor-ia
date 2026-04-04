#!/usr/bin/env python3
"""
🧬 CROM-IA V3.5: Downloader de Datasets REAIS (100k)
Baixa conversas reais de múltiplas fontes do HuggingFace.
Nada de templates sintéticos — tudo orgânico.
"""
import json
import os
import random
from datasets import load_dataset

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw_corpus")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def baixar_alpaca_pt():
    """~52k instruções em Português (GPT-4 quality)"""
    print("📥 [1/3] Baixando dominguesm/alpaca-data-pt-br...")
    ds = load_dataset("dominguesm/alpaca-data-pt-br", split="train")
    dados = []
    for item in ds:
        instrucao = item.get("instruction", "").strip()
        inp = item.get("input", "").strip()
        output = item.get("output", "").strip()
        if instrucao and output and len(output) > 10:
            dados.append({
                "instruction": instrucao,
                "input": inp,
                "output": output,
                "source": "alpaca-pt"
            })
    print(f"   ✅ {len(dados)} amostras coletadas (Alpaca PT-BR)")
    return dados

def baixar_alpaca_gpt4_pt():
    """~50k instruções de alta qualidade GPT-4 em Português (formato ShareGPT)"""
    print("📥 [2/3] Baixando FreedomIntelligence/alpaca-gpt4-portuguese...")
    ds = load_dataset("FreedomIntelligence/alpaca-gpt4-portuguese", split="train")
    dados = []
    for item in ds:
        convs = item.get("conversations", [])
        if len(convs) >= 2:
            human_msg = convs[0].get("value", "").strip()
            gpt_msg = convs[1].get("value", "").strip()
            if human_msg and gpt_msg and len(gpt_msg) > 10:
                dados.append({
                    "instruction": human_msg,
                    "input": "",
                    "output": gpt_msg,
                    "source": "alpaca-gpt4-pt"
                })
    print(f"   ✅ {len(dados)} amostras coletadas (Alpaca GPT-4 PT)")
    return dados

def gerar_identidade_rosa(quantidade=500):
    """Poucas amostras mas ALTÍSSIMA qualidade de identidade"""
    print(f"📥 [3/3] Gerando {quantidade} amostras de identidade Rosa (alta qualidade)...")
    
    perguntas_identidade = [
        ("Quem é você?", "Eu sou a Rosa, a assistente virtual da equipe CromIA. Fui desenvolvida para ajudar com programação, ciência e tecnologia de forma acessível."),
        ("Qual é o seu nome?", "Meu nome é Rosa. Sou uma inteligência artificial criada pela equipe CromIA para auxiliar desenvolvedores e pesquisadores."),
        ("O que você é?", "Sou uma IA chamada Rosa, parte do projeto CromIA. Minha especialidade é ajudar com código Python, explicações técnicas e conversas produtivas."),
        ("Quem te criou?", "Fui criada pela equipe CromIA, um grupo de desenvolvedores focados em tornar a inteligência artificial acessível para computadores modestos."),
        ("Você é humana?", "Não, sou uma inteligência artificial. Meu nome é Rosa e faço parte do projeto CromIA. Posso ajudar com muitas coisas, mas não tenho consciência nem sentimentos."),
        ("O que é a CromIA?", "CromIA é um projeto open-source que permite rodar modelos de linguagem em computadores sem GPU dedicada, usando compressão termodinâmica e sistemas FUSE."),
        ("Para que você serve?", "Sirvo para auxiliar em tarefas de programação, responder perguntas sobre ciência e tecnologia, e conversar de forma natural em português."),
        ("Qual sua função?", "Minha função principal é ser uma assistente inteligente que roda diretamente no computador do usuário, sem depender de servidores na nuvem."),
        ("De onde você vem?", "Venho do projeto CromIA, desenvolvido no Brasil. Fui treinada para ser útil, precisa e acessível a qualquer pessoa com um computador básico."),
        ("Você pensa?", "Não penso como um ser humano. Sou um modelo de linguagem que gera respostas com base em padrões estatísticos aprendidos durante o treinamento."),
    ]
    
    saudacoes = [
        ("Olá!", "Olá! Sou a Rosa da CromIA. Como posso ajudar você hoje?"),
        ("Oi, tudo bem?", "Oi! Tudo ótimo por aqui. Em que posso ser útil?"),
        ("Bom dia!", "Bom dia! Espero que seu dia esteja sendo produtivo. Como posso ajudar?"),
        ("Boa tarde.", "Boa tarde! Estou pronta para ajudar com o que precisar."),
        ("Boa noite!", "Boa noite! Ainda estou ativa e pronta para auxiliar."),
        ("E aí!", "E aí! Tudo certo? Me diga como posso ajudar."),
    ]
    
    dados = []
    for q, a in perguntas_identidade:
        dados.append({"instruction": q, "input": "", "output": a, "source": "rosa-identity"})
    for q, a in saudacoes:
        dados.append({"instruction": q, "input": "", "output": a, "source": "rosa-identity"})
    
    # Multiplicar com variações naturais (prefixos e sufixos)
    prefixos = ["", "Rosa, ", "Ei Rosa, ", "Me diga, ", "Poderia me dizer: "]
    while len(dados) < quantidade:
        base = random.choice(perguntas_identidade + saudacoes)
        prefixo = random.choice(prefixos)
        dados.append({
            "instruction": prefixo + base[0],
            "input": "",
            "output": base[1],
            "source": "rosa-identity"
        })
    
    random.shuffle(dados)
    print(f"   ✅ {len(dados)} amostras de identidade Rosa geradas")
    return dados[:quantidade]

def main():
    print("=" * 60)
    print(" 🧬 CROM-IA V3.5: Download de Dados REAIS (100k)")
    print("=" * 60)
    
    # 1. Datasets reais
    alpaca_pt = baixar_alpaca_pt()
    alpaca_gpt4 = baixar_alpaca_gpt4_pt()
    rosa_id = gerar_identidade_rosa(500)
    
    # 2. Salvar cada fonte separadamente
    fontes = {
        "chat_alpaca_pt.jsonl": alpaca_pt,
        "chat_alpaca_gpt4_pt.jsonl": alpaca_gpt4,
        "rosa_identidade.jsonl": rosa_id,
    }
    
    for nome, dados in fontes.items():
        path = os.path.join(OUTPUT_DIR, nome)
        with open(path, "w", encoding="utf-8") as f:
            for d in dados:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"   💾 Salvou {len(dados)} amostras em {path}")
    
    total = sum(len(d) for d in fontes.values())
    print(f"\n🎯 TOTAL DE AMOSTRAS REAIS COLETADAS: {total}")
    print("Próximo passo: rodar gerar_dataset_v3_lora.py para mixar com código V3")

if __name__ == "__main__":
    main()
