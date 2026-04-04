#!/usr/bin/env python3
import json
import random
import os

def gerar_persona_sintetica(quantidade=25000):
    print(f"🧬 Gerando {quantidade} amostras sintéticas de Persona Rosa...")
    
    # Templates de Identidade
    identidade_q = [
        "Quem é você?", "Qual seu nome?", "O que você é?", "Quem te criou?",
        "Você é humano?", "Rosa, me fale de você.", "Qual sua função?",
        "Quem faz parte da sua equipe?", "De onde você vem?", "Você conhece a CromIA?"
    ]
    identidade_a = [
        "Eu sou a Rosa, a assistente virtual inteligente da equipe CromIA.",
        "Meu nome é Rosa, sou uma IA desenvolvida para auxiliar a equipe CromIA.",
        "Sou um modelo de linguagem avançado, focado em tecnologia e eficiência, parte do projeto CromIA.",
        "Fui criada por engenheiros da equipe CromIA para processar dados de forma revolucionária.",
        "Eu sou a inteligência artificial Rosa. Trabalho junto com a equipe CromIA para simplificar o futuro.",
        "Não sou humana, sou uma entidade digital treinada para ajudar em tarefas complexas na CromIA."
    ]

    # Templates de Ciência / Conhecimento Geral
    ciencia_q = [
        "O que é gravidade?", "Como funciona o universo?", "O que é física quântica?",
        "Explique a relatividade.", "O que são buracos negros?", "Por que o céu é azul?",
        "Como a luz viaja?", "O que é energia termodinâmica?", "O que é entropia?",
        "O que é uma galáxia?", "Como as estrelas nascem?", "O que é o tempo?"
    ]
    ciencia_a = [
        "A gravidade é a força fundamental que atrai objetos com massa uns aos outros.",
        "O universo é a totalidade do espaço, tempo, matéria e energia existentes.",
        "Física quântica estuda as escalas atômicas e subatômicas da matéria.",
        "Relatividade é a teoria da gravitação de Einstein que descreve o tecido do espaço-tempo.",
        "Buracos negros são regiões do espaço onde a gravidade é tão forte que nem a luz escapa.",
        "O céu é azul devido à dispersão da luz solar pelas moléculas de ar na atmosfera.",
        "Entropia é uma medida da desordem ou aleatoriedade em um sistema físico.",
        "As estrelas nascem da condensação de nuvens gigantes de gás e poeira no espaço."
    ]

    # Templates de Tecnologia / Programação (Sem citar V3)
    tech_q = [
        "O que é Python?", "Como funciona um LLM?", "O que é back-end?",
        "O que é Git?", "Por que usar Linux?", "O que é uma API?",
        "Explique o que é código aberto.", "O que é uma rede neural?", "O que é IA?"
    ]
    tech_a = [
        "Python é uma linguagem de programação de alto nível, versátil e muito poderosa.",
        "LLMs são grandes modelos de linguagem treinados em bilhões de palavras para prever texto.",
        "Git é um sistema de controle de versão distribuído para rastrear mudanças no código.",
        "APIs são conjuntos de definições e protocolos para integrar softwares diferentes.",
        "Redes neurais são modelos computacionais inspirados no funcionamento do cérebro humano.",
        "Linux é um kernel de sistema operacional robusto e de código aberto, muito usado em servidores."
    ]

    prefixos_pergunta = [
        "", "Ei Rosa, ", "Rosa, ", "Poderia me dizer ", "Sabe o que é ", 
        "Me explique ", "Gostaria de saber: ", "Diga-me, "
    ]
    
    sufixos_resposta = [
        "", " Espero que isso tenha ajudado!", " De nada!", " Algo mais?", 
        " Fico feliz em explicar isso.", " A equipe CromIA está sempre à disposição."
    ]

    dados = []
    
    # Combinatória para atingir a meta massiva
    while len(dados) < quantidade:
        categoria = random.choice(["id", "ciencia", "tech", "chat"])
        
        if categoria == "id":
            q = random.choice(prefixos_pergunta) + random.choice(identidade_q)
            a = random.choice(identidade_a) + random.choice(sufixos_resposta)
        elif categoria == "ciencia":
            q = random.choice(prefixos_pergunta) + random.choice(ciencia_q)
            a = random.choice(ciencia_a) + random.choice(sufixos_resposta)
        elif categoria == "tech":
            q = random.choice(prefixos_pergunta) + random.choice(tech_q)
            a = random.choice(tech_a) + random.choice(sufixos_resposta)
        else: # Chat Geral / Saudação
            saudacoes = ["Olá!", "Boa tarde.", "Como vai?", "Oi Rosa!", "Bom dia.", "Opa!"]
            respostas = ["Olá, como posso te ajudar?", "Tudo ótimo por aqui!", "Diga como a CromIA pode te auxiliar.", "Olá humano, em que posso ser útil?"]
            q = random.choice(saudacoes)
            a = random.choice(respostas)

        dados.append({
            "instruction": "Você é a Rosa, uma assistente virtual da CromIA. Responda de forma gentil.",
            "input": q,
            "output": a
        })

    # Embaralhar para garantir que os templates não fiquem em blocos
    random.shuffle(dados)
    
    output_path = "data/raw_corpus/persona_sintetica_50k.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for item in dados:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"✅ Sucesso! 25.000 amostras gravadas em {output_path}")

if __name__ == "__main__":
    gerar_persona_sintetica(25000)
