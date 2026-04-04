#!/usr/bin/env python3
"""
CROM-IA V4.1 — Gerador de Dataset de Auto-Conhecimento
Lê todos os .md do projeto e transforma em pares Q&A para treino.
Resultado: Um cérebro que SABE o que é o CROM-IA.
"""

import os
import json
import re
import sys

# Onde procurar documentação
PROJETO_ROOT = "/home/j/Área de trabalho/crompressor-ia"

# Pastas a ignorar (bibliotecas externas, caches)
IGNORAR = [".venv", "node_modules", ".git", ".gemini", "llama.cpp", "v3.6_archived"]

# Templates de perguntas para cada tipo de conteúdo
TEMPLATES_QA = [
    ("O que é {titulo}?", "Com base na documentação do CROM-IA: {conteudo}"),
    ("Explique o conceito de {titulo} no contexto do CROM-IA.", "{conteudo}"),
    ("Como funciona {titulo}?", "De acordo com a arquitetura CROM-IA: {conteudo}"),
    ("Resuma {titulo} em detalhes técnicos.", "{conteudo}"),
]


def deve_ignorar(path):
    for ign in IGNORAR:
        if ign in path:
            return True
    return False


def extrair_titulo(conteudo, filename):
    """Tenta extrair o título do primeiro heading # do markdown."""
    match = re.search(r'^#\s+(.+)$', conteudo, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return os.path.splitext(filename)[0].replace("_", " ").replace("-", " ")


def dividir_em_secoes(conteudo):
    """Divide markdown em seções baseadas nos headings ##."""
    secoes = []
    partes = re.split(r'\n(?=##\s)', conteudo)
    for parte in partes:
        parte = parte.strip()
        if len(parte) > 100:  # Ignorar seções muito curtas
            secoes.append(parte)
    return secoes if secoes else [conteudo]


def coletar_markdown_projeto():
    """Coleta todos os .md do projeto, filtrando lixo."""
    arquivos = []
    for root, dirs, files in os.walk(PROJETO_ROOT):
        if deve_ignorar(root):
            continue
        for f in files:
            if f.endswith(".md"):
                path = os.path.join(root, f)
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as fp:
                        conteudo = fp.read()
                    if len(conteudo) > 200:  # Ignorar READMEs vazios
                        arquivos.append({"path": path, "filename": f, "conteudo": conteudo})
                except Exception:
                    pass
    return arquivos


def gerar_dataset_autoconhecimento(saida_path):
    """Gera pares Q&A a partir da documentação do projeto."""
    arquivos = coletar_markdown_projeto()
    total_pares = 0

    print(f"📖 Encontrados {len(arquivos)} documentos .md no projeto")

    with open(saida_path, "w", encoding="utf-8") as fout:
        for arq in arquivos:
            titulo = extrair_titulo(arq["conteudo"], arq["filename"])
            secoes = dividir_em_secoes(arq["conteudo"])

            for secao in secoes:
                # Limpar o texto (remover links, imagens, etc)
                secao_limpa = re.sub(r'!\[.*?\]\(.*?\)', '', secao)
                secao_limpa = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', secao_limpa)
                secao_limpa = secao_limpa.strip()

                if len(secao_limpa) < 100:
                    continue

                # Truncar seções muito longas para caber no contexto
                if len(secao_limpa) > 2000:
                    secao_limpa = secao_limpa[:2000] + "..."

                # Gerar múltiplos pares Q&A por seção
                for template_q, template_a in TEMPLATES_QA:
                    pergunta = template_q.format(titulo=titulo)
                    resposta = template_a.format(conteudo=secao_limpa)

                    entry = {
                        "instruction": pergunta,
                        "output": resposta
                    }
                    fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    total_pares += 1

    print(f"✅ Dataset de Auto-Conhecimento gerado!")
    print(f"   Pares Q&A: {total_pares}")
    print(f"   Saída: {saida_path}")
    return total_pares


def gerar_dataset_conversa_ptbr(saida_path, num_exemplos=2000):
    """
    Gera dataset de conversa genérica em PT-BR.
    Pares de instrução/resposta sobre tópicos variados.
    """
    conversas = [
        ("Olá, como você está?", "Olá! Estou funcionando perfeitamente. Sou o CROM-IA, um assistente brasileiro especializado em compressão de dados e inteligência artificial. Como posso ajudar?"),
        ("O que é inteligência artificial?", "Inteligência artificial é um campo da ciência da computação que desenvolve sistemas capazes de realizar tarefas que normalmente requerem inteligência humana, como reconhecimento de padrões, aprendizado, raciocínio e tomada de decisões."),
        ("Explique o que é Python.", "Python é uma linguagem de programação de alto nível, interpretada e de propósito geral. Foi criada por Guido van Rossum em 1991. É conhecida pela sua sintaxe limpa e legível, sendo amplamente utilizada em ciência de dados, desenvolvimento web, automação e inteligência artificial."),
        ("O que é compressão de dados?", "Compressão de dados é o processo de reduzir o tamanho de arquivos ou informações para ocupar menos espaço de armazenamento ou banda de transmissão. Existem dois tipos principais: compressão sem perda, onde os dados originais podem ser perfeitamente reconstruídos, e compressão com perda, onde alguma informação é descartada para obter maiores taxas de redução."),
        ("Como funciona uma rede neural?", "Uma rede neural artificial é inspirada no cérebro humano. Ela consiste em camadas de neurônios artificiais interconectados. Cada neurônio recebe entradas, aplica pesos e uma função de ativação, e passa o resultado adiante. Durante o treinamento, os pesos são ajustados através de backpropagation para minimizar o erro entre a saída prevista e a desejada."),
        ("O que é o Linux?", "Linux é um sistema operacional de código aberto baseado no kernel criado por Linus Torvalds em 1991. É amplamente utilizado em servidores, dispositivos embarcados e supercomputadores. Distribuições populares incluem Ubuntu, Fedora, Debian e Arch Linux. O Linux é conhecido pela sua estabilidade, segurança e flexibilidade."),
        ("Explique o que é uma API.", "API significa Application Programming Interface, ou Interface de Programação de Aplicações. É um conjunto de definições e protocolos que permitem que diferentes softwares se comuniquem entre si. Por exemplo, quando um aplicativo de clima mostra a previsão do tempo, ele usa uma API para buscar os dados de um servidor meteorológico."),
        ("O que é machine learning?", "Machine Learning, ou Aprendizado de Máquina, é um subcampo da inteligência artificial que permite que computadores aprendam padrões a partir de dados sem serem explicitamente programados. Os três tipos principais são: aprendizado supervisionado, não supervisionado e por reforço."),
        ("Como funciona a internet?", "A internet funciona através de uma rede global de computadores interconectados usando protocolos padronizados como TCP/IP. Quando você acessa um site, seu navegador envia uma requisição HTTP para um servidor DNS que traduz o nome do domínio em um endereço IP, e então se conecta ao servidor web para baixar o conteúdo da página."),
        ("O que é um banco de dados?", "Um banco de dados é um sistema organizado para armazenar, gerenciar e recuperar informações de forma eficiente. Existem bancos relacionais como PostgreSQL e MySQL que usam tabelas e SQL, e bancos NoSQL como MongoDB que usam documentos JSON. A escolha depende do tipo de dados e requisitos de performance da aplicação."),
        ("Explique o conceito de LoRA.", "LoRA significa Low-Rank Adaptation. É uma técnica de fine-tuning eficiente para modelos de linguagem grandes. Em vez de treinar todos os bilhões de parâmetros do modelo, LoRA congela os pesos originais e injeta pequenas matrizes treináveis nas camadas de atenção. Isso reduz drasticamente o uso de memória e tempo de treinamento, mantendo qualidade comparável ao fine-tuning completo."),
        ("O que é GGUF?", "GGUF é um formato de arquivo usado para armazenar modelos de linguagem quantizados, otimizado para inferência eficiente em CPUs. Foi criado pelo projeto llama.cpp como sucessor do formato GGML. Suporta quantização de 2 a 8 bits, permitindo rodar modelos de bilhões de parâmetros em hardware comum como laptops e Raspberry Pi."),
        ("O que significa quantização em IA?", "Quantização em IA é o processo de reduzir a precisão numérica dos pesos de um modelo neural. Por exemplo, converter pesos de 32 bits para 4 bits reduz o tamanho do modelo em 8 vezes. Técnicas como Q4_K_M mantêm a qualidade enquanto permitem rodar modelos grandes em hardware com pouca memória RAM."),
        ("O que é o CROM-IA?", "CROM-IA é um projeto brasileiro de inteligência artificial que implementa compressão termodinâmica de dados usando codificação DNA sub-simbólica. Ele permite rodar modelos de linguagem grandes em hardware limitado como processadores i5 sem placa de vídeo, usando técnicas de mmap zero-copy e micro-cérebros LoRA especializados."),
        ("Como funciona a compressão DNA no CROM-IA?", "A compressão DNA do CROM-IA funciona substituindo palavras frequentes por tokens compactos usando codificação Radix-4 inspirada no DNA biológico. Por exemplo, a palavra 'import' é comprimida para o token '@@IMP'. Isso reduz o número de tokens que o modelo precisa gerar, acelerando a inferência em hardware limitado."),
    ]

    total = 0
    with open(saida_path, "w", encoding="utf-8") as fout:
        # Repetir e variar as conversas para atingir volume
        for i in range(num_exemplos // len(conversas) + 1):
            for instrucao, resposta in conversas:
                entry = {"instruction": instrucao, "output": resposta}
                fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
                total += 1
                if total >= num_exemplos:
                    break
            if total >= num_exemplos:
                break

    print(f"✅ Dataset de Conversa PT-BR gerado!")
    print(f"   Total: {total} pares")
    print(f"   Saída: {saida_path}")
    return total


if __name__ == "__main__":
    base = "/home/j/Área de trabalho/crompressor-ia/v4.1_multibrain_engine/1_extracao_local/datasets_hibridos"
    os.makedirs(base, exist_ok=True)

    print("=" * 60)
    print("🧠 CROM-IA V4.1 — Gerador de Datasets Inteligentes")
    print("=" * 60)

    # 1. Dataset de Auto-Conhecimento (docs do projeto)
    print("\n📚 [1/2] Gerando dataset de Auto-Conhecimento do CROM-IA...")
    gerar_dataset_autoconhecimento(f"{base}/dataset_CROM_Self_raw.jsonl")

    # 2. Dataset de Conversa PT-BR
    print("\n💬 [2/2] Gerando dataset de Conversa PT-BR...")
    gerar_dataset_conversa_ptbr(f"{base}/dataset_Conversa_PTBR_raw.jsonl")

    print("\n" + "=" * 60)
    print("✅ TODOS OS DATASETS RAW GERADOS!")
    print("   Próximo passo: rodar transpilador_v41.py para aplicar mutação DNA")
    print("=" * 60)
