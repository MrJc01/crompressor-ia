# Contribuindo com o CROM-IA 🧬

Obrigado por ter interesse em expandir o motor termodinâmico Sub-Simbólico! Nós almejamos transformar a inteligência artificial para o "Edge Hardware" mundial.

## 🚀 Como Preparar o Laboratório de Testes

1. Faça o Fork do repositório
2. Ative as permissões com `chmod +x scripts/*.sh`
3. Instale o LLama-CPP e ative: `source pesquisa/.venv/bin/activate`

## 📊 Nossos Pilares de Engenharia

Sua contribuição **deve** respeitar as diretrizes da Arquitetura do Motor CROM:
- **Zero-Swapping:** Nenhuma rota ou script pode gastar mais memória indiscriminadamente. Se sua PR adicionar overhead sem justificar otimização fractal, ela será bloqueada.
- **FUSE Independency:** Nós confiamos rigorosamente na montagem de VFS O(1).
- **Sem Cloud:** A API serve sempre `localhost`.

## 🛠️ Como submeter seu Pull Request (PR)

1. Crie a Branch (*Feature*): `git checkout -b feature/minha-ideia`
2. Escreva o Código E DOCUMENTAÇÃO
3. Garanta que o Smoke Test aprova: `./scripts/run_smoke_test.sh`
4. Comite as mudanças: `git commit -m "feat: adiciona camada O(1) de cache L1"`
5. Envie o Push para a Branch: `git push origin feature/minha-ideia`
6. Submeta o PR detalhando o impacto que terá no motor!

## 🛡️ SRE / Bug Tracking

Se você encontrou um cenário que levante `Kernel Panic` ou exploda Entropia acima de 7.8 na filtragem (Shannon Filter), crie uma ISSUE urgente documentando o RSS footprint.
