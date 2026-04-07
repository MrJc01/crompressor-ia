# 🚀 Guia de Implantação: Salto Cognitivo V4.3
**Versão:** CROM-IA v4.3 (Qwen 3.5 DNA Model)

## 📦 1. Recebimento do Chassis (GGUF)
Assim que o treinamento no Colab terminar, siga estes passos:
1.  Baixe o arquivo `CROM-IA_v4.3_Qwen3.5-2B.gguf` do Colab para o seu computador.
2.  Mova-o para a pasta: `/home/j/Área de trabalho/crompressor-ia/models/`.

## 🏗️ 2. Preparação da Infraestrutura SRE
Antes de iniciar o novo cérebro, execute a inicialização da ponte de memória compartilhada:
```bash
python3 "/home/j/Área de trabalho/crompressor-ia/scripts/setup_mmap_bridge.py"
```
*Isso criará o arquivo `mnt_crom/bridge_lsh_map.mmap` de 64MB para o motor Go.*

## 🧬 3. Iniciação do Motor de Inferência (A100 Trained)
Para testar a decodificação imediata (Modo CLI), use o novo script otimizado:
```bash
bash "/home/j/Área de trabalho/crompressor-ia/v4.3_cognitive_leap/chat_dna_v43_decode.sh"
```

## 🌐 4. Ativação do Visualizador Web (Modo Servidor)
Para usar a interface gráfica com o Qwen 3.5:
1.  O script `visualizador-sre/server.py` já foi atualizado para **ChatML**.
2.  Inicie normalmente:
```bash
cd "/home/j/Área de trabalho/crompressor-ia/visualizador-sre"
python3 server.py
```

## ⚠️ Notas de Estabilidade (SRE Hardening)
- **Contexto:** Limitamos o servidor a `2048` tokens para evitar o Swap Thrashing no hardware local.
- **Memória:** O Qwen 3.5 2B sintonizado consome ~1.4GB de RAM em modo 4-bit (Q4_K_M).
- **Go Engine:** O arquivo `.mmap` é zerado a cada execução do `setup_mmap_bridge.py`. Certifique-se de salvá-lo se contiver dados críticos.
