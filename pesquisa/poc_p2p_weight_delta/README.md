# POC: Treinamento Distribuído Edge (P2P Delta)

## Tese
Não precisamos baixar um checkpoint novo de 5GB toda vez que uma LLM treina. Apenas alguns neurônios mudaram!

## O Experimento
Integrar BitSwap e Kademlia. Se a camada *Attention Head 0* de LlaMA altera-se em 0.05%, nossa IA exporta os safetensors alterados pelo Crompressor e apenas a Inode (KByte) delta transitará P2P para o outro parceiro. Treinamento mundial através de Smartphones (Edge Swarm).
