# POC: Datasets Genéticos (Epigenética de Memória)

## Tese
E se o modelo de descompressão `.cromdb` que alimenta o modelo PyTorch começasse a "sofrer mutações" com base na Loss Function do LLM?

## O Experimento
Desenvolver um Worker SRE que analisa a performance da Inference a cada Epoch. Se o Dataloader envia Tokens HNSW que não resultam num gradiente otimizado (Alta Perda), sinalizamos a Inode CROM para ela reconfigurar dinamicamente o Codebook de 16MB via *Epigenetic Spawning*. O formato comprimido ganharia Neuro-Plasticidade.
