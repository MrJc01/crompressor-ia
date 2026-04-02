# POC: Transformadores (Tokenless)

## Tese
Tokenizadores (Ex: Byte-Pair Encoding BPE) desperdiçam ciclos de GPU reconstruindo o óbvio.

## O Experimento
Alimentar o embedding linear do modelo `PyTorch` passando diretamente os Hashes de FastCDC do Crompressor. Se FastCDC decidir que "Inconstitucionalissimamente" e " maçã" formam uma coisa só no disco, a rede neural deverá aceitar isso como entrada fundamental. O `Input Shape` passaria a modelar não letras, mas sim a termodinâmica abstrata do mundo através do LSH.
