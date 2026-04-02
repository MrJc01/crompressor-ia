# POC: Offload de KV Cache para CROM

## Tese
Modelos locais de Linguagem sempre morrem pelo "OOM" (Out Of Memory) esgotando a RAM da Máquina com a matriz de Contexto (Tokens antigos de conversação).

## O Experimento
Quando o KV Cache exceder 50% de RAM, serializar os tensores passados e despejá-los numa Inode FUSE montada em CROM. A Engine Neural irá acessar memórias do usuário como quem acessa um "HD Antigo", mas instantaneamente com HNSW.
