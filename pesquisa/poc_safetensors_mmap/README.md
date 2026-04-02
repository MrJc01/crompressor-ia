# POC: Extrema Safetensors Mmap

## Tese
Engano quem diz que para ler os pesos neurais de IA local você precisa de "Python Tensors Memory-Cpy".

## O Experimento
A biblioteca `safetensors.mmap` do Pytorch possui chamadas diretas C ao Kernel (`sys_mmap`). Fazer CROM expor `.safetensors` via FUSE. O Pytorch pensará que tem o modelo numa RAM direta rápida, e as page faults do Kernel trigarão as descodificações HNSW nos bastidores. Zero Memory Leak.
