# POC: llama_cpp & FUSE Mmap

## Tese
Utilizar a biblioteca leve baseada em C++ focada em CPU inference (`llama.cpp`) em Hardware de Memória Severamente Restrita (apenas 3GB livres).

## O Experimento
Iremos compilar `llama.cpp` e fazê-lo ler e instanciar *Safetensors/GGUF* que na verdade **não existem fisicamente**: a biblioteca solicitará um ponteiro de disco (`mmap()`) que será amarrado secretamente à ponte FUSE do CROM! 
O modelo não estourará a RAM de 3.5GB porque o FUSE descompactará o CROM Chunk dinamicamente na *Page Cache* em L1 apenas daquele peso neural O(1) exigido.
