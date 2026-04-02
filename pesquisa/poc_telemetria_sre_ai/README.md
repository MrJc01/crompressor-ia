# POC: Telemetria SRE de Data Lakes

## Tese
As arquiteturas clássicas sofrem com Overhead CGO e Cache Starvation no FUSE quando confrontando a leitura Multi-Threading excessiva.

## O Experimento
Automação em Bash (`htop`, `iosnoop`, eBPF / BPFTrace) com um painel provando fisicamente que a latência de uma Edge CPU de 2 cores com NVMe consegue ser inferior instalando sua engine CROM.
