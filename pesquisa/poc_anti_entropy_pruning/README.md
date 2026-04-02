# POC: Anti-Entropy Pruning 

## Tese
Modelos de IA aprendem "lixo" quando forçados a ler JSONLs mal filtrados do Scrapyng Web.

## O Experimento
Rodar o pacote Golang de termodinâmica de Shannon do CROM em um Dataset Base (`TinyStories` ou semelhante). Todo texto que retornar *Score H > 7* (puramente aleatório, códigos hexa, URLs infinitas) deve ser descartado O(1) pelo Pytorch antes de realizar o Forward Pass. Economia de GPU e modelo mais engenhoso.
