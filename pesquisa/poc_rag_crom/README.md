# POC: CROM como Vetor Database Nativo (RAG)

## Tese
Software como Pinecone (VectorDB) sobrecarregam I/O e APIs. O CROM já é em seu próprio núcleo um Banco de Dados de similaridade cosenoidal.

## O Experimento
Usar a matriz interna `.cromdb` que guarda o dicionário de compressão para extrair "chunks de texto" que um usuário perguntar num RAG prompt, pulando completamente o uso de bancos de terceiros e economizando RAM em edge.
