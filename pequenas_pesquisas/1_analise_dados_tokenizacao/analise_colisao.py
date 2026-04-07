import json
import os

def load_codebook(path):
    with open(path, 'r') as f:
        return json.load(f)

def check_collisions(codebook_data):
    cb = codebook_data['codebook']
    tokens = list(cb.values())
    keys = list(cb.keys())
    
    collisions = []
    
    # 1. Check for duplicate tokens
    seen_tokens = {}
    for k, v in cb.items():
        if v in seen_tokens:
            collisions.append(f"TOKEN_DUPLICADO: {v} usado para '{k}' e '{seen_tokens[v]}'")
        seen_tokens[v] = k
        
    # 2. Check if a token is a substring of a key (Self-corruption)
    for token in tokens:
        for key in keys:
            if token in key:
                collisions.append(f"SUBSTRING_COLLISION: Token '{token}' encontrado dentro da chave '{key}'")
                
    # 3. Check collision with common Python keywords
    py_keywords = ["def", "class", "return", "if", "else", "import", "from", "while", "for", "in", "print"]
    for token in tokens:
        for kw in py_keywords:
            if token in kw or kw in token:
                collisions.append(f"PYTHON_KEYWORD_COLLISION: Token '{token}' conflita com keyword '{kw}'")
                
    return collisions

if __name__ == "__main__":
    # Testando o novo símbolo químico para V4.3
    PREFIXO_SOBERANO = "⌬"
    cb_path = "/home/j/Área de trabalho/crompressor-ia/v4.2_multibrain_engine/1_extracao_local/codebooks/codebook_python_v42.json"
    print(f"--- [V4.3] Analisando Codebook com Prefixo: {PREFIXO_SOBERANO} ---")
    
    data = load_codebook(cb_path)
    
    # Simular a troca de prefixo para teste de colisão
    for k, v in data['codebook'].items():
        data['codebook'][k] = v.replace("@@", PREFIXO_SOBERANO)
        
    results = check_collisions(data)
    
    if not results:
        print("✅ Nenhuma colisão óbvia detectada no nível sintático.")
    else:
        print(f"⚠️ Detectadas {len(results)} possíveis colisões:")
        for r in results:
            print(f"  - {r}")
            
    print("\n--- Sugestão SRE ---")
    print("Aumentar o prefixo dos tokens de '@@' para algo mais exótico (ex: '##CROM_') "
          "se o modelo começar a gerar '@@' naturalmente em contextos de regex ou decoradores Python.")
