import sys
import collections

def detect_loop(text, n=4, threshold=3):
    """
    Detects if any n-gram repeats more than 'threshold' times sequentially.
    """
    words = text.split()
    if len(words) < n * 2:
        return False
        
    ngrams = []
    for i in range(len(words) - n + 1):
        ngrams.append(" ".join(words[i:i+n]))
        
    # Check for sequential identical n-grams
    count = 0
    last_ngram = None
    for gram in ngrams:
        if gram == last_ngram:
            count += 1
            if count >= threshold:
                return True
        else:
            count = 1
        last_ngram = gram
            
    return False

if __name__ == "__main__":
    print("--- Guardrail Monitor Ativo (Detectando Loops de 4 palavras) ---")
    buffer = ""
    try:
        for line in sys.stdin:
            print(line, end="", flush=True)
            buffer += " " + line
            if detect_loop(buffer):
                print("\n\n⚠️  [GUARDRAIL] LOOP DETECTADO! Interrompendo fluxo...")
                sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(0)
