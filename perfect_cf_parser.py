"""
BULLETPROOF CF Parser using advanced prompting techniques
LLM ONLY - No emergency fallbacks, make the LLM work properly
"""

import requests
import json
import re

class PerfectCFParser:
    def __init__(self, model="qwen2.5:14b"):
        self.model = model
        self.ollama_url = "http://localhost:11434/api/generate"
        print(f"🎯 BULLETPROOF CF Parser loaded with {model}")
    
    def parse_cf(self, transcription):
        """Use bulletproof prompting to extract CF - LLM ONLY"""
        
        # ORDER-PRESERVING PROMPT - MAINTAIN EXACT SEQUENCE
        prompt = f"""You are a CF extraction machine. PRESERVE ORIGINAL ORDER. No talking.

    CRITICAL: Process words LEFT TO RIGHT in EXACT ORDER from input.

    TASK: Convert speech to CF characters following these rules:
    Letter sequences: SDGL → S,D,G,L  
    City names: Roma→R, Napoli→N, Milano→M, Ancona→A, Torino→T, Bologna→B, Firenze→F
    Numbers: Keep exactly → 31→31, 49→49, 85→85
    Italian numbers: otto→8, cinque→5, nove→9, sei→6, sette→7

    EXAMPLES (PRESERVE ORDER):
    "ABCD 85 Roma Milano" → ABCD85RM (ABCD + 85 + R + M)
    "FG Roma 49 Napoli" → FGR49N (FG + R + 49 + N)
    "SDLK49 Roma-Napoli" → SDLK49RN (SDLK + 49 + R + N)

    PROCESS LEFT TO RIGHT: "{transcription}"
    OUTPUT:"""

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,        # Zero randomness
                "num_predict": 20,       # Very short response
                "stop": ["\n", "INPUT:", "EXAMPLES:", "TASK:", "PROCESS:"]  # Stop immediately
            }
        }
        
        try:
            response = requests.post(self.ollama_url, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                full_response = result['response'].strip()
                
                print(f"🧠 LLM Response: '{full_response}'")
                
                # Clean extraction - just take alphanumeric characters
                cf_code = ''.join(c for c in full_response.upper() if c.isalnum())
                
                print(f"🎯 EXTRACTED CF: '{cf_code}' from '{transcription}'")
                return cf_code
            else:
                print(f"❌ Ollama error: {response.status_code}")
                return ""
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return ""


# TEST THE BULLETPROOF VERSION
def test_bulletproof():
    parser = PerfectCFParser()
    
    # Test your exact failing cases
    test_cases = [
        "SDGL 31 Napoli Roma Ancona",        # Expected: SDGL31NRA
        "SDGL, Napoli, Roma 31 ancora",     # Expected: SDGLNR31A  
        "SDGLA 31 Roma-Napoli-Ancona",      # Expected: SDGLA31RNA
        "ABCD Roma Milano 85",               # Expected: ABCDRM85
        "F come Firenze R come Roma",        # Expected: FR
    ]
    
    print("🧪 TESTING BULLETPROOF CF PARSER")
    print("=" * 50)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n🧪 Test {i}: '{test}'")
        result = parser.parse_cf(test)
        print(f"✅ Result: '{result}' (length: {len(result)})")

if __name__ == "__main__":
    test_bulletproof()