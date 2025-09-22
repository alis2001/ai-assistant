"""
Super Explicit Ollama CF Parser - Forces step-by-step processing
"""

import requests
import json

class OllamaCFParser:
    def __init__(self, model="gemma3:12b"):
        self.model = model
        self.ollama_url = "http://localhost:11434/api/generate"
        print(f"✅ Super Explicit CF Parser loaded with {model}")
    
    def parse_cf(self, transcription):
        """Super explicit CF parsing with forced step-by-step processing"""
        
        prompt = f"""You must extract Italian Codice Fiscale characters. Process EVERY single word.

Input text: "{transcription}"

STEP 1: Break into individual words (split by spaces, commas, hyphens):
[List each word separately]

STEP 2: Convert EACH word using these rules:
- Single letters (A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z) → keep as-is
- Numbers (0,1,2,3,4,5,6,7,8,9,31,85,etc) → keep as-is
- Italian number words: zero→0, uno→1, due→2, tre→3, quattro→4, cinque→5, sei→6, sette→7, otto→8, nove→9
- City names → FIRST LETTER ONLY:
  Roma→R, Napoli→N, Milano→M, Torino→T, Bologna→B, Firenze→F, Ancona→A, Empoli→E, Genova→G, Salerno→S
- "come" → ignore this word

STEP 3: Combine all results in exact order

Example for "SDGL 31 Roma-Napoli-Ancona":
Word 1: SDGL → S,D,G,L
Word 2: 31 → 3,1  
Word 3: Roma-Napoli-Ancona → R,N,A
Final: SDGL31RNA

Now process: "{transcription}"
Word by word conversion:"""

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": 50,
                "stop": []
            }
        }
        
        try:
            response = requests.post(self.ollama_url, json=payload, timeout=25)
            
            if response.status_code == 200:
                result = response.json()
                full_response = result['response'].strip()
                
                print(f"🧠 LLM Full Response: {full_response[:200]}...")
                
                # Look for the final result - typically after "Final:" or at the end
                lines = full_response.split('\n')
                cf_candidates = []
                
                for line in lines:
                    # Look for lines with CF-like patterns
                    clean_line = ''.join(c for c in line if c.isalnum())
                    if clean_line and 3 <= len(clean_line) <= 16:
                        cf_candidates.append(clean_line)
                
                # Take the longest/most complete result
                if cf_candidates:
                    cf_final = max(cf_candidates, key=len)
                    print(f"🧠 Extracted CF: '{cf_final}' from '{transcription}'")
                    return cf_final
                else:
                    # Fallback: extract alphanumeric from end of response
                    cf_clean = ''.join(c for c in full_response.split('\n')[-1] if c.isalnum())[:16]
                    print(f"🧠 Fallback CF: '{cf_clean}' from '{transcription}'")
                    return cf_clean
                
            else:
                print(f"❌ Ollama error: {response.status_code}")
                return ""
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return ""

# Test the super explicit version
def test_super_explicit():
    parser = OllamaCFParser()
    
    # Test the exact problematic case
    test_input = "SDGL 31 Roma-Napoli-Ancona"
    print(f"🧪 Testing: '{test_input}'")
    print(f"Expected: 'SDGL31RNA' (9 characters)")
    
    result = parser.parse_cf(test_input)
    print(f"Got: '{result}' ({len(result)} characters)")
    
    if result == "SDGL31RNA":
        print("✅ SUCCESS!")
    else:
        print("❌ Still not working correctly")

if __name__ == "__main__":
    test_super_explicit()