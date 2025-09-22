"""
SIMPLE CF Parser - FIRST LETTER EXTRACTION ONLY
No city database needed - just extract first letters of words
"""

import requests
import json
import re

class PerfectCFParser:
    def __init__(self, model="qwen2.5:14b"):
        self.model = model
        self.ollama_url = "http://localhost:11434/api/generate"
        print(f"🎯 SIMPLE CF Parser loaded with {model}")
    
    def parse_cf(self, transcription):
        """Multiple prompt strategies to force direct results"""
        
        # Try strategy 1: Ultra direct
        strategies = [
            # Strategy 1: Command style
            f"{transcription} → ",
            
            # Strategy 2: Direct conversion  
            f"Convert: {transcription}\nCF: ",
            
            # Strategy 3: Minimal prompt
            f"Text: {transcription}\nExtract CF code (letters+numbers only): "
        ]
        
        for i, prompt in enumerate(strategies, 1):
            print(f"🎯 Trying strategy {i}...")
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_predict": 20,
                    "top_k": 1,  # Force most likely token
                    "stop": ["\n", " ", "→", "Based", "Rules", ":", "CF"]
                }
            }
            
            try:
                response = requests.post(self.ollama_url, json=payload, timeout=15)
                
                if response.status_code == 200:
                    result = response.json()
                    full_response = result['response'].strip()
                    
                    print(f"🧠 Strategy {i} Response: '{full_response}'")
                    
                    # Extract just alphanumeric
                    cf_code = ''.join(c for c in full_response.upper() if c.isalnum())
                    
                    # If we got a reasonable result, use it
                    if cf_code and 4 <= len(cf_code) <= 16 and not any(word in full_response.lower() for word in ['based', 'rule', 'convert']):
                        print(f"🎯 SUCCESS with strategy {i}: '{cf_code}' from '{transcription}'")
                        return cf_code
                    else:
                        print(f"❌ Strategy {i} failed: '{cf_code}' (length: {len(cf_code)})")
                        
            except Exception as e:
                print(f"❌ Strategy {i} error: {e}")
                continue
        
        # If all strategies fail, use manual fallback
        print(f"🔄 All LLM strategies failed, using manual parsing...")
        return self.parse_cf_manual(transcription)

    def parse_cf_manual(self, transcription):
        """RELIABLE manual implementation - Use as PRIMARY method"""
        
        # Clean and split input
        words = transcription.replace(',', ' ').replace('-', ' ').split()
        result = []
        
        print(f"🔧 Manual parsing: {words}")
        
        for word in words:
            word = word.strip()
            if not word:
                continue
            
            print(f"   Processing: '{word}'")
            
            # Pure numbers (like 47, 85, 42)
            if word.isdigit():
                result.append(word)
                print(f"     → Number: {word}")
                
            # Pure uppercase letter sequences (like FK, LK, SDGL)
            elif word.isupper() and word.isalpha() and len(word) <= 6:
                result.append(word)
                print(f"     → Letters: {word}")
                
            # Mixed alphanumeric (like SDGL21D47)
            elif any(c.isdigit() for c in word) and any(c.isalpha() for c in word):
                cleaned = ''.join(c for c in word.upper() if c.isalnum())
                result.append(cleaned)
                print(f"     → Alphanumeric: {cleaned}")
                
            # Any other word - take first letter (like Napoli→N, Roma→R)
            elif word.isalpha():
                first_letter = word[0].upper()
                result.append(first_letter)
                print(f"     → Word→First: {word}→{first_letter}")
                
            # Mixed case with special chars - clean and process
            else:
                cleaned = ''.join(c for c in word if c.isalnum())
                if cleaned.isdigit():
                    result.append(cleaned)
                    print(f"     → Cleaned number: {cleaned}")
                elif cleaned.isupper() and len(cleaned) <= 6:
                    result.append(cleaned)
                    print(f"     → Cleaned letters: {cleaned}")
                else:
                    first_letter = cleaned[0].upper() if cleaned else ''
                    if first_letter:
                        result.append(first_letter)
                        print(f"     → Cleaned→First: {word}→{first_letter}")
                
        final_cf = ''.join(result)
        print(f"🎯 Manual result: '{final_cf}' from parts {result}")
        return final_cf

    def parse_cf_primary(self, transcription):
        """Primary parsing method - Manual first, LLM as backup"""
        
        print(f"🔧 Using MANUAL parsing as primary method...")
        manual_result = self.parse_cf_manual(transcription)
        
        # If manual result looks good, use it
        if manual_result and 4 <= len(manual_result) <= 16:
            print(f"✅ Manual parsing successful: '{manual_result}'")
            return manual_result
        
        # Otherwise try LLM as backup
        print(f"🤖 Manual parsing unclear, trying LLM backup...")
        return self.parse_cf(transcription)


def test_simple_parser():
    parser = PerfectCFParser()
    
    test_cases = [
        ("LK Empoli Roma 54 62", "LKER5462"),
        ("Sdgla21d47 Empoli Roma", "SDGLA21D47ER"), 
        ("FK Firenze Palermo Ancona 425", "FKFPA425"),
        ("ABCD hello world 85", "ABCDHW85"),
        ("test word 123", "TW123"),
    ]
    
    print("🧪 TESTING SIMPLE CF PARSER")
    print("=" * 50)
    
    for i, (test_input, expected) in enumerate(test_cases, 1):
        print(f"\n🧪 Test {i}: '{test_input}'")
        print(f"📝 Expected: '{expected}'")
        
        # Test LLM version
        llm_result = parser.parse_cf(test_input)
        print(f"🧠 LLM: '{llm_result}'")
        
        # Test manual version for comparison
        manual_result = parser.parse_cf_manual(test_input)
        print(f"⚙️  Manual: '{manual_result}'")
        
        status = "✅ PASS" if llm_result == expected else "❌ FAIL"
        print(f"{status} LLM Result")

if __name__ == "__main__":
    test_simple_parser()