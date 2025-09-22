"""
CF Parser Service - Extracts Codice Fiscale from Italian transcriptions
Uses the existing city_names.json file from the root directory
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple

class CFParserService:
    """
    Codice Fiscale parser that matches city names from city_names.json
    and extracts first letters to build CF codes.
    """
    
    def __init__(self):
        """Initialize with city data from existing city_names.json"""
        self.cities = self._load_cities_from_json()
        self.city_lookup = self._build_city_lookup()
        self.number_map = self._build_italian_numbers()
        print(f"✅ CF Parser loaded with {len(self.cities)} Italian cities")
    
    def _load_cities_from_json(self) -> List[str]:
        """Load cities from the existing city_names.json file in root directory."""
        # Try multiple paths to find the city_names.json file
        possible_paths = [
            "city_names.json",                    # Current directory
            "../../../city_names.json",          # From api/app/services/ to root
            "../../city_names.json",             # Alternative path
            os.path.join(os.getcwd(), "city_names.json")  # Absolute from current working dir
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        cities_data = json.load(f)
                    
                    # Expecting format: ["Agliè", "Airasca", "Ala di Stura", ...]
                    if isinstance(cities_data, list) and cities_data:
                        # Clean the city names
                        clean_cities = []
                        for city in cities_data:
                            if city and isinstance(city, str) and city.strip():
                                clean_cities.append(city.strip())
                        
                        print(f"📍 Loaded {len(clean_cities)} cities from {path}")
                        return clean_cities
                        
                except Exception as e:
                    print(f"⚠️ Error loading {path}: {e}")
                    continue
        
        # Fallback if file not found
        print("⚠️ city_names.json not found, using minimal fallback cities")
        return [
            "Roma", "Milano", "Napoli", "Torino", "Firenze", "Bologna", 
            "Genova", "Palermo", "Bari", "Catania", "Venezia", "Verona"
        ]
    
    def _build_city_lookup(self) -> Dict[str, str]:
        """
        Build lookup dictionary for fast city name to first letter conversion.
        Format: {"roma": "R", "milano": "M", "agliè": "A", ...}
        """
        lookup = {}
        for city in self.cities:
            if city:
                # Store city in lowercase for case-insensitive matching
                city_lower = city.lower()
                first_letter = city[0].upper()  # Get first letter from original (handles accents)
                lookup[city_lower] = first_letter
        
        return lookup
    
    def _build_italian_numbers(self) -> Dict[str, str]:
        """Build Italian number words to digits mapping."""
        return {
            'zero': '0', 'uno': '1', 'due': '2', 'tre': '3', 'quattro': '4',
            'cinque': '5', 'sei': '6', 'sette': '7', 'otto': '8', 'nove': '9',
            'dieci': '10', 'undici': '11', 'dodici': '12', 'tredici': '13',
            'quattordici': '14', 'quindici': '15', 'sedici': '16', 'diciassette': '17',
            'diciotto': '18', 'diciannove': '19', 'venti': '20',
            # Common variations
            'un': '1', 'una': '1'
        }
    
    def parse_transcription_to_cf(self, transcription: str) -> Dict:
        """
        Main parsing function: converts transcription to CF code.
        
        Args:
            transcription: Italian speech transcription from Whisper
            
        Returns:
            Dict with CF parsing results
        """
        if not transcription or not transcription.strip():
            return self._empty_result(transcription)
        
        original_text = transcription.strip()
        
        # Normalize for processing (lowercase, clean spaces)
        normalized_text = self._normalize_text(original_text)
        words = normalized_text.split()
        
        cf_letters = []
        parsing_details = []
        
        # Process each word
        i = 0
        while i < len(words):
            word = words[i]
            processed = False
            
            # PATTERN 1: "X come [CityName]" -> validate and extract X
            if i + 2 < len(words) and words[i + 1] == "come":
                letter_candidate = word.upper()
                city_name = words[i + 2]
                
                # Check if it's a valid letter and city combination
                if (len(letter_candidate) == 1 and 
                    letter_candidate.isalpha() and 
                    city_name in self.city_lookup and
                    self.city_lookup[city_name] == letter_candidate):
                    
                    cf_letters.append(letter_candidate)
                    parsing_details.append({
                        'type': 'letter_come_city',
                        'result': letter_candidate,
                        'source': f"{word} come {city_name}",
                        'confidence': 1.0
                    })
                    i += 3  # Skip the "come city" part
                    processed = True
            
            # PATTERN 2: Direct city name -> extract first letter
            if not processed and word in self.city_lookup:
                letter = self.city_lookup[word]
                cf_letters.append(letter)
                parsing_details.append({
                    'type': 'city_name',
                    'result': letter,
                    'source': word,
                    'original_city': self._get_original_city_name(word),
                    'confidence': 0.95
                })
                processed = True
            
            # PATTERN 3: Italian number words -> digits
            if not processed and word in self.number_map:
                digits = self.number_map[word]
                # Add each digit separately for multi-digit numbers
                for digit in digits:
                    cf_letters.append(digit)
                
                parsing_details.append({
                    'type': 'number_word',
                    'result': digits,
                    'source': word,
                    'confidence': 0.9
                })
                processed = True
            
            # PATTERN 4: Direct digits
            if not processed and word.isdigit():
                # Add each digit separately
                for digit in word:
                    cf_letters.append(digit)
                
                parsing_details.append({
                    'type': 'digits',
                    'result': word,
                    'source': word,
                    'confidence': 0.8
                })
                processed = True
            
            # PATTERN 5: Single letters (already spelled out)
            if not processed and len(word) == 1 and word.isalpha():
                letter = word.upper()
                cf_letters.append(letter)
                parsing_details.append({
                    'type': 'single_letter',
                    'result': letter,
                    'source': word,
                    'confidence': 0.7
                })
                processed = True
            
            i += 1
        
        # Build final CF code
        cf_code = ''.join(cf_letters)
        
        # Calculate overall confidence
        total_confidence = sum(detail['confidence'] for detail in parsing_details)
        avg_confidence = total_confidence / len(parsing_details) if parsing_details else 0.0
        
        return {
            'original_transcription': original_text,
            'normalized_transcription': normalized_text,
            'cf_code': cf_code,
            'cf_letters': cf_letters,
            'length': len(cf_code),
            'is_complete_cf': len(cf_code) == 16,
            'confidence': avg_confidence,
            'parsing_details': parsing_details,
            'cities_matched': len([d for d in parsing_details if d['type'] in ['city_name', 'letter_come_city']]),
            'stats': {
                'total_words': len(words),
                'elements_parsed': len(parsing_details),
                'cities_in_database': len(self.cities)
            }
        }
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for processing (lowercase, clean spaces)."""
        return ' '.join(text.lower().split())
    
    def _get_original_city_name(self, normalized_city: str) -> str:
        """Get the original city name with proper capitalization."""
        for city in self.cities:
            if city.lower() == normalized_city:
                return city
        return normalized_city.title()
    
    def _empty_result(self, transcription: str) -> Dict:
        """Return empty result structure."""
        return {
            'original_transcription': transcription,
            'normalized_transcription': '',
            'cf_code': '',
            'cf_letters': [],
            'length': 0,
            'is_complete_cf': False,
            'confidence': 0.0,
            'parsing_details': [],
            'cities_matched': 0,
            'stats': {
                'total_words': 0,
                'elements_parsed': 0,
                'cities_in_database': len(self.cities)
            }
        }
    
    def get_parser_info(self) -> Dict:
        """Get information about the parser state."""
        return {
            'cities_loaded': len(self.cities),
            'sample_cities': self.cities[:10] if len(self.cities) >= 10 else self.cities,
            'lookup_entries': len(self.city_lookup),
            'number_mappings': len(self.number_map),
            'first_letters_available': sorted(list(set(self.city_lookup.values())))
        }

# Test function to verify the parser works
def test_cf_parser():
    """Test the CF parser with sample transcriptions."""
    parser = CFParserService()
    
    print(f"\n🧪 Testing CF Parser")
    print(f"📊 Parser Info: {parser.get_parser_info()}")
    
    # Test cases based on your examples
    test_cases = [
        "roma napoli milano",                    # Should extract: R-N-M
        "f come firenze r come roma",            # Should extract: F-R
        "agliè airasca",                        # Should extract: A-A (if these cities are loaded)
        "otto cinque roma",                      # Should extract: 8-5-R
        "milano otto cinque napoli due"          # Should extract: M-8-5-N-2
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n--- Test {i} ---")
        print(f"Input: '{test}'")
        
        result = parser.parse_transcription_to_cf(test)
        
        print(f"CF Code: '{result['cf_code']}'")
        print(f"CF Letters: {result['cf_letters']}")
        print(f"Cities Matched: {result['cities_matched']}")
        print(f"Confidence: {result['confidence']:.2f}")
        
        # Show detailed parsing
        for detail in result['parsing_details']:
            print(f"  {detail['type']}: '{detail['source']}' -> '{detail['result']}'")

if __name__ == "__main__":
    test_cf_parser()