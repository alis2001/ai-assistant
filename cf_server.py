#!/usr/bin/env python3
"""
Complete CF-Enabled Audio Server
Ready to use - includes CF parser embedded
"""

import socket
import wave
import audioop
import numpy as np
from scipy.signal import resample
import whisper
import json
import time
import os
import re
from typing import List, Dict, Tuple, Optional, Any

class CFParser:
    """Italian Codice Fiscale parser from speech transcription."""
    
    def __init__(self, city_data_path: str = None):
        """Initialize CF parser with city data."""
        self.cities = self._load_cities(city_data_path)
        self.city_lookup = self._build_city_lookup()
        self.number_map = self._build_number_map()
        
    def _load_cities(self, data_path: str = None) -> List[str]:
        """Load Italian city names from JSON file."""
        possible_paths = [
            'city_names.json',
            'data/city_names.json',
            '../data/city_names.json', 
            'istat-cities.json'
        ]
        
        if data_path:
            possible_paths.insert(0, data_path)
            
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        if 'istat' in path:
                            data = json.load(f)
                            return [entry.get("Denominazione in italiano", "") for entry in data if entry.get("Denominazione in italiano")]
                        else:
                            return json.load(f)
                except Exception as e:
                    print(f"Warning: Could not load {path}: {e}")
                    continue
        
        print("Warning: No city data found, using minimal set")
        return ["Roma", "Milano", "Napoli", "Torino", "Firenze", "Ancona", "Bologna"]
    
    def _build_city_lookup(self) -> Dict[str, List[str]]:
        """Build lookup dictionary: first letter -> list of cities."""
        lookup = {}
        for city in self.cities:
            if city:
                first_letter = city[0].upper()
                if first_letter not in lookup:
                    lookup[first_letter] = []
                lookup[first_letter].append(city.lower())
        return lookup
    
    def _build_number_map(self) -> Dict[str, str]:
        """Build Italian number word to digit mapping."""
        return {
            'zero': '0', 'uno': '1', 'due': '2', 'tre': '3', 'quattro': '4',
            'cinque': '5', 'sei': '6', 'sette': '7', 'otto': '8', 'nove': '9',
            'un': '1', 'una': '1'
        }
    
    def parse_cf_from_transcription(self, transcription: str) -> Dict:
        """Parse Codice Fiscale from Italian transcription."""
        if not transcription or not transcription.strip():
            return self._empty_result(transcription)
        
        text = transcription.lower().strip()
        words = text.split()
        
        cf_parts = []
        parsed_elements = []
        
        i = 0
        while i < len(words):
            word = words[i]
            found = False
            
            # Pattern 1: "X come [città]" -> extract X
            if i + 2 < len(words) and words[i + 1] == "come":
                letter_candidate = word.upper()
                city_name = words[i + 2]
                
                if (len(letter_candidate) == 1 and 
                    letter_candidate.isalpha() and
                    letter_candidate in self.city_lookup and
                    city_name in self.city_lookup[letter_candidate]):
                    
                    cf_parts.append(letter_candidate)
                    parsed_elements.append({
                        'type': 'letter_come_city',
                        'letter': letter_candidate,
                        'city': city_name,
                        'original': f"{word} come {city_name}"
                    })
                    i += 3
                    found = True
            
            # Pattern 2: Direct city name -> extract first letter
            if not found:
                for city in self.cities[:1000]:  # Check common cities
                    if city and word == city.lower():
                        letter = city[0].upper()
                        cf_parts.append(letter)
                        parsed_elements.append({
                            'type': 'city_name',
                            'letter': letter,
                            'city': city,
                            'original': word
                        })
                        found = True
                        break
            
            # Pattern 3: Italian numbers
            if not found and word in self.number_map:
                digit = self.number_map[word]
                cf_parts.append(digit)
                parsed_elements.append({
                    'type': 'number',
                    'digit': digit,
                    'original': word
                })
                found = True
            
            # Pattern 4: Direct digits
            if not found and word.isdigit():
                cf_parts.append(word)
                parsed_elements.append({
                    'type': 'digit',
                    'digit': word,
                    'original': word
                })
                found = True
            
            # Pattern 5: Single letters
            if not found and len(word) == 1 and word.isalpha():
                letter = word.upper()
                cf_parts.append(letter)
                parsed_elements.append({
                    'type': 'direct_letter',
                    'letter': letter,
                    'original': word
                })
                found = True
            
            i += 1
        
        cf_code = ''.join(cf_parts)
        
        return {
            'transcription': transcription,
            'parsed_cf': cf_code,
            'cf_parts': cf_parts,
            'parsed_elements': parsed_elements,
            'length': len(cf_code),
            'is_complete': len(cf_code) == 16,
            'confidence': self._calculate_confidence(parsed_elements)
        }
    
    def _calculate_confidence(self, parsed_elements: List[Dict]) -> float:
        """Calculate confidence score."""
        if not parsed_elements:
            return 0.0
        
        confidence_scores = {
            'letter_come_city': 1.0,
            'city_name': 0.9,
            'number': 0.8,
            'digit': 0.7,
            'direct_letter': 0.6
        }
        
        total_score = sum(confidence_scores.get(elem['type'], 0.5) for elem in parsed_elements)
        return min(total_score / len(parsed_elements), 1.0)
    
    def _empty_result(self, transcription: str) -> Dict:
        """Return empty parsing result."""
        return {
            'transcription': transcription,
            'parsed_cf': '',
            'cf_parts': [],
            'parsed_elements': [],
            'length': 0,
            'is_complete': False,
            'confidence': 0.0
        }

class CFAudioServer:
    """Audio server with Codice Fiscale parsing capabilities."""
    
    def __init__(self, host="0.0.0.0", port=8000):
        self.host = host
        self.port = port
        self.sample_rate_original = 8000
        self.sample_rate_target = 16000
        
        # Create output directories
        os.makedirs("audio_output", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        # Initialize Whisper
        print("🤖 Loading Whisper model...")
        self.model = whisper.load_model("medium")
        print("✅ Whisper model loaded for Italian transcription.")
        
        # Initialize CF parser
        print("🇮🇹 Loading CF parser...")
        self.cf_parser = CFParser()
        print(f"✅ CF parser loaded with {len(self.cf_parser.cities)} cities.")
    
    def process_and_save_audio(self, ulaw_data: bytes, output_file: str) -> bool:
        """Process and save audio."""
        try:
            pcm_data = audioop.ulaw2lin(ulaw_data, 2)
            pcm_array = np.frombuffer(pcm_data, dtype=np.int16)
            num_samples = int(len(pcm_array) * (self.sample_rate_target / self.sample_rate_original))
            resampled_array = resample(pcm_array, num_samples).astype(np.int16)

            with wave.open(output_file, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate_target)
                wf.writeframes(resampled_array.tobytes())

            print(f"✅ Audio saved: {output_file}")
            return True
        except Exception as e:
            print(f"❌ Audio processing error: {e}")
            return False

    def transcribe_audio(self, file_path: str) -> str:
        """Transcribe audio using Whisper."""
        try:
            print(f"🎤 Transcribing: {file_path}")
            result = self.model.transcribe(file_path, fp16=False, language="it")
            transcription = result["text"].strip()
            print(f"📝 Raw transcription: {transcription}")
            return transcription
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return "Errore durante la trascrizione."

    def handle_client(self, client_socket, client_address):
        """Handle client with CF processing."""
        print(f"🔗 Connection from {client_address}")
        
        try:
            audio_data = b""
            
            # Receive audio
            while True:
                data = client_socket.recv(4096)
                if not data:
                    break
                if data == b"END":
                    print("🏁 Processing audio...")
                    break
                audio_data += data

            if audio_data:
                print(f"📊 Received {len(audio_data)} bytes")
                
                # Process audio
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_file = f"audio_output/cf_{timestamp}.wav"
                
                if self.process_and_save_audio(audio_data, output_file):
                    # Transcribe
                    transcription = self.transcribe_audio(output_file)
                    
                    # Parse CF
                    cf_result = self.cf_parser.parse_cf_from_transcription(transcription)
                    
                    # Display results
                    print(f"🔍 CF ANALYSIS:")
                    print(f"   Input: {transcription}")
                    print(f"   CF:    {cf_result['parsed_cf']}")
                    print(f"   Parts: {cf_result['cf_parts']}")
                    print(f"   Length: {cf_result['length']}/16")
                    print(f"   Complete: {cf_result['is_complete']}")
                    print(f"   Confidence: {cf_result['confidence']:.2f}")
                    
                    # Prepare response
                    response = {
                        'status': 'success',
                        'cf_code': cf_result['parsed_cf'],
                        'is_complete': cf_result['is_complete'],
                        'confidence': cf_result['confidence'],
                        'length': cf_result['length'],
                        'raw_transcription': transcription,
                        'parsed_elements': cf_result['parsed_elements'],
                        'timestamp': time.time()
                    }
                    
                    response_json = json.dumps(response, ensure_ascii=False, indent=2)
                    client_socket.sendall(response_json.encode("utf-8"))
                    
                    # Log session
                    with open('logs/cf_sessions.log', 'a', encoding='utf-8') as f:
                        log_entry = {
                            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                            'client': str(client_address),
                            'cf_code': cf_result['parsed_cf'],
                            'transcription': transcription,
                            'confidence': cf_result['confidence']
                        }
                        f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
                
                else:
                    client_socket.sendall(b'{"status": "error", "message": "Audio processing failed"}')
            else:
                client_socket.sendall(b'{"status": "error", "message": "No audio received"}')

        except Exception as e:
            print(f"❌ Client error: {e}")
            error_msg = f'{{"status": "error", "message": "{str(e)}"}}'
            try:
                client_socket.sendall(error_msg.encode("utf-8"))
            except:
                pass

        finally:
            client_socket.close()
            print(f"🔚 Connection closed: {client_address}")

    def start_server(self):
        """Start the server."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
                server_socket.bind((self.host, self.port))
                server_socket.listen(5)
                
                print("🚀 CF AUDIO SERVER STARTED!")
                print(f"📍 Listening on {self.host}:{self.port}")
                print(f"🎯 Ready for Codice Fiscale processing...")
                print("💡 Try speaking: 'F come Firenze, R come Roma, otto cinque'")
                print("🔄 Waiting for connections...\n")

                while True:
                    client_socket, client_address = server_socket.accept()
                    self.handle_client(client_socket, client_address)
                    
        except KeyboardInterrupt:
            print("\n🛑 Server stopped by user")
        except Exception as e:
            print(f"❌ Server error: {e}")
        finally:
            print("🏁 Server shutdown complete")

if __name__ == "__main__":
    server = CFAudioServer()
    server.start_server()