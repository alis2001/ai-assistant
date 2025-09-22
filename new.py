import socket
import wave
import audioop
import numpy as np
from scipy.signal import resample
import whisper
import json
import time

# Server configuration
HOST = "0.0.0.0"  # Listen on all network interfaces
PORT = 8000       # Port for TCP connections
SAMPLE_RATE_ORIGINAL = 8000  # µ-law audio sample rate
SAMPLE_RATE_TARGET = 16000  # Target sample rate

# Load Whisper model
model = whisper.load_model("medium")
print("Whisper model loaded for Italian transcription.")

class SimpleCFParser:
    """Reliable manual CF parser - no LLM needed"""
    
    def __init__(self):
        print("🎯 Simple CF Parser loaded (manual only)")
    
    def parse_cf(self, transcription):
        """Manual CF parsing with detailed logging"""
        
        # Clean and split input
        words = transcription.replace(',', ' ').replace('-', ' ').split()
        result = []
        
        print(f"🔧 Manual parsing: {words}")
        
        for word in words:
            word = word.strip()
            if not word:
                continue
            
            print(f"   Processing: '{word}'")
            
            # Pure numbers (like 47, 85, 425)
            if word.isdigit():
                result.append(word)
                print(f"     → Number: {word}")
                
            # Pure uppercase letter sequences (like FK, FKF, SDGL)
            elif word.isupper() and word.isalpha() and len(word) <= 6:
                result.append(word)
                print(f"     → Letters: {word}")
                
            # Mixed alphanumeric (like SDGL21D47)
            elif any(c.isdigit() for c in word) and any(c.isalpha() for c in word):
                cleaned = ''.join(c for c in word.upper() if c.isalnum())
                result.append(cleaned)
                print(f"     → Alphanumeric: {cleaned}")
                
            # Any other word - take first letter (like Napoli→N, Ancona→A)
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

# Initialize CF parser
cf_parser = SimpleCFParser()

def process_and_save_audio(ulaw_data, output_file):
    """
    Decode µ-law data, resample to 16 kHz, and save as a WAV file.
    """
    try:
        pcm_data = audioop.ulaw2lin(ulaw_data, 2)  # µ-law to 16-bit PCM
        pcm_array = np.frombuffer(pcm_data, dtype=np.int16)

        # Resample to 16 kHz
        num_samples = int(len(pcm_array) * (SAMPLE_RATE_TARGET / SAMPLE_RATE_ORIGINAL))
        resampled_array = resample(pcm_array, num_samples).astype(np.int16)

        # Save as WAV file
        with wave.open(output_file, "wb") as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 16-bit audio
            wf.setframerate(SAMPLE_RATE_TARGET)  # 16 kHz sample rate
            wf.writeframes(resampled_array.tobytes())

        print(f"Audio saved to {output_file}")
        return True
    except Exception as e:
        print(f"Error processing audio: {e}")
        return False

def transcribe_audio(file_path):
    """
    Transcribe audio using Whisper in Italian.
    """
    try:
        print(f"Transcribing audio from {file_path}...")
        result = model.transcribe(file_path, fp16=False, language="it")
        transcription = result["text"].strip()
        print(f"Transcription (Italian): {transcription}")
        return transcription
    except Exception as e:
        print(f"Error during transcription: {e}")
        return "Errore durante la trascrizione."

def handle_client(client_socket, client_address):
    """
    Handle incoming client connection.
    Receives µ-law audio data, processes it, saves it to a WAV file, and transcribes it in Italian.
    """
    print(f"Connessione stabilita con {client_address}")
    try:
        audio_data = b""
        while True:
            data = client_socket.recv(4096)
            if not data:
                print(f"Il client {client_address} si è disconnesso inaspettatamente.")
                break
            if data == b"END":  # End of audio signal
                print("Segnale END ricevuto. Elaborazione dell'audio...")
                break
            audio_data += data

        if audio_data:
            print(f"Dati audio ricevuti: {len(audio_data)} byte.")
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = f"received_audio_{timestamp}.wav"
            
            if process_and_save_audio(audio_data, output_file):
                transcription = transcribe_audio(output_file)
                
                # Use reliable manual CF parsing
                print("🔧 Using manual CF extraction...")
                cf_code = cf_parser.parse_cf(transcription)
                
                # Display results
                print(f"🔍 CF ANALYSIS:")
                print(f"   Input: {transcription}")
                print(f"   CF Code: {cf_code}")
                print(f"   Length: {len(cf_code)}/16")
                print(f"   Complete: {'✅ YES' if len(cf_code) == 16 else '❌ NO'}")
                
                # Send response
                response = {
                    "status": "success",
                    "transcription": transcription,
                    "cf_code": cf_code,
                    "length": len(cf_code),
                    "is_complete": len(cf_code) == 16,
                    "method": "Manual Parser",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
                client_socket.sendall(json.dumps(response, ensure_ascii=False).encode("utf-8"))
            else:
                client_socket.sendall(b"Errore nel salvataggio o nell'elaborazione dell'audio.")
        else:
            print(f"Nessun audio ricevuto da {client_address}.")
            client_socket.sendall(b"Nessun audio ricevuto.")

    except Exception as e:
        print(f"Errore nella gestione del client {client_address}: {e}")

    finally:
        client_socket.close()
        print(f"Connessione chiusa con {client_address}")

def start_server():
    """
    Start the TCP server.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.bind((HOST, PORT))
            server_socket.listen(5)  # Allow up to 5 simultaneous connections
            print(f"🚀 CF Audio Server running on {HOST}:{PORT}")
            print(f"🎯 Using reliable manual CF parsing")
            print(f"🔄 Ready for connections...")

            while True:
                client_socket, client_address = server_socket.accept()
                handle_client(client_socket, client_address)
    except Exception as e:
        print(f"Errore del server: {e}")
    finally:
        print("Chiusura del server.")

if __name__ == "__main__":
    start_server()