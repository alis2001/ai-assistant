import socket
import audioop
import numpy as np
from scipy.signal import resample
import whisper
import noisereduce as nr
import wave

# Server configuration
HOST = "0.0.0.0"  # Listen on all network interfaces
PORT = 8000       # Port for TCP connections
SAMPLE_RATE_ORIGINAL = 8000  # µ-law audio sample rate
SAMPLE_RATE_TARGET = 16000  # Whisper expects 16 kHz

# Output files for testing various qualities
RAW_AUDIO_FILE = "raw_audio.raw"
DECODED_AUDIO_FILE = "decoded_audio_8k.wav"
RESAMPLED_AUDIO_FILE_16K = "resampled_audio_16k.wav"
RESAMPLED_AUDIO_FILE_32K = "resampled_audio_32k.wav"
RESAMPLED_AUDIO_FILE_48K = "resampled_audio_48k.wav"
FINAL_PROCESSED_AUDIO_FILE = "processed_audio.wav"

# Load Whisper model
model = whisper.load_model("medium")
print("Whisper model loaded for Italian transcription.")

def save_raw_data(data, filename):
    """Save raw data to a file for debugging."""
    with open(filename, "wb") as f:
        f.write(data)
    print(f"Raw data saved to {filename}")

def save_wav(audio_data, sample_rate, filename):
    """Save audio data to a WAV file."""
    try:
        scaled_data = (audio_data * 32768).astype(np.int16)
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 16-bit audio
            wf.setframerate(sample_rate)  # Sample rate
            wf.writeframes(scaled_data.tobytes())
        print(f"Audio saved to {filename}")
    except Exception as e:
        print(f"Error saving WAV file {filename}: {e}")

def decode_resample_reduce_noise(ulaw_data):
    """
    Decode µ-law audio to PCM, resample it to various sample rates, and apply noise reduction.
    """
    try:
        # Save raw data for analysis
        save_raw_data(ulaw_data, RAW_AUDIO_FILE)

        # Decode µ-law to PCM
        pcm_data = audioop.ulaw2lin(ulaw_data, 2)  # µ-law to 16-bit PCM
        pcm_array = np.frombuffer(pcm_data, dtype=np.int16)

        # Save decoded PCM at 8kHz
        save_wav(pcm_array / 32768.0, SAMPLE_RATE_ORIGINAL, DECODED_AUDIO_FILE)

        # Resample to 16 kHz
        num_samples_16k = int(len(pcm_array) * (16000 / SAMPLE_RATE_ORIGINAL))
        resampled_16k = resample(pcm_array, num_samples_16k).astype(np.float32) / 32768.0
        save_wav(resampled_16k, 16000, RESAMPLED_AUDIO_FILE_16K)

        # Resample to 32 kHz
        num_samples_32k = int(len(pcm_array) * (32000 / SAMPLE_RATE_ORIGINAL))
        resampled_32k = resample(pcm_array, num_samples_32k).astype(np.float32) / 32768.0
        save_wav(resampled_32k, 32000, RESAMPLED_AUDIO_FILE_32K)

        # Resample to 48 kHz
        num_samples_48k = int(len(pcm_array) * (48000 / SAMPLE_RATE_ORIGINAL))
        resampled_48k = resample(pcm_array, num_samples_48k).astype(np.float32) / 32768.0
        save_wav(resampled_48k, 48000, RESAMPLED_AUDIO_FILE_48K)

        # Apply noise reduction to 16kHz audio
        reduced_noise = nr.reduce_noise(y=resampled_16k, sr=16000)
        save_wav(reduced_noise, 16000, FINAL_PROCESSED_AUDIO_FILE)

        return reduced_noise
    except Exception as e:
        print(f"Error in processing audio: {e}")
        raise e

def transcribe_audio_stream(audio_stream):
    """
    Transcribe an audio stream using Whisper.
    """
    try:
        print("Starting transcription...")
        result = model.transcribe(audio_stream, fp16=False, language="it", task="transcribe")
        transcription = result["text"].strip()
        print(f"Transcription: {transcription}")
        return transcription
    except Exception as e:
        print(f"Error during transcription: {e}")
        return "Errore durante la trascrizione."

def handle_client(client_socket, client_address):
    """
    Handle incoming client connection.
    Transcribe µ-law audio data in real-time and save processed audio.
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
            processed_audio = decode_resample_reduce_noise(audio_data)

            # Whisper expects audio as a NumPy array; pass directly
            transcription = transcribe_audio_stream(processed_audio)
            client_socket.sendall(transcription.encode("utf-8"))
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
            print(f"Server in ascolto su {HOST}:{PORT}")

            while True:
                client_socket, client_address = server_socket.accept()
                handle_client(client_socket, client_address)
    except Exception as e:
        print(f"Errore del server: {e}")
    finally:
        print("Chiusura del server.")

if __name__ == "__main__":
    start_server()
