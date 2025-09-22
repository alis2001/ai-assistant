import socket
import sounddevice as sd
import numpy as np
import audioop
from queue import Queue
import threading

HOST = "6.tcp.eu.ngrok.io"  # Replace with your Ngrok host
PORT = 14684  # Replace with your Ngrok port
SAMPLE_RATE = 8000  # µ-law compatible sample rate
CHUNK_SIZE = 1024  # Audio chunk size

audio_queue = Queue()

def audio_callback(indata, frames, time, status):
    """Capture audio and enqueue it for processing."""
    if status:
        print(f"Audio stream status: {status}")
    pcm_data = (indata * 32768).astype(np.int16)  # Convert float32 to PCM
    ulaw_data = audioop.lin2ulaw(pcm_data.tobytes(), 2)  # Convert PCM to µ-law
    audio_queue.put(ulaw_data)

def audio_streamer(client_socket):
    """Send audio data from the queue to the server."""
    while True:
        data = audio_queue.get()
        if data is None:
            break
        client_socket.sendall(data)
    client_socket.sendall(b"END")  # Signal end of audio transmission
    print("Sent END signal to server.")

try:
    # Connect to the server
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((HOST, PORT))
    print(f"Connected to server at {HOST}:{PORT}")

    # Start the audio streaming thread
    streamer_thread = threading.Thread(target=audio_streamer, args=(client_socket,))
    streamer_thread.start()

    # Start recording audio
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback, blocksize=CHUNK_SIZE):
        print("Recording... Speak now. Press Enter to stop.")
        input()

    # Stop audio streaming thread
    audio_queue.put(None)
    streamer_thread.join()

    # Wait for server response
    response = client_socket.recv(1024).decode("utf-8")
    print(f"Server response: {response}")

except Exception as e:
    print(f"Error: {e}")

finally:
    if client_socket:
        client_socket.close()
        print("Connection closed.")
