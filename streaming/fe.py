import asyncio
import websockets
import pyaudio
import wave
import time
import os

uri = "wss://41df-93-63-213-92.ngrok-free.app"

chunk_size = 1024
sample_rate = 16000
channels = 1
format = pyaudio.paInt16
duration = 10

save_dir = "wavs"

if not os.path.exists(save_dir):
    os.makedirs(save_dir) 

async def stream_audio():
    p = pyaudio.PyAudio()
    stream_input = p.open(format=format,
                          channels=channels,
                          rate=sample_rate,
                          input=True,
                          frames_per_buffer=chunk_size)
    stream_output = p.open(format=format,
                           channels=channels,
                           rate=sample_rate,
                           output=True)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_filename = os.path.join(save_dir, f"recorded_audio_{timestamp}.wav")

    with wave.open(output_filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(format))
        wf.setframerate(sample_rate)

        print(f"Starting audio capture, streaming, playback, and saving to {output_filename}...")

        start_time = time.time()

        async with websockets.connect(uri) as websocket:
            print("Connected to WebSocket server.")
            while time.time() - start_time < duration:
                audio_data = stream_input.read(chunk_size, exception_on_overflow=False)
                wf.writeframes(audio_data)
                stream_output.write(audio_data)
                await websocket.send(audio_data)
                print(f"Sent {len(audio_data)} bytes of audio data.")

        print(f"Audio saved to {output_filename}")

    print("Streaming finished. Closing streams...")
    stream_input.stop_stream()
    stream_input.close()
    stream_output.stop_stream()
    stream_output.close()
    p.terminate()

asyncio.run(stream_audio())

