import whisper
import torch
import time

# Load the Whisper model
model = whisper.load_model("base")

# Set the device to GPU (if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Define the audio file path (you can change this as needed)
audio_file = "audio.wav"

# Continuous loop
while True:
    # Transcribe the audio file
    result = model.transcribe(audio_file)
    
    # Print the transcription
    print(result["text"])
    
    # Check GPU performance
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated(device) / (1024**2)  # in MB
        gpu_max_memory = torch.cuda.memory_reserved(device) / (1024**2)  # in MB
        gpu_name = torch.cuda.get_device_name(device)
        
        print(f"GPU: {gpu_name}")
        print(f"Memory Allocated: {gpu_memory:.2f} MB")
        print(f"Max Memory Reserved: {gpu_max_memory:.2f} MB")
    else:
        print("Running on CPU")

    # Sleep for a while before repeating the process (adjust the time as needed)
    time.sleep(5)  # Adjust the sleep time for performance monitoring interval
    print(device.center)