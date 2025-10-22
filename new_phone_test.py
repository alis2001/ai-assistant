
import socket
import wave
import numpy as np
from scipy.signal import resample
# import whisper  # NO TRANSCRIPTION
import json
import time
import threading
import os
from datetime import datetime
import audioop
import subprocess

# Configuration
HOST = "0.0.0.0"
PORT = 8000
SAMPLE_RATE_ORIGINAL = 8000
SAMPLE_RATE_TARGET = 16000

print("Phone testing ready - NO TRANSCRIPTION")

# Directory for phone test files
PHONE_TEST_DIR = "phone_test_audio"
os.makedirs(PHONE_TEST_DIR, exist_ok=True)

def save_phone_test_audio(audio_data, filename_prefix, sample_rate=8000):
    """Save audio for phone testing as MP3 (like call center reference)"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # First save as WAV
    wav_filename = f"{PHONE_TEST_DIR}/{filename_prefix}_{timestamp}.wav"
    with wave.open(wav_filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data)
    
    # Convert to MP3 (like call center reference)
    mp3_filename = f"{PHONE_TEST_DIR}/{filename_prefix}_{timestamp}.mp3"
    try:
        # Use ffmpeg to convert WAV to MP3
        subprocess.run([
            'ffmpeg', '-y', '-i', wav_filename, '-acodec', 'mp3', 
            '-ar', str(sample_rate), '-ac', '1', mp3_filename
        ], check=True, capture_output=True)
        
        # Remove WAV file, keep only MP3
        os.remove(wav_filename)
        print(f"📱 Phone test audio saved as MP3: {mp3_filename}")
        return mp3_filename
        
    except Exception as e:
        print(f"⚠️ MP3 conversion failed, keeping WAV: {e}")
        print(f"📱 Phone test audio saved as WAV: {wav_filename}")
        return wav_filename

def alaw_to_linear_sample(a_val):
    """Custom A-law decoder matching call center server.js"""
    a_val ^= 0x55
    t = (a_val & 0x0F) << 4
    seg = (a_val & 0x70) >> 4
    if seg == 0:
        t += 8
    elif seg == 1:
        t += 0x108
    else:
        t += 0x108
        t <<= seg - 1
    return t if (a_val & 0x80) == 0 else -t

def decode_alaw_custom(buffer):
    """Custom A-law decoder matching call center approach"""
    pcm = bytearray(len(buffer) * 2)
    for i in range(len(buffer)):
        sample = alaw_to_linear_sample(buffer[i])
        pcm[i * 2] = sample & 0xFF
        pcm[i * 2 + 1] = (sample >> 8) & 0xFF
    return bytes(pcm)

def process_phone_test_audio(audio_data):
    """Match call center server.js EXACTLY - including frame protocol and FFMPEG"""
    try:
        print("📱 Using call center server.js EXACT approach...")
        
        # Step 1: Parse frames like they do (3-byte header: type, length, payload)
        print("🔍 Parsing frames with 3-byte headers...")
        buffer_acc = bytearray(audio_data)
        valid_frames = []
        
        while len(buffer_acc) >= 3:
            frame_type = buffer_acc[0]
            frame_length = int.from_bytes(buffer_acc[1:3], 'big')
            
            if len(buffer_acc) < 3 + frame_length:
                break
                
            payload = buffer_acc[3:3 + frame_length]
            buffer_acc = buffer_acc[3 + frame_length:]
            
            # Only process type 0x10 frames (like they do)
            if frame_type == 0x10:
                valid_frames.append(payload)
                print(f"✅ Frame type 0x10, length: {frame_length}")
            else:
                print(f"⚠️ Ignoring frame type 0x{frame_type:02x}")
        
        if not valid_frames:
            print("❌ No valid frames found, using raw data")
            valid_frames = [audio_data]
        
        # Step 2: Process valid frames with their custom decoder
        all_pcm = bytearray()
        for frame in valid_frames:
            frame_pcm = decode_alaw_custom(frame)
            all_pcm.extend(frame_pcm)
        
        save_phone_test_audio(bytes(all_pcm), "call_center_frame_protocol", 8000)
        print("✅ Step 1: Frame protocol + custom decoder")
        
        # Step 3: Apply their EXACT FFMPEG filters using subprocess
        try:
            import tempfile
            import subprocess
            
            # Save raw PCM file
            with tempfile.NamedTemporaryFile(suffix='.raw', delete=False) as temp_file:
                temp_file.write(all_pcm)
                temp_raw = temp_file.name
            
            # Apply their EXACT FFMPEG command
            mp3_output = f"{PHONE_TEST_DIR}/call_center_ffmpeg_exact_{time.strftime('%Y%m%d_%H%M%S')}.mp3"
            cmd = [
                'ffmpeg', '-y',
                '-f', 's16le', '-ar', '8000', '-ac', '1', '-i', temp_raw,
                '-filter_complex', '[0:a]highpass=f=200,lowpass=f=3400,dynaudnorm[a]',
                '-map', '[a]', '-ac', '1', '-ar', '8000', '-b:a', '192k', mp3_output
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Step 2: FFMPEG with dynaudnorm: {mp3_output}")
            else:
                print(f"⚠️ FFMPEG failed: {result.stderr}")
            
            # Cleanup
            os.unlink(temp_raw)
            
        except Exception as e:
            print(f"⚠️ FFMPEG processing failed: {e}")
        
        # Step 4: Test with their exact audioop approach (fallback)
        try:
            pcm_audioop = audioop.alaw2lin(audio_data, 2)
            save_phone_test_audio(pcm_audioop, "call_center_audioop_fallback", 8000)
            print("✅ Step 3: Audioop fallback")
        except Exception as e:
            print(f"⚠️ Audioop fallback failed: {e}")
        
        print("📱 Call center EXACT approach test complete!")
        print("🎧 The FFMPEG version should match their MP3 quality!")
        
        return True
        
    except Exception as e:
        print(f"❌ Call center EXACT approach failed: {e}")
        return False

# NO TRANSCRIPTION - Just audio quality testing

def handle_phone_test_client(client_socket, client_address):
    """Handle client for phone testing"""
    print(f"📱 PHONE TEST CONNECTION: {client_address}")
    call_start_time = time.time()
    
    try:
        audio_data = b""
        last_data_time = time.time()
        client_socket.settimeout(10.0)
        
        print(f"📱 Receiving audio for phone testing...")
        
        while True:
            current_time = time.time()
            
            try:
                data = client_socket.recv(4096)
                
                if not data:
                    print(f"📱 Client disconnected - processing {len(audio_data)} bytes")
                    break
                    
                if data == b"END":
                    print("📱 END signal received")
                    break
                
                audio_data += data
                last_data_time = current_time
                
                if len(audio_data) % 16384 == 0:
                    duration = len(audio_data) // 2 / SAMPLE_RATE_ORIGINAL
                    print(f"📱 Received ~{duration:.1f}s audio ({len(audio_data)} bytes)")
                
            except socket.timeout:
                time_since_data = current_time - last_data_time
                
                if time_since_data > 90.0:  # 90 seconds timeout
                    print(f"📱 Processing timeout - using {len(audio_data)} bytes")
                    break
                elif len(audio_data) == 0:
                    print(f"📱 No initial data timeout")
                    break

        if len(audio_data) > 0:
            print(f"📱 Processing {len(audio_data)} bytes for phone testing...")
            
            # Process and save test files
            success = process_phone_test_audio(audio_data)
            
            if success:
                response = {
                    "status": "success",
                    "test_files_created": True,
                    "test_directory": PHONE_TEST_DIR,
                    "message": "Audio files saved - check quality!",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
                response_json = json.dumps(response, ensure_ascii=False, indent=2)
                client_socket.sendall(response_json.encode("utf-8"))
            else:
                error_response = {
                    "status": "error",
                    "message": "Phone test processing failed",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                client_socket.sendall(json.dumps(error_response).encode("utf-8"))
        else:
            print("📱 No audio data received")

    except Exception as e:
        print(f"📱 Phone test client error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        try:
            client_socket.close()
        except:
            pass
        
        total_time = time.time() - call_start_time
        print(f"📱 Phone test call completed: {total_time:.1f}s")

def start_phone_test_server():
    """Start phone test server"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((HOST, PORT))
            server_socket.listen(5)
            
            print("📱 PHONE TEST SERVER")
            print("=" * 60)
            print(f"📍 Listening: {HOST}:{PORT}")
            print(f"📁 Test files will be saved in: {PHONE_TEST_DIR}")
            print(f"🎯 Purpose: Test different audio formats for phone quality")
            print(f"📞 Call the phone to test audio quality!")
            print("=" * 60)

            while True:
                try:
                    client_socket, client_address = server_socket.accept()
                    handle_phone_test_client(client_socket, client_address)
                except KeyboardInterrupt:
                    print("\n🛑 Phone test server shutdown")
                    break
                except Exception as e:
                    print(f"📱 Server error: {e}")
                    import traceback
                    traceback.print_exc()
                    
    except Exception as e:
        print(f"📱 Phone test server startup error: {e}")
    finally:
        print("📱 Phone test server shutdown")

if __name__ == "__main__":
    start_phone_test_server()
