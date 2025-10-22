import socket
import sounddevice as sd
import numpy as np
import json
from queue import Queue
import threading
import time
import select
import sys
import audioop

HOST = "15.160.171.248"  # AWS test server
PORT = 8080             # Port 8080
SAMPLE_RATE = 8000  # slin@8000: 8kHz sample rate
CHUNK_SIZE = 1024   # Audio chunk size
RECORDING_TIME = 30 # Maximum recording time in seconds

audio_queue = Queue()
recording_active = False

def audio_callback(indata, frames, time, status):
    """Capture audio and enqueue it for processing."""
    global recording_active
    
    if status:
        print(f"⚠️  Audio stream status: {status}")
    
    if recording_active:
        # ✅ Convert to µ-law (telephony format)
        pcm_data = (indata * 32767).astype(np.int16)  # Convert float32 to int16 PCM
        pcm_bytes = pcm_data.tobytes()  # Convert to bytes
        ulaw_bytes = audioop.lin2ulaw(pcm_bytes, 2)  # Convert to µ-law
        
        # Debug: Print byte info occasionally
        if len(ulaw_bytes) > 0 and frames % (SAMPLE_RATE * 2) == 0:  # Every 2 seconds
            print(f"🔍 Sending {len(ulaw_bytes)} bytes (µ-law format)")
        
        audio_queue.put(ulaw_bytes)  # Send µ-law bytes (telephony format)
        
        # Show recording progress
        if frames % (SAMPLE_RATE // 4) == 0:  # Every 0.25 seconds
            print("🎤 Recording µ-law@8000... (speak your CF clearly)")

def audio_streamer(client_socket):
    """Send audio data from the queue to the server."""
    global recording_active
    
    bytes_sent = 0
    start_time = time.time()
    
    while recording_active:
        try:
            data = audio_queue.get(timeout=1.0)  # 1 second timeout
            if data is None:
                break
                
            client_socket.sendall(data)
            bytes_sent += len(data)
            
            # Show progress (slin@8000: 2 bytes per sample)
            elapsed = time.time() - start_time
            if elapsed > 0 and bytes_sent % 8192 == 0:  # Every 8KB
                duration = bytes_sent / (8000 * 2)  # 2 bytes per sample at 8kHz
                print(f"📤 Sent {bytes_sent} bytes (~{duration:.1f}s slin@8000 audio)")
                
        except Exception as e:
            print(f"❌ Streaming error: {e}")
            break
    
    # Send END signal
    try:
        client_socket.sendall(b"END")
        print("✅ END signal sent to server")
        total_duration = bytes_sent / (8000 * 2)  # slin@8000 duration calculation
        print(f"📊 Total sent: {bytes_sent} bytes (~{total_duration:.1f}s slin@8000 audio)")
        
        # Check if byte count is even (should be for slin@8000)
        if bytes_sent % 2 != 0:
            print(f"⚠️ WARNING: Sent ODD byte count ({bytes_sent}) - this may cause server issues!")
        else:
            print(f"✅ Byte count is even ({bytes_sent}) - correct for slin@8000")
            
    except Exception as e:
        print(f"❌ Error sending END: {e}")

def test_cf_dictation():
    """Test CF dictation with slin@8000 client"""
    global recording_active
    
    try:
        # Connect to server
        print(f"🔗 Connecting to slin@8000 CF server at {HOST}:{PORT}")
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((HOST, PORT))
        print("✅ Connected to slin@8000 CF server!")

        # Start audio streaming thread
        recording_active = True
        streamer_thread = threading.Thread(target=audio_streamer, args=(client_socket,))
        streamer_thread.start()
        
        print("\n" + "="*60)
        print("🎤 SLIN@8000 CODICE FISCALE DICTATION TEST")
        print("="*60)
        print("📋 Instructions:")
        print("   • Format: slin@8000 (Signed Linear PCM 16-bit @ 8kHz)")
        print("   • Speak clearly and slowly")
        print("   • Use format: 'F come Firenze, R come Roma, otto cinque...'")
        print("   • Press ENTER when finished speaking")
        print("   • Server will process your CF automatically")
        print("="*60)
        
        # Start recording
        with sd.InputStream(
            samplerate=SAMPLE_RATE, 
            channels=1, 
            callback=audio_callback, 
            blocksize=CHUNK_SIZE
        ):
            print("🚀 Recording started! Dictate your Codice Fiscale now...")
            print("💡 Example: 'RSSMRA85M01H501Z: R come Roma, S come Salerno...'")
            
            # Wait for user to finish or timeout
            start_time = time.time()
            try:
                while True:
                    # Check for user input (non-blocking)
                    if select.select([sys.stdin], [], [], 0.1) == ([sys.stdin], [], []):
                        input()  # User pressed Enter
                        break
                    
                    # Check timeout
                    elapsed = time.time() - start_time
                    if elapsed >= RECORDING_TIME:
                        print(f"⏰ Recording timeout ({RECORDING_TIME}s)")
                        break
                    
                    time.sleep(0.1)
                    
            except KeyboardInterrupt:
                print("\n🛑 Recording stopped by user")
            except Exception as e:
                print(f"❌ Recording error: {e}")
        
        print("🔄 Stopping recording and processing...")
        recording_active = False
        
        # Stop audio streaming
        audio_queue.put(None)  # Signal to stop
        streamer_thread.join(timeout=5.0)
        
        # Wait for server response
        print("⏳ Waiting for slin@8000 CF analysis from server...")
        try:
            client_socket.settimeout(15.0)  # 15 second timeout for response
            response_data = b""
            
            while True:
                chunk = client_socket.recv(1024)
                if not chunk:
                    break
                response_data += chunk
                
                # Try to parse JSON (check if complete)
                try:
                    response_text = response_data.decode("utf-8")
                    response = json.loads(response_text)
                    break
                except:
                    continue  # Wait for more data
            
            if response_data:
                print("\n" + "="*60)
                print("📋 SLIN@8000 CF ANALYSIS RESULT:")
                print("="*60)
                
                try:
                    response = json.loads(response_data.decode("utf-8"))
                    print(f"📝 Transcription: {response.get('transcription', 'N/A')}")
                    print(f"🆔 CF Code: {response.get('cf_code', 'N/A')}")
                    print(f"📏 Length: {response.get('length', 0)}/16 characters")
                    print(f"✅ Complete: {'YES' if response.get('is_complete', False) else 'NO'}")
                    print(f"🎯 Processing Method: {response.get('processing_method', 'N/A')}")
                    print(f"⏰ Audio Duration: {response.get('audio_duration', 0):.1f}s")
                    
                    if response.get('is_complete'):
                        print(f"\n🎉 SUCCESS! Complete CF extracted: {response.get('cf_code')}")
                    else:
                        print(f"\n⚠️  Partial CF extracted. Please dictate remaining characters.")
                        
                except Exception as e:
                    print(f"❌ JSON parsing error: {e}")
                    print(f"Raw server response: {response_data.decode('utf-8', errors='ignore')}")
            else:
                print("❌ No response from server")
                
        except socket.timeout:
            print("⏰ Server response timeout")
        except Exception as e:
            print(f"❌ Response error: {e}")

    except ConnectionRefusedError:
        print(f"❌ Could not connect to {HOST}:{PORT}")
        print("💡 Make sure your slin@8000 CF server is running!")
    except Exception as e:
        print(f"❌ Client error: {e}")

    finally:
        recording_active = False
        try:
            client_socket.close()
        except:
            pass
        print("🔚 slin@8000 CF dictation test completed")

if __name__ == "__main__":
    # Test audio system first
    try:
        print("🔧 Testing audio system...")
        devices = sd.query_devices()
        print(f"✅ Audio system ready. Found {len(devices)} audio devices.")
        print(f"🎯 Client configured for slin@8000 (Signed Linear PCM 16-bit @ 8kHz)")
        
        # Run CF dictation test
        test_cf_dictation()
        
    except Exception as e:
        print(f"❌ Audio system error: {e}")
        print("💡 Install: pip install sounddevice")