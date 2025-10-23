import socket
import os
import time

class VoicePlayer:
    """Voice prompt player for call center workflow"""
    
    def __init__(self):
        self.voice_dir = "voice_prompts"
        self.prompt_files = {
            "start": "welcome.wav",
            "impegnativa_request": "impegnativa_request.wav", 
            "cf_retry": "cf_retry.wav",
            "cf_retry_2": "cf_retry_2.wav", 
            "cf_retry_3": "cf_retry_3.wav",
            "cf_failed": "cf_failed.wav",
            "impegnativa_retry": "impegnativa_retry.wav",
            "impegnativa_retry_2": "impegnativa_retry_2.wav",
            "success": "success.wav",
            "error": "error.wav"
        }
        print("🔊 Voice player initialized")
    
    def play_workflow_prompt(self, client_socket, prompt_type, attempt_number=None):
        """Play voice prompt based on workflow status"""
        try:
            # Determine which file to play based on prompt type and attempt
            if prompt_type == "cf_retry" and attempt_number is not None:
                if attempt_number == 1:
                    prompt_type = "cf_retry"
                elif attempt_number == 2:
                    prompt_type = "cf_retry_2"
                elif attempt_number >= 3:
                    prompt_type = "cf_retry_3"
            
            # Get the file path
            if prompt_type in self.prompt_files:
                filename = self.prompt_files[prompt_type]
                filepath = os.path.join(self.voice_dir, filename)
                
                if os.path.exists(filepath):
                    print(f"🔊 Playing voice prompt: {filename}")
                    self._send_audio_file(client_socket, filepath)
                    return True
                else:
                    print(f"⚠️ Voice file not found: {filepath}")
                    return False
            else:
                print(f"⚠️ Unknown prompt type: {prompt_type}")
                return False
                
        except Exception as e:
            print(f"❌ Error playing voice prompt {prompt_type}: {e}")
            return False
    
    def _send_audio_file(self, client_socket, filepath):
        """Send audio file to client in call center format"""
        try:
            # Check if file exists
            if not os.path.exists(filepath):
                print(f"❌ Voice file not found: {filepath}")
                return False
            
            # Set socket timeout for voice transmission
            client_socket.settimeout(10.0)
            
            print(f"🔊 Sending voice file: {os.path.basename(filepath)}")
            
            # Based on server.js analysis, send audio in the EXACT same format as incoming audio
            # Frame format: [0x10, 0x01, 0x40] + 320 bytes of A-law data
            # Send frames every 20ms (50 frames per second)
            
            with open(filepath, 'rb') as audio_file:
                wav_data = audio_file.read()
                
                # Convert WAV to PCM and then to A-law frames
                # For now, let's try sending the WAV data in the correct frame format
                
                # Get frame header (same as server.js)
                def get_header():
                    return bytes([0x10, 0x01, 0x40])  # type, length high, length low
                
                # Send the WAV data in 320-byte frames with headers
                frame_size = 320
                offset = 0
                
                while offset < len(wav_data):
                    # Get frame data
                    frame_data = wav_data[offset:offset + frame_size]
                    if len(frame_data) < frame_size:
                        # Pad with zeros if needed
                        frame_data = frame_data + b'\x00' * (frame_size - len(frame_data))
                    
                    # Send frame with header
                    frame = get_header() + frame_data
                    client_socket.sendall(frame)
                    
                    offset += frame_size
                    
                    # Wait 20ms between frames (like server.js)
                    import time
                    time.sleep(0.02)
                
                print(f"✅ Voice prompt sent successfully: {os.path.basename(filepath)}")
                return True
                
        except Exception as e:
            print(f"❌ Error sending audio file {filepath}: {e}")
            return False

# Create global instance
voice_player = VoicePlayer()
