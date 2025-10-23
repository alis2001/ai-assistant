import socket
import wave
import numpy as np
from scipy.signal import resample
from scipy import signal
import whisper
import json
import time
import threading
import psutil
import torch
import struct
import audioop
import os
import shutil
from datetime import datetime, timedelta
import urllib.request
import urllib.error
import queue
import uuid

try:
    import librosa
    import noisereduce as nr
    from scipy.signal import butter, filtfilt, wiener
    ENHANCED_AUDIO = True
    print("Technical noise reduction libraries loaded")
except ImportError:
    ENHANCED_AUDIO = False
    print("Install: 'pip install librosa noisereduce' for technical noise reduction")

HOST = "0.0.0.0"
PORT = 8000
SAMPLE_RATE_ORIGINAL = 8000
SAMPLE_RATE_TARGET = 16000

NOISE_PROFILE_LENGTH = 0.5
SPECTRAL_FLOOR = 0.1
WIENER_FILTER_SIZE = 5

CLIENT_TIMEOUT = 10.0
MAX_CALL_DURATION = 120.0
CF_DICTATION_TIME = 90.0

print("Loading Whisper LARGE with technical noise optimization...")
model = whisper.load_model("large")
print("Technical noise-resistant transcription ready")

# Professional queuing system for call center
transcription_queue = queue.Queue()
transcription_workers = []
client_responses = {}  # Store client sockets for responses
response_lock = threading.Lock()

# Two-step workflow tracking
client_workflows = {}  # Store workflows by client_id

# File management configuration - Only transcription files
TRANSCRIPTION_DIR = "transcription_files"
MAX_FILES_PER_DIR = 100  # Keep only latest 100 transcription files
CLEANUP_AGE_DAYS = 3     # Remove files older than 3 days

# Results log for frontend listing
CF_RESULTS_LOG = "cf_results.jsonl"

# Optional remote ingest (VM frontend) configuration
VM_INGEST_URL = os.environ.get("VM_INGEST_URL")  # e.g., http://10.10.13.122:38473/api/ingest
VM_INGEST_TOKEN = os.environ.get("VM_INGEST_TOKEN")  # shared secret, optional

def setup_directories():
    """Create and setup directory structure - only transcription files"""
    os.makedirs(TRANSCRIPTION_DIR, exist_ok=True)
    print(f"📁 Transcription directory ready: {TRANSCRIPTION_DIR}")

def cleanup_old_files(directory, max_files=MAX_FILES_PER_DIR, max_age_days=CLEANUP_AGE_DAYS):
    """Clean up old files in a directory"""
    if not os.path.exists(directory):
        return
    
    try:
        # Get all files with their modification times
        files = []
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                mtime = os.path.getmtime(filepath)
                files.append((filepath, mtime))
        
        # Sort by modification time (newest first)
        files.sort(key=lambda x: x[1], reverse=True)
        
        # Remove files older than max_age_days
        cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
        removed_old = 0
        for filepath, mtime in files:
            if mtime < cutoff_time:
                os.remove(filepath)
                removed_old += 1
        
        # Remove excess files (keep only max_files)
        removed_excess = 0
        if len(files) - removed_old > max_files:
            for filepath, mtime in files[max_files:]:
                if os.path.exists(filepath):
                    os.remove(filepath)
                    removed_excess += 1
        
        if removed_old > 0 or removed_excess > 0:
            print(f"🧹 Cleaned {directory}: {removed_old} old files, {removed_excess} excess files")
            
    except Exception as e:
        print(f"⚠️ Cleanup error in {directory}: {e}")

def get_transcription_filename(client_id):
    """Generate transcription filename with client ID and timestamp"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(TRANSCRIPTION_DIR, f"transcription_{client_id}_{timestamp}.wav")

def cleanup_old_transcription_files():
    """Clean up old transcription files periodically"""
    try:
        cleanup_old_files(TRANSCRIPTION_DIR, MAX_FILES_PER_DIR, CLEANUP_AGE_DAYS)
    except Exception as e:
        print(f"⚠️ Transcription cleanup error: {e}")

def append_cf_result(result):
    """Append a single CF result entry to JSONL log for frontend consumption"""
    try:
        # Ensure minimal stable schema
        entry = {
            "timestamp": result.get("timestamp") or time.strftime("%Y-%m-%d %H:%M:%S"),
            "cf_code": result.get("cf_code", ""),
            "is_complete": bool(result.get("is_complete", False)),
            "confidence": float(result.get("confidence", 0.0)),
            "length": int(result.get("length", 0)),
            "transcription": result.get("transcription", "")
        }
        with open(CF_RESULTS_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"⚠️ Could not write CF result log: {e}")

def post_cf_result_to_vm(result):
    """Post CF result to remote VM ingest endpoint if configured"""
    if not VM_INGEST_URL:
        return
    data = json.dumps(result).encode("utf-8")
    req = urllib.request.Request(VM_INGEST_URL, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if VM_INGEST_TOKEN:
        req.add_header("Authorization", f"Bearer {VM_INGEST_TOKEN}")
    last_err = None
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(req, timeout=7.0) as resp:
                if 200 <= resp.status < 300:
                    print("📤 Ingested CF result to VM")
                    return
                else:
                    print(f"⚠️ VM ingest non-OK (attempt {attempt}): {resp.status}")
        except urllib.error.HTTPError as e:
            last_err = e
            print(f"⚠️ VM ingest HTTP error (attempt {attempt}): {e}")
        except Exception as e:
            last_err = e
            print(f"⚠️ VM ingest failed (attempt {attempt}): {e}")
        time.sleep(0.75 * attempt)
    if last_err:
        print(f"⚠️ VM ingest ultimately failed after retries: {last_err}")

class TechnicalCFParser:
    
    def __init__(self):
        print("Technical CF Parser loaded")
    
    def parse_cf(self, transcription):
        """Simple CF parsing - NO MAPPING, just extract first letters and numbers"""
        
        words = transcription.replace(',', ' ').replace('-', ' ').replace('.', ' ').split()
        
        cf_parts = []
        
        for word in words:
            word = word.strip()
            
            # Numbers - keep as is
            if word.isdigit():
                cf_parts.append(word)
                continue
            
            # Words - check if it's 1-3 capital letters (treat as individual letters)
            if word.isalpha() and len(word) <= 3 and word.isupper():
                for char in word:
                    cf_parts.append(char)
                continue
            
            # Words - take first letter only (for longer words or lowercase)
            if word.isalpha():
                cf_parts.append(word[0].upper())
                continue
            
            # Mixed alphanumeric - take each character
            if any(c.isalpha() for c in word):
                for char in word.upper():
                    if char.isalpha() or char.isdigit():
                        cf_parts.append(char)
                continue
        
        cf_code = ''.join(cf_parts)
        
        return {
            'cf_code': cf_code,
            'length': len(cf_code),
            'is_complete': len(cf_code) == 16,
            'parts': cf_parts,
            'confidence': min(1.0, len(cf_parts) / 16.0) if cf_parts else 0.0
        }

class ImpegnativaParser:
    """Parser for 6-digit impegnativa numbers"""
    
    def __init__(self):
        print("Impegnativa Parser loaded")
    
    def parse_impegnativa(self, transcription):
        """Extract 6-digit number from transcription"""
        import re
        
        # Remove common words and clean text
        text = transcription.lower()
        text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
        words = text.split()
        
        # Look for 6-digit numbers
        for word in words:
            if word.isdigit() and len(word) == 6:
                return {
                    'number': word,
                    'length': len(word),
                    'is_valid': True,
                    'confidence': 1.0
                }
        
        # Look for numbers that might be spoken as separate digits
        digits = []
        for word in words:
            if word.isdigit() and len(word) == 1:
                digits.append(word)
        
        if len(digits) >= 6:
            # Take first 6 digits
            number = ''.join(digits[:6])
            return {
                'number': number,
                'length': len(number),
                'is_valid': True,
                'confidence': 0.8
            }
        
        # No valid number found
        return {
            'number': '',
            'length': 0,
            'is_valid': False,
            'confidence': 0.0
        }

class CallCenterWorkflow:
    """Manages the two-step call center workflow"""
    
    def __init__(self):
        self.step = 1  # 1 = CF, 2 = Impegnativa
        self.cf_code = ""
        self.impegnativa = ""
        self.cf_attempts = 0
        self.max_cf_attempts = 3
        self.cf_timeout = 12  # seconds to wait for CF dictation
        self.impegnativa_timeout = 10  # seconds to wait for impegnativa
        self.cf_dictation_start = None
        self.cf_dictation_active = False
        
    def get_current_prompt(self):
        """Get the current prompt for the user"""
        if self.step == 1:
            if self.cf_attempts == 0:
                return "Fornisci il codice fiscale"
            else:
                return f"Ripeti il codice fiscale (tentativo {self.cf_attempts + 1}/{self.max_cf_attempts})"
        else:
            return "Fornisci il numero dell'impegnativa"
    
    def validate_cf(self, cf_code):
        """Validate CF format (16 characters, alphanumeric)"""
        if not cf_code or len(cf_code) != 16:
            return False
        
        # Check if it's alphanumeric
        if not cf_code.isalnum():
            return False
            
        return True
    
    def validate_impegnativa(self, number):
        """Validate impegnativa format (6 digits)"""
        if not number or len(number) != 6:
            return False
        return number.isdigit()
    
    def process_cf_result(self, transcription, cf_code):
        """Process CF step result"""
        self.cf_code = cf_code
        
        if self.validate_cf(cf_code):
            # CF is valid, move to step 2
            self.step = 2
            return {
                "status": "cf_valid",
                "cf_code": cf_code,
                "next_prompt": self.get_current_prompt(),
                "step": 2
            }
        else:
            # CF is invalid, retry or fail
            self.cf_attempts += 1
            
            if self.cf_attempts >= self.max_cf_attempts:
                return {
                    "status": "cf_failed",
                    "message": "Numero massimo di tentativi CF raggiunto",
                    "cf_code": cf_code,
                    "attempts": self.cf_attempts
                }
            else:
                return {
                    "status": "cf_retry",
                    "cf_code": cf_code,
                    "next_prompt": self.get_current_prompt(),
                    "attempts": self.cf_attempts,
                    "step": 1
                }
    
    def process_impegnativa_result(self, transcription, number):
        """Process impegnativa step result"""
        self.impegnativa = number
        
        if self.validate_impegnativa(number):
            # Both steps completed successfully
            return {
                "status": "complete",
                "cf_code": self.cf_code,
                "impegnativa": self.impegnativa,
                "message": "Entrambi i dati raccolti con successo"
            }
        else:
            # Impegnativa invalid, retry
            return {
                "status": "impegnativa_retry",
                "impegnativa": number,
                "next_prompt": "Ripeti il numero dell'impegnativa (6 cifre)",
                "step": 2
            }
    
    def get_timeout(self):
        """Get current step timeout"""
        if self.step == 1:
            return self.cf_timeout
        else:
            return self.impegnativa_timeout
    
    def is_complete(self):
        """Check if workflow is complete"""
        return self.step == 2 and self.cf_code and self.impegnativa
    
    def start_cf_dictation(self):
        """Start CF dictation timer"""
        self.cf_dictation_start = time.time()
        self.cf_dictation_active = True
        print(f"⏰ CF dictation started - {self.cf_timeout}s timeout")
    
    def check_cf_dictation_timeout(self):
        """Check if CF dictation should timeout"""
        if not self.cf_dictation_active or not self.cf_dictation_start:
            return False
        
        elapsed = time.time() - self.cf_dictation_start
        if elapsed >= self.cf_timeout:
            self.cf_dictation_active = False
            print(f"⏰ CF dictation timeout after {elapsed:.1f}s")
            return True
        return False
    
    def stop_cf_dictation(self):
        """Stop CF dictation"""
        self.cf_dictation_active = False
        if self.cf_dictation_start:
            elapsed = time.time() - self.cf_dictation_start
            print(f"⏹️ CF dictation stopped after {elapsed:.1f}s")
    
    def should_process_cf_now(self):
        """Check if we should process CF now (timeout or manual stop)"""
        return self.check_cf_dictation_timeout() or not self.cf_dictation_active

# Initialize parsers after class definitions
impegnativa_parser = ImpegnativaParser()

# Import voice player
try:
    from voice_player import voice_player
    VOICE_ENABLED = True  # Re-enable voice prompts with correct protocol
    print("🔊 Voice prompts enabled with correct call center protocol")
    print(f"🔊 Voice player object: {voice_player}")
except ImportError as e:
    VOICE_ENABLED = False
    print(f"⚠️ Voice prompts disabled (voice_player.py not found): {e}")

# Debug functions removed - only transcription files are saved

def analyze_audio_characteristics(pcm_array):
    """Analyze audio for processing decisions"""
    if len(pcm_array) < 1000:
        return {'needs_processing': False, 'snr_db': 30}
    
    try:
        audio_float = pcm_array.astype(np.float32) / 32768.0
        
        rms = np.sqrt(np.mean(audio_float**2))
        peak = np.max(np.abs(audio_float))
        
        noise_sample_size = max(100, len(audio_float) // 10)
        noise_floor = np.std(audio_float[:noise_sample_size])
        
        if noise_floor > 0:
            snr_estimate = 20 * np.log10(rms / noise_floor)
        else:
            snr_estimate = 30
        
        needs_processing = snr_estimate < 20 or peak < 0.1 or rms < 0.01
        
        print(f"🔊 Audio Analysis: RMS={rms:.4f}, Peak={peak:.4f}, SNR={snr_estimate:.1f}dB")
        
        return {
            'needs_processing': needs_processing,
            'snr_db': max(0, min(50, snr_estimate)),
            'rms': rms,
            'peak': peak
        }
    except:
        return {'needs_processing': True, 'snr_db': 15}

def technical_noise_reduction(audio_data, sr, analysis):
    """Advanced noise reduction without assumptions"""
    try:
        print("Applying technical signal processing...")
        
        nyquist = sr / 2
        low_cutoff = max(300 / nyquist, 0.01)
        high_cutoff = min(3400 / nyquist, 0.95)
        
        b, a = signal.butter(4, [low_cutoff, high_cutoff], btype='band')
        filtered_audio = signal.filtfilt(b, a, audio_data)
        
        if ENHANCED_AUDIO and analysis['needs_processing']:
            print("Applying adaptive spectral noise reduction...")
            
            reduction_strength = min(0.95, max(0.7, 1.0 - (analysis['snr_db'] / 30.0)))
            
            reduced_audio = nr.reduce_noise(
                y=filtered_audio,
                sr=sr,
                stationary=False,
                prop_decrease=reduction_strength
            )
            
            if len(reduced_audio) > WIENER_FILTER_SIZE:
                wiener_filtered = wiener(reduced_audio, WIENER_FILTER_SIZE)
            else:
                wiener_filtered = reduced_audio
                
            final_audio = wiener_filtered
            print(f"Advanced noise reduction applied (strength: {reduction_strength:.2f})")
            
        else:
            final_audio = filtered_audio
            print("Basic filtering applied")
        
        compressed_audio = np.tanh(final_audio * 2.0) / 2.0
        
        max_val = np.max(np.abs(compressed_audio))
        if max_val > 0:
            normalized_audio = compressed_audio / max_val * 0.8
        else:
            normalized_audio = compressed_audio
            
        print("Technical audio processing complete")
        return normalized_audio
        
    except Exception as e:
        print(f"Technical processing error: {e}")
        return audio_data / 32768.0

def alaw_to_linear_sample(a_val):
    """Custom A-law decoder EXACTLY matching call center server.js"""
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
    # FIXED: Match server.js logic exactly
    return -t if (a_val & 0x80) else t

def decode_alaw_custom(buffer):
    """Custom A-law decoder matching call center approach"""
    pcm = bytearray(len(buffer) * 2)
    for i in range(len(buffer)):
        sample = alaw_to_linear_sample(buffer[i])
        pcm[i * 2] = sample & 0xFF
        pcm[i * 2 + 1] = (sample >> 8) & 0xFF
    return bytes(pcm)

def parse_audio_frames(audio_data):
    """Parse audio frames EXACTLY like call center server.js"""
    buffer_acc = bytearray(audio_data)
    valid_frames = []
    
    while len(buffer_acc) >= 3:
        # EXACT match to server.js parsing
        frame_type = buffer_acc[0]
        frame_length = int.from_bytes(buffer_acc[1:3], 'big')  # readUInt16BE
        
        if len(buffer_acc) < 3 + frame_length:
            break
            
        payload = buffer_acc[3:3 + frame_length]
        buffer_acc = buffer_acc[3 + frame_length:]
        
        # EXACT match: Only process type 0x10 frames
        if frame_type == 0x10:
            valid_frames.append(payload)
            print(f"✅ Frame type 0x10, length: {frame_length}")
        else:
            print(f"⚠️ Ignoring frame type 0x{frame_type:02x}")
    
    # If no valid frames found, treat as raw audio (fallback)
    if not valid_frames:
        print("⚠️ No valid frames found, using raw audio")
        return [audio_data]
    
    return valid_frames

def process_slin_audio(audio_data, output_file):
    """Process audio using call center's EXACT approach for quality"""
    try:
        print("Processing audio with call center's quality approach...")
        
        # Step 1: Parse frames like call center does
        print("🔍 Parsing audio frames...")
        valid_frames = parse_audio_frames(audio_data)
        print(f"✅ Found {len(valid_frames)} valid audio frames")
        
        # Step 2: Process frames with custom A-law decoder
        all_pcm = bytearray()
        for frame in valid_frames:
            frame_pcm = decode_alaw_custom(frame)
            all_pcm.extend(frame_pcm)
        
        print("✅ Custom A-law decoder applied")
        
        # Step 3: Apply call center's FFMPEG processing for quality
        try:
            import tempfile
            import subprocess
            
            # Save raw PCM for FFMPEG processing
            with tempfile.NamedTemporaryFile(suffix='.raw', delete=False) as temp_file:
                temp_file.write(all_pcm)
                temp_raw = temp_file.name
            
            # Apply call center's EXACT FFMPEG command for quality
            temp_processed = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_processed.close()
            
            # EXACT match to server.js FFMPEG command
            cmd = [
                'ffmpeg', '-y',
                '-f', 's16le', '-ar', '8000', '-ac', '1', '-i', temp_raw,
                '-filter_complex', '[0:a]highpass=f=200,lowpass=f=3400,dynaudnorm[a]',
                '-map', '[a]', '-ac', '1', '-ar', '16000', '-b:a', '192k', temp_processed.name
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ FFMPEG quality processing applied (dynaudnorm)")
                # Copy the processed file to our output
                import shutil
                shutil.copy2(temp_processed.name, output_file)
                os.unlink(temp_raw)
                os.unlink(temp_processed.name)
                return True, "A-law (call center quality)"
            else:
                print(f"⚠️ FFMPEG failed, using fallback: {result.stderr}")
                os.unlink(temp_raw)
                os.unlink(temp_processed.name)
                
        except Exception as e:
            print(f"⚠️ FFMPEG processing failed: {e}")
        
        # Step 4: Fallback to high-quality processing without resampling
        print("🔄 Using high-quality fallback processing...")
        
        # Convert to float for processing
        pcm_array = np.frombuffer(all_pcm, dtype=np.int16)
        processed_audio = pcm_array.astype(np.float32) / 32768.0
        
        # Apply quality filters (matching call center approach)
        from scipy.signal import butter, filtfilt
        
        # High-pass filter (200Hz) - remove low-frequency noise
        nyquist = 8000 / 2
        high_cutoff = 200 / nyquist
        b, a = butter(4, high_cutoff, btype='high')
        highpassed = filtfilt(b, a, processed_audio)
        
        # Low-pass filter (3400Hz) - remove high-frequency noise
        low_cutoff = 3400 / nyquist
        b, a = butter(4, low_cutoff, btype='low')
        filtered = filtfilt(b, a, highpassed)
        
        # Resample to 16kHz for Whisper (gentle resampling)
        num_samples = int(len(filtered) * (SAMPLE_RATE_TARGET / SAMPLE_RATE_ORIGINAL))
        resampled_audio = resample(filtered, num_samples)
        
        # Convert back to int16
        resampled_int16 = (resampled_audio * 32767).astype(np.int16)
        
        # Save with quality settings
        with wave.open(output_file, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE_TARGET)
            wf.writeframes(resampled_int16.tobytes())

        print(f"✅ High-quality processing complete: {output_file}")
        print(f"📊 Format: A-law (call center quality approach)")
        return True, "A-law (call center quality)"
        
    except Exception as e:
        print(f"❌ Quality processing failed: {e}")
        return False

def transcribe_audio(file_path):
    """Transcribe using Whisper - called by background workers"""
    try:
        result = model.transcribe(file_path, fp16=False, language="it")
        transcription = result["text"].strip()
        print(f"📝 Transcription: {transcription}")
        return transcription
    except Exception as e:
        print(f"Transcription error: {e}")
        return "Errore durante la trascrizione."

def transcription_worker():
    """Background worker that processes transcription queue with two-step workflow"""
    while True:
        try:
            # Get next transcription task from queue
            task = transcription_queue.get()
            if task is None:  # Shutdown signal
                break
                
            audio_file, client_id, call_start_time = task
            print(f"🔄 Processing transcription for client {client_id}")
            
            # Transcribe the audio
            transcription = transcribe_audio(audio_file)
            
            # Get or create workflow for client
            if client_id not in client_workflows:
                client_workflows[client_id] = CallCenterWorkflow()
                print(f"📋 Started workflow for client {client_id}: {client_workflows[client_id].get_current_prompt()}")
            
            workflow = client_workflows[client_id]
            
            # Process based on workflow step
            print(f"🔍 Processing transcription for client {client_id}, current step: {workflow.step}")
            if workflow.step == 1:
                # Step 1: Process CF
                response = _process_cf_step(client_id, transcription, call_start_time, workflow)
                print(f"🔍 CF step result: {response}")
            else:
                # Step 2: Process Impegnativa
                response = _process_impegnativa_step(client_id, transcription, call_start_time, workflow)
                print(f"🔍 Impegnativa step result: {response}")
            
            # Send response to client
            _send_workflow_response(client_id, response)
            
            # Mark task as done
            transcription_queue.task_done()
            
        except Exception as e:
            print(f"❌ Transcription worker error: {e}")
            import traceback
            traceback.print_exc()

def _process_cf_step(client_id, transcription, call_start_time, workflow):
    """Process CF step"""
    cf_parser = TechnicalCFParser()
    cf_result = cf_parser.parse_cf(transcription)
    
    # Process through workflow
    result = workflow.process_cf_result(transcription, cf_result["cf_code"])
    
    # Add timing and metadata
    result.update({
        "transcription": transcription,
        "call_duration": time.time() - call_start_time,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "client_id": client_id,
        "step": workflow.step
    })
    
    return result

def _process_impegnativa_step(client_id, transcription, call_start_time, workflow):
    """Process impegnativa step"""
    # Parse impegnativa from transcription
    impegnativa_result = impegnativa_parser.parse_impegnativa(transcription)
    
    # Process through workflow
    result = workflow.process_impegnativa_result(transcription, impegnativa_result['number'])
    
    # Add timing and metadata
    result.update({
        "transcription": transcription,
        "call_duration": time.time() - call_start_time,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "client_id": client_id,
        "step": workflow.step
    })
    
    return result

def _send_workflow_response(client_id, response):
    """Send workflow response to client with voice prompts"""
    with response_lock:
        if client_id in client_responses:
            try:
                client_socket = client_responses[client_id]
                
                # Send voice prompt based on workflow status
                if VOICE_ENABLED:
                    print(f"🔊 VOICE_ENABLED is True, sending voice prompt...")
                    try:
                        _send_voice_prompt_for_response(client_socket, response)
                        print(f"🔊 Voice prompt sent, skipping JSON response")
                        # Don't send JSON response when using voice prompts
                        # The client expects only audio, not JSON
                    except Exception as e:
                        print(f"⚠️ Voice prompt failed, sending text prompt: {e}")
                        _send_text_prompt_for_response(client_socket, response)
                        # Send JSON response only if voice prompt failed
                        response_json = json.dumps(response, ensure_ascii=False, indent=2)
                        client_socket.sendall(response_json.encode("utf-8"))
                else:
                    print(f"⚠️ VOICE_ENABLED is False, sending text prompt")
                    _send_text_prompt_for_response(client_socket, response)
                    # Send JSON response when using text prompts
                    response_json = json.dumps(response, ensure_ascii=False, indent=2)
                    client_socket.sendall(response_json.encode("utf-8"))
                
                # Only close connection if workflow is complete or failed
                status = response.get('status')
                if status in ['complete', 'cf_failed']:
                    # Workflow finished - close connection and cleanup
                    client_socket.close()
                    del client_responses[client_id]
                    if client_id in client_workflows:
                        del client_workflows[client_id]
                    print(f"✅ Workflow completed for client {client_id}")
                else:
                    # Workflow continues - keep connection open
                    print(f"⏳ Workflow step {response.get('step', 1)} for client {client_id} - connection kept open for next step")
                    print(f"🔍 Client should continue listening for audio input...")
                    # Remove client from responses so main workflow loop can continue
                    del client_responses[client_id]
                    print(f"🔍 Removed client {client_id} from responses - main loop can continue")
                    
            except Exception as e:
                print(f"❌ Failed to send workflow response to client {client_id}: {e}")
                if client_id in client_responses:
                    del client_responses[client_id]
                if client_id in client_workflows:
                    del client_workflows[client_id]

def _send_voice_prompt_for_response(client_socket, response):
    """Send appropriate voice prompt based on response status"""
    try:
        status = response.get('status')
        print(f"🔊 Sending voice prompt for status: {status}")
        
        if status == 'cf_valid':
            voice_player.play_workflow_prompt(client_socket, "impegnativa_request")
            print(f"🔊 Sent impegnativa_request prompt")
        elif status == 'cf_retry':
            attempts = response.get('attempts', 0)
            voice_player.play_workflow_prompt(client_socket, "cf_retry", attempts)
            print(f"🔊 Sent cf_retry prompt (attempt {attempts})")
        elif status == 'cf_failed':
            voice_player.play_workflow_prompt(client_socket, "cf_failed")
            print(f"🔊 Sent cf_failed prompt")
        elif status == 'impegnativa_retry':
            voice_player.play_workflow_prompt(client_socket, "impegnativa_retry")
            print(f"🔊 Sent impegnativa_retry prompt")
        elif status == 'complete':
            voice_player.play_workflow_prompt(client_socket, "success")
            print(f"🔊 Sent success prompt")
        else:
            voice_player.play_workflow_prompt(client_socket, "error")
            print(f"🔊 Sent error prompt")
            
    except Exception as e:
        print(f"⚠️ Failed to send voice prompt: {e}")

def _send_text_prompt_for_response(client_socket, response):
    """Send text prompt based on response status"""
    try:
        status = response.get('status')
        print(f"📝 Sending text prompt for status: {status}")
        
        if status == 'cf_valid':
            prompt = "Perfetto! Ora fornisci il numero dell'impegnativa (6 cifre)."
        elif status == 'cf_retry':
            attempts = response.get('attempts', 0)
            prompt = f"Codice fiscale non valido. Ripeti il codice fiscale (tentativo {attempts + 1}/3)."
        elif status == 'cf_failed':
            prompt = "Numero massimo di tentativi raggiunto. Chiamata terminata."
        elif status == 'impegnativa_retry':
            prompt = "Numero impegnativa non valido. Ripeti il numero dell'impegnativa (6 cifre)."
        elif status == 'complete':
            prompt = "Perfetto! Entrambi i dati sono stati raccolti con successo. Chiamata completata."
        else:
            prompt = "Si è verificato un errore. Riprova."
        
        # Send text prompt as simple message
        prompt_data = f"PROMPT:{prompt}".encode("utf-8")
        print(f"📝 Sending text prompt data: {prompt_data}")
        client_socket.sendall(prompt_data)
        print(f"📝 Text prompt sent successfully: {prompt}")
        
    except Exception as e:
        print(f"⚠️ Failed to send text prompt: {e}")

def _continue_listening_for_step2(client_socket, client_id, call_start_time):
    """Continue listening for step 2 (impegnativa) after step 1 is complete"""
    try:
        workflow = client_workflows[client_id]
        print(f"🔄 Continuing workflow for client {client_id} - step {workflow.step}")
        
        audio_data = b""
        last_data_time = time.time()
        client_socket.settimeout(CLIENT_TIMEOUT)
        
        timeout_duration = workflow.get_timeout()
        print(f"Receiving audio data for step {workflow.step} (timeout: {timeout_duration}s)...")
        
        while True:
            current_time = time.time()
            call_duration = current_time - call_start_time
            
            if call_duration > MAX_CALL_DURATION:
                print(f"⏰ Maximum duration reached")
                break
            
            # Check timeout for step 2
            if current_time - last_data_time > timeout_duration:
                print(f"⏰ Step 2 timeout - processing {len(audio_data)} bytes")
                break
            
            try:
                data = client_socket.recv(4096)
                
                if not data:
                    print(f"Client disconnected - processing {len(audio_data)} bytes")
                    break
                    
                if data == b"END":
                    print("END signal received")
                    break
                
                audio_data += data
                last_data_time = current_time
                
                if len(audio_data) % 16384 == 0:
                    estimated_samples = len(audio_data) // 2
                    duration = estimated_samples / SAMPLE_RATE_ORIGINAL
                    print(f"Received ~{duration:.1f}s audio ({len(audio_data)} bytes) - step {workflow.step}")
                
            except socket.timeout:
                time_since_data = current_time - last_data_time
                
                if time_since_data > timeout_duration:
                    print(f"⏰ Step 2 timeout - processing {len(audio_data)} bytes")
                    break
                elif time_since_data > CF_DICTATION_TIME:
                    print(f"⏰ Processing timeout - using {len(audio_data)} bytes")
                    break
                elif len(audio_data) == 0:
                    print(f"⏰ No initial data timeout")
                    break
        
        if len(audio_data) > 0:
            print(f"Processing {len(audio_data)} bytes for step {workflow.step}...")
            
            output_file = get_transcription_filename(client_id)
            success, format_detected = process_slin_audio(audio_data, output_file)
            
            if success:
                print(f"✅ Audio processed successfully for client {client_id} (step {workflow.step})")
                
                # Store client socket for response
                with response_lock:
                    client_responses[client_id] = client_socket
                
                # Add to transcription queue (FIFO - First In, First Out)
                transcription_queue.put((output_file, client_id, call_start_time))
                
                print(f"⏳ Client {client_id} queued for transcription (step {workflow.step})")
                print(f"📊 Queue status: {transcription_queue.qsize()} clients waiting")
                
                # Wait for final transcription to complete
                while client_id in client_responses:
                    time.sleep(0.1)
                
            else:
                print(f"❌ Audio processing failed for client {client_id}")
                error_response = {
                    "status": "error",
                    "message": "Audio processing failed",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "client_id": client_id,
                    "step": workflow.step
                }
                client_socket.sendall(json.dumps(error_response).encode("utf-8"))
        else:
            print("No audio data received for step 2")
            
    except Exception as e:
        print(f"Error in step 2 continuation: {e}")
        import traceback
        traceback.print_exc()

def handle_technical_client(client_socket, client_address):
    """Technical client handling with professional queuing and voice prompts"""
    client_id = str(uuid.uuid4())[:8]  # Short unique ID
    print(f"📞 CALL CENTER CONNECTION: {client_address} (ID: {client_id})")
    call_start_time = time.time()
    
    # Send welcome message
    if VOICE_ENABLED:
        try:
            print(f"🔊 Attempting to send welcome prompt to client {client_id}")
            voice_player.play_workflow_prompt(client_socket, "start")
            print(f"🔊 Welcome prompt sent to client {client_id}")
        except Exception as e:
            print(f"⚠️ Failed to send welcome prompt: {e}")
            # Fallback to text prompt
            try:
                welcome_text = "PROMPT:Benvenuto! Fornisci il tuo codice fiscale (16 caratteri)."
                client_socket.sendall(welcome_text.encode("utf-8"))
                print(f"📝 Welcome text prompt sent to client {client_id}")
            except Exception as e2:
                print(f"⚠️ Failed to send welcome text prompt: {e2}")
    else:
        print(f"⚠️ Voice prompts disabled - skipping welcome message")
        print(f"🔍 Client {client_id} connected, waiting for audio input...")
        print(f"🔍 Socket info: {client_socket.getsockname()} -> {client_socket.getpeername()}")
    
    # Initialize workflow for client
    if client_id not in client_workflows:
        client_workflows[client_id] = CallCenterWorkflow()
        print(f"📋 Started workflow for client {client_id}")
    
    workflow = client_workflows[client_id]
    print(f"🔍 Workflow initialized for client {client_id}, step: {workflow.step}")
    
    try:
        # Handle the complete workflow in a loop
        while not workflow.is_complete():
            
            audio_data = b""
            last_data_time = time.time()
            client_socket.settimeout(CLIENT_TIMEOUT)
            
            current_step = workflow.step
            timeout_duration = workflow.get_timeout()
            
            # Start CF dictation timer for step 1
            if current_step == 1:
                workflow.start_cf_dictation()
            
            print(f"🔍 Starting audio collection for step {current_step} (timeout: {timeout_duration}s)...")
            
            while True:
                current_time = time.time()
                call_duration = current_time - call_start_time
                
                if call_duration > MAX_CALL_DURATION:
                    print(f"⏰ Maximum duration reached")
                    break
                
                # Check timeout for current step
                if current_step == 1 and workflow.should_process_cf_now():
                    print(f"⏰ CF dictation timeout - processing {len(audio_data)} bytes")
                    break
                elif current_step == 2 and current_time - last_data_time > timeout_duration:
                    print(f"⏰ Step 2 timeout - processing {len(audio_data)} bytes")
                    break
                
                try:
                    data = client_socket.recv(4096)
                    
                    if not data:
                        print(f"⚠️ Client disconnected - processing {len(audio_data)} bytes")
                        print(f"🔍 Client {client_id} disconnected during step {current_step}")
                        print(f"🔍 Audio data received so far: {len(audio_data)} bytes")
                        print(f"🔍 Workflow state: step={workflow.step}, cf_attempts={workflow.cf_attempts}")
                        break
                        
                    if data == b"END":
                        print("END signal received")
                        if current_step == 1:
                            workflow.stop_cf_dictation()
                        break
                    
                    audio_data += data
                    last_data_time = current_time
                    
                    if len(audio_data) % 16384 == 0:
                        estimated_samples = len(audio_data) // 2
                        duration = estimated_samples / SAMPLE_RATE_ORIGINAL
                        print(f"Received ~{duration:.1f}s audio ({len(audio_data)} bytes) - step {current_step}")
                    
                except socket.timeout:
                    time_since_data = current_time - last_data_time
                    
                    # Check if we should process based on workflow step
                    if current_step == 1 and workflow.should_process_cf_now():
                        print(f"⏰ CF dictation timeout - processing {len(audio_data)} bytes")
                        break
                    elif current_step == 2 and time_since_data > timeout_duration:
                        print(f"⏰ Step 2 timeout - processing {len(audio_data)} bytes")
                        break
                    elif time_since_data > CF_DICTATION_TIME:
                        print(f"⏰ Processing timeout - using {len(audio_data)} bytes")
                        break
                    elif len(audio_data) == 0:
                        print(f"⏰ No initial data timeout")
                        break

            if len(audio_data) > 0:
                print(f"Processing {len(audio_data)} bytes for step {current_step}...")
                
                output_file = get_transcription_filename(client_id)
                success, format_detected = process_slin_audio(audio_data, output_file)
                
                if success:
                    print(f"✅ Audio processed successfully for client {client_id} (step {current_step})")
                    
                    # Store client socket for response
                    with response_lock:
                        client_responses[client_id] = client_socket
                    
                    # Add to transcription queue (FIFO - First In, First Out)
                    transcription_queue.put((output_file, client_id, call_start_time))
                    
                    print(f"⏳ Client {client_id} queued for transcription (step {current_step})")
                    print(f"📊 Queue status: {transcription_queue.qsize()} clients waiting")
                    
                    # Wait for transcription to complete before continuing
                    # This ensures the workflow progresses step by step
                    while client_id in client_responses:
                        time.sleep(0.1)
                    
                    # Debug workflow state
                    print(f"🔍 Workflow state after transcription: step={workflow.step}, cf_attempts={workflow.cf_attempts}, cf_code='{workflow.cf_code}', impegnativa='{workflow.impegnativa}'")
                    
                    # Check if workflow is complete or failed
                    if workflow.is_complete():
                        print(f"✅ Workflow completed successfully for client {client_id}")
                        break
                    elif workflow.cf_attempts >= workflow.max_cf_attempts:
                        print(f"❌ Workflow failed - max CF attempts reached for client {client_id}")
                        break
                    else:
                        print(f"🔄 Workflow continuing to step {workflow.step} for client {client_id}")
                        print(f"🔍 Continuing to listen for next audio input...")
                        print(f"🔍 Workflow is_complete(): {workflow.is_complete()}")
                        print(f"🔍 CF attempts: {workflow.cf_attempts}/{workflow.max_cf_attempts}")
                        print(f"🔍 About to continue to main workflow loop...")
                        
                        # Restart CF dictation timer for next attempt
                        if workflow.step == 1:
                            workflow.start_cf_dictation()
                            print(f"🔍 Restarted CF dictation timer for next attempt")
                        
                        # Continue to next iteration of the outer workflow loop
                        print(f"🔍 Continuing to next workflow iteration...")
                        continue
                        
                else:
                    print(f"❌ Audio processing failed for client {client_id}")
                    error_response = {
                        "status": "error",
                        "message": "Audio processing failed",
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "client_id": client_id,
                        "step": current_step
                    }
                    client_socket.sendall(json.dumps(error_response).encode("utf-8"))
                    break
            else:
                print("No audio data received")
                break
        
        # This point should not be reached if the workflow is continuing properly
        print(f"🔍 Workflow ended - this should not happen if workflow is continuing")

    except Exception as e:
        print(f"Technical client error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        try:
            client_socket.close()
        except:
            pass
        
        total_time = time.time() - call_start_time
        print(f"🔚 Call completed for client {client_id}: {total_time:.1f}s")
        print("=" * 50)

def start_technical_server():
    """Start technical server with auto-format detection"""
    # Setup directories and cleanup
    setup_directories()
    cleanup_old_transcription_files()
    
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((HOST, PORT))
            server_socket.listen(1000)  # Allow 1000 pending connections for large call center
            
            # Start background transcription workers
            print("🚀 Starting professional call center queuing system...")
            num_workers = 1  # Start with 1 worker (can be increased for multiple GPUs)
            for i in range(num_workers):
                worker = threading.Thread(target=transcription_worker, daemon=True)
                worker.start()
                transcription_workers.append(worker)
                print(f"✅ Transcription worker {i+1} started")
            
            print("PROFESSIONAL CALL CENTER TWO-STEP WORKFLOW SERVER")
            print("=" * 60)
            print(f"📍 Listening: {HOST}:{PORT}")
            print(f"👥 Max connections: 1000 (unlimited callers)")
            print(f"🔄 Transcription workers: {num_workers}")
            print(f"📋 Queue system: FIFO (First In, First Out)")
            print(f"🎯 WORKFLOW: Step 1 = CF (16 chars), Step 2 = Impegnativa (6 digits)")
            print(f"⏱️ TIMING: CF=12s dictation timeout, Impegnativa=10s timeout")
            print(f"🔄 RETRIES: CF max 3 attempts, Impegnativa unlimited")
            print(f"🎯 FORMAT: A-law with call center quality processing")
            print(f"🔧 FRAME PROTOCOL: 3-byte headers with type 0x10 frames")
            print(f"🎵 CUSTOM DECODER: Call center's exact A-law decoder")
            print(f"🎛️ FFMPEG FILTERS: highpass=200, lowpass=3400, dynaudnorm")
            print(f"📊 Sample rate: 8kHz → 16kHz (quality resampling)")
            print(f"🧹 AUTO-CLEANUP: Removes old transcription files")
            print(f"📁 Directory: {TRANSCRIPTION_DIR}/ (transcription files only)")
            print(f"🔤 CF Parser: Simple first-letter extraction (no mapping)")
            print(f"🔢 Impegnativa Parser: 6-digit number extraction")
            
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                print(f"🎮 GPU: {gpu_name}")
            
            print("✅ Ready for unlimited call center connections...")
            print("=" * 60)

            while True:
                try:
                    client_socket, client_address = server_socket.accept()
                    # Handle each client in a separate thread for unlimited connections
                    client_thread = threading.Thread(
                        target=handle_technical_client, 
                        args=(client_socket, client_address),
                        daemon=True
                    )
                    client_thread.start()
                    print(f"📞 New client thread started for {client_address}")
                except KeyboardInterrupt:
                    print("\n🛑 Server shutdown")
                    break
                except Exception as e:
                    print(f"Server error: {e}")
                    import traceback
                    traceback.print_exc()
                    
    except Exception as e:
        print(f"Server startup error: {e}")
    finally:
        print("🏁 Multi-format server shutdown")

if __name__ == "__main__":
    start_technical_server()