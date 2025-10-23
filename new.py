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
except ImportError:
    ENHANCED_AUDIO = False

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

model = whisper.load_model("large")

transcription_queue = queue.Queue()
transcription_workers = []
client_responses = {}
response_lock = threading.Lock()

client_workflows = {}

TRANSCRIPTION_DIR = "transcription_files"
MAX_FILES_PER_DIR = 100
CLEANUP_AGE_DAYS = 3

CF_RESULTS_LOG = "cf_results.jsonl"
VM_INGEST_URL = os.environ.get("VM_INGEST_URL")  # e.g., http://10.10.13.122:38473/api/ingest
VM_INGEST_TOKEN = os.environ.get("VM_INGEST_TOKEN")  # shared secret, optional

def setup_directories():
    os.makedirs(TRANSCRIPTION_DIR, exist_ok=True)

def cleanup_old_files(directory, max_files=MAX_FILES_PER_DIR, max_age_days=CLEANUP_AGE_DAYS):
    if not os.path.exists(directory):
        return
    
    try:
        files = []
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                mtime = os.path.getmtime(filepath)
                files.append((filepath, mtime))
        
        files.sort(key=lambda x: x[1], reverse=True)
        
        cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
        removed_old = 0
        for filepath, mtime in files:
            if mtime < cutoff_time:
                os.remove(filepath)
                removed_old += 1
        
        removed_excess = 0
        if len(files) - removed_old > max_files:
            for filepath, mtime in files[max_files:]:
                if os.path.exists(filepath):
                    os.remove(filepath)
                    removed_excess += 1
        
    except Exception as e:
        pass

def get_transcription_filename(client_id):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(TRANSCRIPTION_DIR, f"transcription_{client_id}_{timestamp}.wav")

def cleanup_old_transcription_files():
    try:
        cleanup_old_files(TRANSCRIPTION_DIR, MAX_FILES_PER_DIR, CLEANUP_AGE_DAYS)
    except Exception as e:
        pass

def append_cf_result(result):
    try:
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
        pass

def post_cf_result_to_vm(result):
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
                    return
        except urllib.error.HTTPError as e:
            last_err = e
        except Exception as e:
            last_err = e
        time.sleep(0.75 * attempt)

class TechnicalCFParser:
    
    def __init__(self):
        pass
    
    def parse_cf(self, transcription):
        
        words = transcription.replace(',', ' ').replace('-', ' ').replace('.', ' ').split()
        
        cf_parts = []
        
        for word in words:
            word = word.strip()
            
            if word.isdigit():
                cf_parts.append(word)
                continue
            
            if word.isalpha() and len(word) <= 8 and word.isupper():
                for char in word:
                    cf_parts.append(char)
                continue
            
            if word.isalpha():
                cf_parts.append(word[0].upper())
                continue
            
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
    
    def __init__(self):
        pass
    
    def parse_impegnativa(self, transcription):
        import re
        
        text = transcription.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        words = text.split()
        
        for word in words:
            if word.isdigit() and len(word) == 6:
                return {
                    'number': word,
                    'length': len(word),
                    'is_valid': True,
                    'confidence': 1.0
                }
        
        digits = []
        for word in words:
            if word.isdigit() and len(word) == 1:
                digits.append(word)
        
        if len(digits) >= 6:
            number = ''.join(digits[:6])
            return {
                'number': number,
                'length': len(number),
                'is_valid': True,
                'confidence': 0.8
            }
        
        return {
            'number': '',
            'length': 0,
            'is_valid': False,
            'confidence': 0.0
        }

class CallCenterWorkflow:
    
    def __init__(self):
        self.step = 1
        self.cf_code = ""
        self.impegnativa = ""
        self.cf_attempts = 0
        self.max_cf_attempts = 3
        self.cf_timeout = 12
        self.impegnativa_timeout = 10
        self.cf_dictation_start = None
        self.cf_dictation_active = False
        self.impegnativa_dictation_start = None
        self.impegnativa_dictation_active = False
        
    def get_current_prompt(self):
        if self.step == 1:
            if self.cf_attempts == 0:
                return "Fornisci il codice fiscale"
            else:
                return f"Ripeti il codice fiscale (tentativo {self.cf_attempts + 1}/{self.max_cf_attempts})"
        else:
            return "Fornisci il numero dell'impegnativa"
    
    def validate_cf(self, cf_code):
        if not cf_code or len(cf_code) != 16:
            return False
        if not cf_code.isalnum():
            return False
        return True
    
    def validate_impegnativa(self, number):
        if not number or len(number) != 6:
            return False
        return number.isdigit()
    
    def process_cf_result(self, transcription, cf_code):
        self.cf_code = cf_code
        
        if self.validate_cf(cf_code):
            self.step = 2
            self.start_impegnativa_dictation()
            return {
                "status": "cf_valid",
                "cf_code": cf_code,
                "next_prompt": self.get_current_prompt(),
                "step": 2
            }
        else:
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
        self.impegnativa = number
        
        if self.validate_impegnativa(number):
            return {
                "status": "complete",
                "cf_code": self.cf_code,
                "impegnativa": self.impegnativa,
                "message": "Entrambi i dati raccolti con successo"
            }
        else:
            return {
                "status": "impegnativa_retry",
                "impegnativa": number,
                "next_prompt": "Ripeti il numero dell'impegnativa (6 cifre)",
                "step": 2
            }
    
    def get_timeout(self):
        if self.step == 1:
            return self.cf_timeout
        else:
            return self.impegnativa_timeout
    
    def is_complete(self):
        return self.step == 2 and self.cf_code and self.impegnativa
    
    def start_cf_dictation(self):
        self.cf_dictation_start = time.time()
        self.cf_dictation_active = True
    
    def start_impegnativa_dictation(self):
        self.impegnativa_dictation_start = time.time()
        self.impegnativa_dictation_active = True
    
    def check_cf_dictation_timeout(self):
        if not self.cf_dictation_active or not self.cf_dictation_start:
            return False
        
        elapsed = time.time() - self.cf_dictation_start
        if elapsed >= self.cf_timeout:
            self.cf_dictation_active = False
            return True
        return False
    
    def check_impegnativa_dictation_timeout(self):
        if not self.impegnativa_dictation_active or not self.impegnativa_dictation_start:
            return False
        
        elapsed = time.time() - self.impegnativa_dictation_start
        if elapsed >= self.impegnativa_timeout:
            self.impegnativa_dictation_active = False
            return True
        
        return False
    
    def stop_cf_dictation(self):
        self.cf_dictation_active = False
    
    def should_process_cf_now(self):
        return self.check_cf_dictation_timeout() or not self.cf_dictation_active

impegnativa_parser = ImpegnativaParser()
try:
    from voice_player import voice_player
    VOICE_ENABLED = True
except ImportError as e:
    VOICE_ENABLED = False

def analyze_audio_characteristics(pcm_array):
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
        
        return {
            'needs_processing': needs_processing,
            'snr_db': max(0, min(50, snr_estimate)),
            'rms': rms,
            'peak': peak
        }
    except:
        return {'needs_processing': True, 'snr_db': 15}

def technical_noise_reduction(audio_data, sr, analysis):
    try:
        
        nyquist = sr / 2
        low_cutoff = max(300 / nyquist, 0.01)
        high_cutoff = min(3400 / nyquist, 0.95)
        
        b, a = signal.butter(4, [low_cutoff, high_cutoff], btype='band')
        filtered_audio = signal.filtfilt(b, a, audio_data)
        
        if ENHANCED_AUDIO and analysis['needs_processing']:
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
            
        else:
            final_audio = filtered_audio
        
        compressed_audio = np.tanh(final_audio * 2.0) / 2.0
        
        max_val = np.max(np.abs(compressed_audio))
        if max_val > 0:
            normalized_audio = compressed_audio / max_val * 0.8
        else:
            normalized_audio = compressed_audio
            
        return normalized_audio
        
    except Exception as e:
        return audio_data / 32768.0

def alaw_to_linear_sample(a_val):
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
    return -t if (a_val & 0x80) else t

def decode_alaw_custom(buffer):
    pcm = bytearray(len(buffer) * 2)
    for i in range(len(buffer)):
        sample = alaw_to_linear_sample(buffer[i])
        pcm[i * 2] = sample & 0xFF
        pcm[i * 2 + 1] = (sample >> 8) & 0xFF
    return bytes(pcm)

def parse_audio_frames(audio_data):
    buffer_acc = bytearray(audio_data)
    valid_frames = []
    
    while len(buffer_acc) >= 3:
        frame_type = buffer_acc[0]
        frame_length = int.from_bytes(buffer_acc[1:3], 'big')
        
        if len(buffer_acc) < 3 + frame_length:
            break
            
        payload = buffer_acc[3:3 + frame_length]
        buffer_acc = buffer_acc[3 + frame_length:]
        
        if frame_type == 0x10:
            valid_frames.append(payload)
    
    if not valid_frames:
        return [audio_data]
    
    return valid_frames

def process_slin_audio(audio_data, output_file):
    try:
        valid_frames = parse_audio_frames(audio_data)
        
        all_pcm = bytearray()
        for frame in valid_frames:
            frame_pcm = decode_alaw_custom(frame)
            all_pcm.extend(frame_pcm)
        
        try:
            import tempfile
            import subprocess
            
            with tempfile.NamedTemporaryFile(suffix='.raw', delete=False) as temp_file:
                temp_file.write(all_pcm)
                temp_raw = temp_file.name
            
            temp_processed = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_processed.close()
            
            cmd = [
                'ffmpeg', '-y',
                '-f', 's16le', '-ar', '8000', '-ac', '1', '-i', temp_raw,
                '-filter_complex', '[0:a]highpass=f=200,lowpass=f=3400,dynaudnorm[a]',
                '-map', '[a]', '-ac', '1', '-ar', '16000', '-b:a', '192k', temp_processed.name
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                import shutil
                shutil.copy2(temp_processed.name, output_file)
                os.unlink(temp_raw)
                os.unlink(temp_processed.name)
                return True, "A-law (call center quality)"
            else:
                os.unlink(temp_raw)
                os.unlink(temp_processed.name)
                
        except Exception as e:
            pass
        
        pcm_array = np.frombuffer(all_pcm, dtype=np.int16)
        processed_audio = pcm_array.astype(np.float32) / 32768.0
        
        from scipy.signal import butter, filtfilt
        
        nyquist = 8000 / 2
        high_cutoff = 200 / nyquist
        b, a = butter(4, high_cutoff, btype='high')
        highpassed = filtfilt(b, a, processed_audio)
        
        low_cutoff = 3400 / nyquist
        b, a = butter(4, low_cutoff, btype='low')
        filtered = filtfilt(b, a, highpassed)
        
        num_samples = int(len(filtered) * (SAMPLE_RATE_TARGET / SAMPLE_RATE_ORIGINAL))
        resampled_audio = resample(filtered, num_samples)
        
        resampled_int16 = (resampled_audio * 32767).astype(np.int16)
        
        with wave.open(output_file, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE_TARGET)
            wf.writeframes(resampled_int16.tobytes())

        return True, "A-law (call center quality)"
        
    except Exception as e:
        return False

def transcribe_audio(file_path):
    try:
        result = model.transcribe(file_path, fp16=False, language="it")
        transcription = result["text"].strip()
        return transcription
    except Exception as e:
        return "Errore durante la trascrizione."

def transcription_worker():
    while True:
        try:
            task = transcription_queue.get()
            if task is None:
                break
                
            audio_file, client_id, call_start_time = task
            
            transcription = transcribe_audio(audio_file)
            
            if client_id not in client_workflows:
                client_workflows[client_id] = CallCenterWorkflow()
            
            workflow = client_workflows[client_id]
            
            if workflow.step == 1:
                response = _process_cf_step(client_id, transcription, call_start_time, workflow)
            else:
                response = _process_impegnativa_step(client_id, transcription, call_start_time, workflow)
            
            _send_workflow_response(client_id, response)
            
            transcription_queue.task_done()
            
        except Exception as e:
            pass

def _process_cf_step(client_id, transcription, call_start_time, workflow):
    cf_parser = TechnicalCFParser()
    cf_result = cf_parser.parse_cf(transcription)
    
    result = workflow.process_cf_result(transcription, cf_result["cf_code"])
    
    result.update({
        "transcription": transcription,
        "call_duration": time.time() - call_start_time,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "client_id": client_id,
        "step": workflow.step
    })
    
    return result

def _process_impegnativa_step(client_id, transcription, call_start_time, workflow):
    impegnativa_result = impegnativa_parser.parse_impegnativa(transcription)
    
    result = workflow.process_impegnativa_result(transcription, impegnativa_result['number'])
    
    result.update({
        "transcription": transcription,
        "call_duration": time.time() - call_start_time,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "client_id": client_id,
        "step": workflow.step
    })
    
    return result

def _send_workflow_response(client_id, response):
    with response_lock:
        if client_id in client_responses:
            try:
                client_socket = client_responses[client_id]
                
                if VOICE_ENABLED:
                    try:
                        _send_voice_prompt_for_response(client_socket, response)
                    except Exception as e:
                        _send_text_prompt_for_response(client_socket, response)
                        response_json = json.dumps(response, ensure_ascii=False, indent=2)
                        client_socket.sendall(response_json.encode("utf-8"))
                else:
                    _send_text_prompt_for_response(client_socket, response)
                    response_json = json.dumps(response, ensure_ascii=False, indent=2)
                    client_socket.sendall(response_json.encode("utf-8"))
                
                status = response.get('status')
                if status in ['complete', 'cf_failed']:
                    client_socket.close()
                    del client_responses[client_id]
                    if client_id in client_workflows:
                        del client_workflows[client_id]
                else:
                    del client_responses[client_id]
                    
            except Exception as e:
                if client_id in client_responses:
                    del client_responses[client_id]
                if client_id in client_workflows:
                    del client_workflows[client_id]

def _send_voice_prompt_for_response(client_socket, response):
    try:
        status = response.get('status')
        
        if status == 'cf_valid':
            voice_player.play_workflow_prompt(client_socket, "impegnativa_request")
        elif status == 'cf_retry':
            attempts = response.get('attempts', 0)
            voice_player.play_workflow_prompt(client_socket, "cf_retry", attempts)
        elif status == 'cf_failed':
            voice_player.play_workflow_prompt(client_socket, "cf_failed")
        elif status == 'impegnativa_retry':
            voice_player.play_workflow_prompt(client_socket, "impegnativa_retry")
        elif status == 'complete':
            voice_player.play_workflow_prompt(client_socket, "success")
        else:
            voice_player.play_workflow_prompt(client_socket, "error")
            
    except Exception as e:
        pass

def _send_text_prompt_for_response(client_socket, response):
    try:
        status = response.get('status')
        
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
        
        prompt_data = f"PROMPT:{prompt}".encode("utf-8")
        client_socket.sendall(prompt_data)
        
    except Exception as e:
        pass

def _continue_listening_for_step2(client_socket, client_id, call_start_time):
    try:
        workflow = client_workflows[client_id]
        
        audio_data = b""
        last_data_time = time.time()
        client_socket.settimeout(CLIENT_TIMEOUT)
        
        timeout_duration = workflow.get_timeout()
        
        while True:
            current_time = time.time()
            call_duration = current_time - call_start_time
            
            if call_duration > MAX_CALL_DURATION:
                break
            
            if current_time - last_data_time > timeout_duration:
                break
            
            try:
                data = client_socket.recv(4096)
                
                if not data:
                    break
                    
                if data == b"END":
                    break
                
                audio_data += data
                last_data_time = current_time
                
            except socket.timeout:
                time_since_data = current_time - last_data_time
                
                if time_since_data > timeout_duration:
                    break
                elif time_since_data > CF_DICTATION_TIME:
                    break
                elif len(audio_data) == 0:
                    break
        
        if len(audio_data) > 0:
            
            output_file = get_transcription_filename(client_id)
            success, format_detected = process_slin_audio(audio_data, output_file)
            
            if success:
                with response_lock:
                    client_responses[client_id] = client_socket
                
                transcription_queue.put((output_file, client_id, call_start_time))
                
                while client_id in client_responses:
                    time.sleep(0.1)
                
            else:
                error_response = {
                    "status": "error",
                    "message": "Audio processing failed",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "client_id": client_id,
                    "step": workflow.step
                }
                client_socket.sendall(json.dumps(error_response).encode("utf-8"))
            
    except Exception as e:
        pass

def handle_technical_client(client_socket, client_address):
    client_id = str(uuid.uuid4())[:8]
    call_start_time = time.time()
    
    if VOICE_ENABLED:
        try:
            voice_player.play_workflow_prompt(client_socket, "start")
        except Exception as e:
            try:
                welcome_text = "PROMPT:Benvenuto! Fornisci il tuo codice fiscale (16 caratteri)."
                client_socket.sendall(welcome_text.encode("utf-8"))
            except Exception as e2:
                pass
    
    if client_id not in client_workflows:
        client_workflows[client_id] = CallCenterWorkflow()
    
    workflow = client_workflows[client_id]
    
    try:
        while not workflow.is_complete():
            
            audio_data = b""
            last_data_time = time.time()
            client_socket.settimeout(CLIENT_TIMEOUT)
            
            current_step = workflow.step
            timeout_duration = workflow.get_timeout()
            
            if current_step == 1:
                workflow.start_cf_dictation()
            
            while True:
                current_time = time.time()
                call_duration = current_time - call_start_time
                
                if call_duration > MAX_CALL_DURATION:
                    break
                
                if current_step == 1 and workflow.should_process_cf_now():
                    break
                elif current_step == 2 and workflow.check_impegnativa_dictation_timeout():
                    break
                
                try:
                    data = client_socket.recv(4096)
                    
                    if not data:
                        break
                        
                    if data == b"END":
                        if current_step == 1:
                            workflow.stop_cf_dictation()
                        break
                    
                    audio_data += data
                    last_data_time = current_time
                    
                except socket.timeout:
                    time_since_data = current_time - last_data_time
                    
                    if current_step == 1 and workflow.should_process_cf_now():
                        break
                    elif current_step == 2 and time_since_data > timeout_duration:
                        break
                    elif time_since_data > CF_DICTATION_TIME:
                        break
                    elif len(audio_data) == 0:
                        break

            if len(audio_data) > 0:
                output_file = get_transcription_filename(client_id)
                success, format_detected = process_slin_audio(audio_data, output_file)
                
                if success:
                    with response_lock:
                        client_responses[client_id] = client_socket
                    
                    transcription_queue.put((output_file, client_id, call_start_time))
                    
                    while client_id in client_responses:
                        time.sleep(0.1)
                    
                    if workflow.is_complete():
                        break
                    elif workflow.cf_attempts >= workflow.max_cf_attempts:
                        break
                    else:
                        if workflow.step == 1:
                            workflow.start_cf_dictation()
                        elif workflow.step == 2:
                            workflow.start_impegnativa_dictation()
                        
                        continue
                        
                else:
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
                break

    except Exception as e:
        pass

    finally:
        try:
            client_socket.close()
        except:
            pass

def start_technical_server():
    setup_directories()
    cleanup_old_transcription_files()
    
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((HOST, PORT))
            server_socket.listen(1000)
            
            num_workers = 1
            for i in range(num_workers):
                worker = threading.Thread(target=transcription_worker, daemon=True)
                worker.start()
                transcription_workers.append(worker)
            while True:
                try:
                    client_socket, client_address = server_socket.accept()
                    client_thread = threading.Thread(
                        target=handle_technical_client, 
                        args=(client_socket, client_address),
                        daemon=True
                    )
                    client_thread.start()
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    pass
                    
    except Exception as e:
        pass

if __name__ == "__main__":
    start_technical_server()