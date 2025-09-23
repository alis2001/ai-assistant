import socket
import wave
import audioop
import numpy as np
from scipy.signal import resample
from scipy import signal
import whisper
import json
import time
import threading
import psutil
import torch

# Technical noise reduction libraries
try:
    import librosa
    import noisereduce as nr
    from scipy.signal import butter, filtfilt, wiener
    ENHANCED_AUDIO = True
    print("✅ Technical noise reduction libraries loaded")
except ImportError:
    ENHANCED_AUDIO = False
    print("⚠️  Install: 'pip install librosa noisereduce' for technical noise reduction")

# Server configuration
HOST = "0.0.0.0"
PORT = 8000
SAMPLE_RATE_ORIGINAL = 8000
SAMPLE_RATE_TARGET = 16000

# Technical noise reduction parameters
NOISE_PROFILE_LENGTH = 0.5      # Seconds of audio to use for noise profiling
SPECTRAL_FLOOR = 0.1           # Minimum spectral energy to preserve
WIENER_FILTER_SIZE = 5         # Wiener filter kernel size

# Extended timeouts for noisy environments
CLIENT_TIMEOUT = 10.0
MAX_CALL_DURATION = 120.0
CF_DICTATION_TIME = 90.0

print("🤖 Loading Whisper LARGE with technical noise optimization...")
model = whisper.load_model("large")
print("✅ Technical noise-resistant transcription ready")

active_transcriptions = 0
transcription_lock = threading.Lock()

class TechnicalCFParser:
    """Pure CF parser without noise assumptions"""
    
    def __init__(self):
        print("⚙️ Technical CF Parser loaded (no hardcoded assumptions)")
    
    def parse_cf(self, transcription):
        """Clean CF parsing without noise word filtering"""
        
        # Basic text cleaning only
        words = transcription.replace(',', ' ').replace('-', ' ').replace('.', ' ').split()
        result = []
        
        print(f"🔧 Technical CF parsing: {words}")
        
        for word in words:
            word = word.strip()
            if not word:
                continue
            
            print(f"   Processing: '{word}'")
            
            # Numbers
            if word.isdigit():
                result.append(word)
                print(f"     → Number: {word}")
                
            # Letter sequences
            elif word.isupper() and word.isalpha() and len(word) <= 6:
                result.append(word)
                print(f"     → Letters: {word}")
                
            # Mixed alphanumeric
            elif any(c.isdigit() for c in word) and any(c.isalpha() for c in word):
                cleaned = ''.join(c for c in word.upper() if c.isalnum())
                result.append(cleaned)
                print(f"     → Alphanumeric: {cleaned}")
                
            # Words - extract first letter (likely city names)
            elif word.isalpha():
                first_letter = word[0].upper()
                result.append(first_letter)
                print(f"     → Word→First: {word}→{first_letter}")
                
            # Mixed with special characters
            else:
                cleaned = ''.join(c for c in word if c.isalnum())
                if cleaned:
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
        print(f"🎯 Technical CF result: '{final_cf}' from {result}")
        return final_cf

cf_parser = TechnicalCFParser()

def analyze_audio_characteristics(audio_data):
    """Technical analysis of audio characteristics"""
    try:
        audio_float = audio_data.astype(np.float32) / 32768.0
        
        # Technical metrics
        rms_energy = np.sqrt(np.mean(audio_float ** 2))
        peak_level = np.max(np.abs(audio_float))
        dynamic_range = 20 * np.log10(peak_level / (np.std(audio_float) + 1e-10))
        
        # Spectral characteristics
        if len(audio_float) > 1024:
            fft = np.fft.fft(audio_float)
            power_spectrum = np.abs(fft) ** 2
            spectral_centroid = np.sum(np.arange(len(power_spectrum)) * power_spectrum) / np.sum(power_spectrum)
            spectral_rolloff = np.percentile(power_spectrum, 95)
        else:
            spectral_centroid = 0
            spectral_rolloff = 0
        
        # SNR estimation using statistical methods
        sorted_energy = np.sort(np.abs(audio_float))
        noise_floor = np.mean(sorted_energy[:len(sorted_energy)//10])  # Bottom 10%
        signal_peak = np.mean(sorted_energy[-len(sorted_energy)//10:])  # Top 10%
        snr_estimate = 20 * np.log10(signal_peak / max(noise_floor, 1e-10))
        
        processing_needed = snr_estimate < 20  # Technical threshold
        
        print(f"🔊 TECHNICAL AUDIO ANALYSIS:")
        print(f"   RMS: {rms_energy:.4f} | Peak: {peak_level:.4f}")
        print(f"   Dynamic Range: {dynamic_range:.1f} dB")
        print(f"   Estimated SNR: {snr_estimate:.1f} dB")
        print(f"   Heavy Processing: {'Required' if processing_needed else 'Optional'}")
        
        return {
            'snr_db': snr_estimate,
            'dynamic_range': dynamic_range,
            'needs_processing': processing_needed,
            'rms_energy': rms_energy,
            'spectral_quality': spectral_centroid / len(audio_float) if len(audio_float) > 0 else 0
        }
        
    except Exception as e:
        print(f"⚠️ Audio analysis failed: {e}")
        return {'snr_db': 0, 'needs_processing': True}

def technical_noise_reduction(audio_data, sr, analysis):
    """Pure technical noise reduction using signal processing"""
    try:
        print("🔬 Applying technical signal processing...")
        
        # Stage 1: Multi-band filtering
        # High-pass filter to remove low-frequency noise (AC hum, handling noise)
        nyquist = sr / 2
        low_cutoff = 100 / nyquist  # Remove below 100Hz
        high_cutoff = 3800 / nyquist  # Telephony upper limit
        
        # Bandpass filter for telephony range
        b, a = signal.butter(4, [low_cutoff, high_cutoff], btype='band')
        filtered_audio = signal.filtfilt(b, a, audio_data)
        
        # Stage 2: Adaptive noise reduction based on analysis
        if ENHANCED_AUDIO and analysis['needs_processing']:
            print("🔬 Applying adaptive spectral noise reduction...")
            
            # Adaptive noise reduction strength based on SNR
            reduction_strength = min(0.95, max(0.7, 1.0 - (analysis['snr_db'] / 30.0)))
            
            reduced_audio = nr.reduce_noise(
                y=filtered_audio,
                sr=sr,
                stationary=False,  # Adaptive to changing noise
                prop_decrease=reduction_strength
            )
            
            # Stage 3: Wiener filtering for additional cleanup
            if len(reduced_audio) > WIENER_FILTER_SIZE:
                wiener_filtered = wiener(reduced_audio, WIENER_FILTER_SIZE)
            else:
                wiener_filtered = reduced_audio
                
            final_audio = wiener_filtered
            print(f"✅ Advanced noise reduction applied (strength: {reduction_strength:.2f})")
            
        else:
            final_audio = filtered_audio
            print("✅ Basic filtering applied")
        
        # Stage 4: Dynamic range compression and normalization
        # Gentle compression to even out levels
        compressed_audio = np.tanh(final_audio * 2.0) / 2.0
        
        # Normalize to optimal level for Whisper
        max_val = np.max(np.abs(compressed_audio))
        if max_val > 0:
            normalized_audio = compressed_audio / max_val * 0.8
        else:
            normalized_audio = compressed_audio
            
        print("✅ Technical audio processing complete")
        return normalized_audio
        
    except Exception as e:
        print(f"❌ Technical processing error: {e}")
        return audio_data / 32768.0  # Fallback to basic normalization

def enhanced_telephony_processing(ulaw_data, output_file):
    """Technical telephony processing without assumptions"""
    try:
        print("🔬 Technical telephony processing...")
        
        # Decode µ-law
        pcm_data = audioop.ulaw2lin(ulaw_data, 2)
        pcm_array = np.frombuffer(pcm_data, dtype=np.int16)
        
        # Analyze audio characteristics
        analysis = analyze_audio_characteristics(pcm_array)
        
        # Apply technical noise reduction
        processed_audio = technical_noise_reduction(
            pcm_array.astype(np.float32) / 32768.0,
            SAMPLE_RATE_ORIGINAL,
            analysis
        )
        
        # High-quality resampling
        if ENHANCED_AUDIO:
            try:
                resampled_audio = librosa.resample(
                    processed_audio,
                    orig_sr=SAMPLE_RATE_ORIGINAL,
                    target_sr=SAMPLE_RATE_TARGET,
                    res_type='kaiser_best'
                )
                print("✅ High-quality resampling applied")
            except Exception as e:
                print(f"⚠️ Advanced resampling failed: {e}")
                num_samples = int(len(processed_audio) * (SAMPLE_RATE_TARGET / SAMPLE_RATE_ORIGINAL))
                resampled_audio = resample(processed_audio, num_samples)
        else:
            num_samples = int(len(processed_audio) * (SAMPLE_RATE_TARGET / SAMPLE_RATE_ORIGINAL))
            resampled_audio = resample(processed_audio, num_samples)
        
        # Convert to int16 and save
        resampled_int16 = (resampled_audio * 32767).astype(np.int16)
        
        with wave.open(output_file, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE_TARGET)
            wf.writeframes(resampled_int16.tobytes())

        print(f"📞 Technical processing complete: {output_file}")
        return True
        
    except Exception as e:
        print(f"❌ Technical processing failed: {e}")
        return False

def basic_telephony_processing(ulaw_data, output_file):
    """Fallback processing"""
    try:
        pcm_data = audioop.ulaw2lin(ulaw_data, 2)
        pcm_array = np.frombuffer(pcm_data, dtype=np.int16)
        num_samples = int(len(pcm_array) * (SAMPLE_RATE_TARGET / SAMPLE_RATE_ORIGINAL))
        resampled_array = resample(pcm_array, num_samples).astype(np.int16)
        
        with wave.open(output_file, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE_TARGET)
            wf.writeframes(resampled_array.tobytes())
        return True
    except:
        return False

def process_audio_with_technical_enhancement(ulaw_data, output_file):
    """Main audio processing with technical enhancement"""
    if ENHANCED_AUDIO:
        if enhanced_telephony_processing(ulaw_data, output_file):
            return True
        print("🔄 Falling back to basic processing...")
    
    return basic_telephony_processing(ulaw_data, output_file)

def transcribe_with_noise_optimization(file_path):
    """Whisper transcription optimized for noisy conditions"""
    global active_transcriptions
    
    with transcription_lock:
        active_transcriptions += 1
        current_load = active_transcriptions
    
    try:
        print(f"🎤 Technical transcription #{current_load} (concurrent: {current_load})")
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated(0) / 1024**3
            print(f"🔥 GPU: {gpu_memory:.1f}GB allocated")
        
        start_time = time.time()
        
        # Whisper optimized for noisy telephony
        result = model.transcribe(
            file_path,
            fp16=True,
            language="it",
            task="transcribe",
            
            # Noise-resistant parameters:
            temperature=0.0,                    # Deterministic for consistency
            compression_ratio_threshold=2.4,   # Lenient for compressed/noisy audio  
            logprob_threshold=-1.0,            # Lenient probability threshold
            no_speech_threshold=0.6,           # Adjusted for noise
            condition_on_previous_text=False,  # Independent processing
            
            # Beam search for better accuracy with noise
            beam_size=5,                       # Use beam search instead of greedy
            best_of=5,                         # Consider multiple candidates
            
            # Italian context
            initial_prompt="Codice fiscale italiano"
        )
        
        end_time = time.time()
        transcription = result["text"].strip()
        
        print(f"⚡ Technical transcription completed in {end_time-start_time:.2f}s")
        print(f"📝 Result: {transcription}")
        
        return transcription
        
    except Exception as e:
        print(f"❌ Technical transcription error: {e}")
        return "Errore tecnico durante la trascrizione."
    
    finally:
        with transcription_lock:
            active_transcriptions -= 1

def handle_technical_client(client_socket, client_address):
    """Technical client handling without assumptions"""
    print(f"📞 TECHNICAL CONNECTION: {client_address}")
    call_start_time = time.time()
    
    try:
        audio_data = b""
        last_data_time = time.time()
        client_socket.settimeout(CLIENT_TIMEOUT)
        
        print(f"⏳ Receiving audio data (max {CF_DICTATION_TIME:.0f}s)...")
        
        while True:
            current_time = time.time()
            call_duration = current_time - call_start_time
            
            if call_duration > MAX_CALL_DURATION:
                print(f"⏰ Maximum duration reached")
                break
            
            try:
                data = client_socket.recv(4096)
                
                if not data:
                    print(f"📞 Client disconnected - processing {len(audio_data)} bytes")
                    break
                    
                if data == b"END":
                    print("✅ END signal received")
                    break
                
                audio_data += data
                last_data_time = current_time
                
                # Progress indicator
                if len(audio_data) % 16384 == 0:  # Every 16KB
                    duration = len(audio_data) / 8000
                    print(f"📊 Received {duration:.1f}s audio...")
                
            except socket.timeout:
                time_since_data = current_time - last_data_time
                
                if time_since_data > CF_DICTATION_TIME:
                    print(f"⏰ Processing timeout - using {len(audio_data)} bytes")
                    break
                elif len(audio_data) == 0:
                    print(f"⏰ No initial data timeout")
                    break

        # Process received audio
        if audio_data:
            audio_duration = len(audio_data) / 8000
            
            print(f"📊 TECHNICAL SUMMARY:")
            print(f"   Data: {len(audio_data)} bytes (~{audio_duration:.1f}s)")
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = f"technical_cf_{timestamp}.wav"
            
            if process_audio_with_technical_enhancement(audio_data, output_file):
                transcription = transcribe_with_noise_optimization(output_file)
                cf_code = cf_parser.parse_cf(transcription)
                
                # Results
                cf_length = len(cf_code)
                is_complete = cf_length == 16
                
                print(f"🔍 TECHNICAL CF ANALYSIS:")
                print(f"   Input: {transcription}")
                print(f"   CF: {cf_code}")
                print(f"   Length: {cf_length}/16")
                print(f"   Complete: {'✅ YES' if is_complete else '❌ NO'}")
                
                response = {
                    "status": "success",
                    "transcription": transcription,
                    "cf_code": cf_code,
                    "length": cf_length,
                    "is_complete": is_complete,
                    "audio_duration": round(audio_duration, 1),
                    "processing_method": "Technical",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
                try:
                    client_socket.sendall(json.dumps(response, ensure_ascii=False).encode("utf-8"))
                    print("✅ Response sent")
                except Exception as e:
                    print(f"⚠️ Response error: {e}")
            
            else:
                print("❌ Technical processing failed")
                
        else:
            print("❌ No audio data received")

    except Exception as e:
        print(f"❌ Technical client error: {e}")

    finally:
        try:
            client_socket.close()
        except:
            pass
        
        total_time = time.time() - call_start_time
        print(f"🔚 Technical call completed: {total_time:.1f}s")
        print("=" * 50)

def start_technical_server():
    """Start technical noise-resistant server"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((HOST, PORT))
            server_socket.listen(5)
            
            print("🚀 TECHNICAL NOISE-RESISTANT CF SERVER")
            print("=" * 50)
            print(f"📍 Listening: {HOST}:{PORT}")
            print(f"🔬 Technical noise reduction: {'✅ Active' if ENHANCED_AUDIO else '⚠️ Basic'}")
            print(f"🎯 Pure signal processing approach")
            print(f"⏳ Extended timeouts for noisy environments")
            
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                print(f"🔥 GPU: {gpu_name}")
            
            print("🔄 Ready for noisy telephony connections...")
            print("=" * 50)

            while True:
                try:
                    client_socket, client_address = server_socket.accept()
                    handle_technical_client(client_socket, client_address)
                except KeyboardInterrupt:
                    print("\n🛑 Server shutdown")
                    break
                except Exception as e:
                    print(f"❌ Server error: {e}")
                    
    except Exception as e:
        print(f"❌ Server startup error: {e}")
    finally:
        print("🏁 Technical server shutdown")

if __name__ == "__main__":
    start_technical_server()