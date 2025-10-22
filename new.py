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

active_transcriptions = 0
transcription_lock = threading.Lock()

# File management configuration
AUDIO_DIR = "processed_audio"
DEBUG_DIR = "debug_audio"
RAW_EVIDENCE_DIR = "raw_evidence"
MAX_FILES_PER_DIR = 50  # Keep only latest 50 files per directory
CLEANUP_AGE_DAYS = 7    # Remove files older than 7 days

# Results log for frontend listing
CF_RESULTS_LOG = "cf_results.jsonl"

# Optional remote ingest (VM frontend) configuration
VM_INGEST_URL = os.environ.get("VM_INGEST_URL")  # e.g., http://10.10.13.122:38473/api/ingest
VM_INGEST_TOKEN = os.environ.get("VM_INGEST_TOKEN")  # shared secret, optional

def setup_directories():
    """Create and setup directory structure"""
    directories = [AUDIO_DIR, DEBUG_DIR, RAW_EVIDENCE_DIR]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Directory ready: {directory}")

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

def get_timestamped_filename(directory, prefix, extension="wav"):
    """Generate timestamped filename in specified directory"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(directory, f"{prefix}_{timestamp}.{extension}")

def cleanup_after_call():
    """Clean up all files after each call processing"""
    try:
        print("🧹 Cleaning up files after call...")
        
        # Clean debug files (remove all)
        if os.path.exists(DEBUG_DIR):
            debug_files = [f for f in os.listdir(DEBUG_DIR) if f.endswith('.wav')]
            for file in debug_files:
                os.remove(os.path.join(DEBUG_DIR, file))
            if debug_files:
                print(f"   Removed {len(debug_files)} debug files")
        
        # Clean processed audio files (keep only latest 3)
        if os.path.exists(AUDIO_DIR):
            audio_files = []
            for filename in os.listdir(AUDIO_DIR):
                filepath = os.path.join(AUDIO_DIR, filename)
                if os.path.isfile(filepath) and filename.endswith('.wav'):
                    mtime = os.path.getmtime(filepath)
                    audio_files.append((filepath, mtime))
            
            # Sort by modification time (newest first)
            audio_files.sort(key=lambda x: x[1], reverse=True)
            
            # Remove all but the latest 3 files
            removed_count = 0
            for filepath, mtime in audio_files[3:]:
                os.remove(filepath)
                removed_count += 1
            
            if removed_count > 0:
                print(f"   Removed {removed_count} old processed audio files")
        
        # Clean raw evidence files (keep only latest 3)
        if os.path.exists(RAW_EVIDENCE_DIR):
            evidence_files = []
            for filename in os.listdir(RAW_EVIDENCE_DIR):
                filepath = os.path.join(RAW_EVIDENCE_DIR, filename)
                if os.path.isfile(filepath) and (filename.endswith('.bin') or filename.endswith('.json')):
                    mtime = os.path.getmtime(filepath)
                    evidence_files.append((filepath, mtime))
            
            # Sort by modification time (newest first)
            evidence_files.sort(key=lambda x: x[1], reverse=True)
            
            # Remove all but the latest 6 files (3 .bin + 3 .json pairs)
            removed_count = 0
            for filepath, mtime in evidence_files[6:]:
                os.remove(filepath)
                removed_count += 1
            
            if removed_count > 0:
                print(f"   Removed {removed_count} old evidence files")
        
        print("✅ Directory cleanup complete")
        
    except Exception as e:
        print(f"⚠️ Cleanup error: {e}")

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

def debug_audio_data(audio_data):
    """Enhanced debug audio data to find patterns and headers"""
    print("COMPREHENSIVE AUDIO DEBUG ANALYSIS:")
    print(f"   Total bytes: {len(audio_data)}")
    print(f"   🔢 Odd/Even: {'ODD' if len(audio_data) % 2 == 1 else 'EVEN'} (16-bit needs EVEN)")
    
    if len(audio_data) < 64:
        print("   Too little data for analysis")
        return
    
    hex_preview = ' '.join([f'{b:02x}' for b in audio_data[:64]])
    print(f"   First 64 bytes (hex):")
    print(f"      {hex_preview[:48]}")
    print(f"      {hex_preview[48:]}")
    
    hex_end = ' '.join([f'{b:02x}' for b in audio_data[-32:]])
    print(f"   Last 32 bytes (hex): {hex_end}")
    
    first_bytes = audio_data[:16]
    print(f"   First 16 bytes as values: {list(first_bytes)}")
    
    if audio_data[:4] == b'RIFF':
        print("   DETECTED: WAV header (RIFF)")
    elif audio_data[:3] == b'ID3':
        print("   DETECTED: MP3 ID3 tag")
    elif audio_data[0:2] == bytes([0xFF, 0xFB]) or audio_data[0:2] == bytes([0xFF, 0xFA]):
        print("   DETECTED: MP3 frame header")
    elif len(set(audio_data[:10])) < 3:
        print("   DETECTED: Very low entropy start (possibly padding/header)")
    
    byte_counts = {}
    for b in audio_data[:1000]:
        byte_counts[b] = byte_counts.get(b, 0) + 1
    
    most_common = sorted(byte_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"   Most common bytes (first 1000): {most_common}")
    
    if len(audio_data) >= 100:
        for period in [1, 2, 4, 8, 16, 32]:
            if len(audio_data) >= period * 10:
                chunks = [audio_data[i:i+period] for i in range(0, min(100, len(audio_data)), period)]
                unique_chunks = len(set(chunks))
                if unique_chunks < len(chunks) * 0.5:
                    print(f"   PATTERN: Possible {period}-byte repeating pattern detected")
    
    print("   🔊 Quick format quality test (first 8000 bytes):")
    test_data = audio_data[:8000] if len(audio_data) >= 8000 else audio_data
    
    formats_to_test = [
        ("8-bit unsigned", lambda x: np.frombuffer(x, dtype='u1')),
        ("µ-law", lambda x: np.frombuffer(audioop.ulaw2lin(x, 2), dtype='i2')),
        ("A-law", lambda x: np.frombuffer(audioop.alaw2lin(x, 2), dtype='i2')),
    ]
    
    if len(test_data) % 2 == 0:
        formats_to_test.extend([
            ("16-bit LE", lambda x: np.frombuffer(x, dtype='<i2')),
            ("16-bit BE", lambda x: np.frombuffer(x, dtype='>i2')),
            ("slin@8000 (unsigned 16-bit LE)", lambda x: (np.frombuffer(x, dtype='<u2') - 32768).astype(np.int16)),
            ("slin@8000 (unsigned 16-bit BE)", lambda x: (np.frombuffer(x, dtype='>u2') - 32768).astype(np.int16)),
        ])
    
    for format_name, converter in formats_to_test:
        try:
            samples = converter(test_data)
            if len(samples) > 100:
                samples_f = samples.astype(np.float32)
                if format_name == "8-bit unsigned":
                    samples_f = (samples_f - 128) / 128.0
                elif "slin@8000" in format_name:
                    samples_f = samples_f / 32768.0
                else:
                    samples_f = samples_f / 32768.0
                    
                rms = np.sqrt(np.mean(samples_f**2))
                peak = np.max(np.abs(samples_f))
                
                zero_crossings = np.sum(np.diff(np.sign(samples_f)) != 0)
                zc_rate = zero_crossings / len(samples_f)
                
                print(f"      {format_name}: RMS={rms:.4f}, Peak={peak:.4f}, ZC={zc_rate:.4f}")
                
                if 0.05 < rms < 0.8 and peak < 1.0 and 0.01 < zc_rate < 0.3:
                    print(f"         LOOKS PROMISING!")
                
        except Exception as e:
            print(f"      {format_name}: FAILED ({e})")
            
    print("   Look for formats marked as 'LOOKS PROMISING!' above")

def save_debug_audio(audio_data, filename_prefix):
    """Save audio in ALL possible interpretations for debugging"""
    try:
        print(f"Creating comprehensive debug files in {DEBUG_DIR}/...")
        
        # Clean up old debug files first
        cleanup_old_files(DEBUG_DIR, max_files=20, max_age_days=3)  # Debug files expire faster
        
        if len(audio_data) % 2 == 1:
            even_data = audio_data[:-1]
        else:
            even_data = audio_data
            
        debug_files_created = []
        
        sample_rates = [8000, 16000, 22050, 44100, 11025, 12000]
        
        for sr in sample_rates:
            try:
                samples_le = np.frombuffer(even_data, dtype='<i2')
                filename = get_timestamped_filename(DEBUG_DIR, f"{filename_prefix}_16bit_LE_{sr}Hz")
                with wave.open(filename, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sr)
                    wf.writeframes(samples_le.tobytes())
                debug_files_created.append(f"16bit_LE_{sr}Hz")
                
                samples_be = np.frombuffer(even_data, dtype='>i2')
                filename = get_timestamped_filename(DEBUG_DIR, f"{filename_prefix}_16bit_BE_{sr}Hz")
                with wave.open(filename, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sr)
                    wf.writeframes(samples_be.tobytes())
                debug_files_created.append(f"16bit_BE_{sr}Hz")
                
                samples_slin_le = (np.frombuffer(even_data, dtype='<u2') - 32768).astype(np.int16)
                filename = get_timestamped_filename(DEBUG_DIR, f"{filename_prefix}_slin_LE_{sr}Hz")
                with wave.open(filename, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sr)
                    wf.writeframes(samples_slin_le.tobytes())
                debug_files_created.append(f"slin_LE_{sr}Hz")
                
                samples_slin_be = (np.frombuffer(even_data, dtype='>u2') - 32768).astype(np.int16)
                filename = get_timestamped_filename(DEBUG_DIR, f"{filename_prefix}_slin_BE_{sr}Hz")
                with wave.open(filename, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sr)
                    wf.writeframes(samples_slin_be.tobytes())
                debug_files_created.append(f"slin_BE_{sr}Hz")
            except: pass
        
        for sr in sample_rates:
            try:
                samples_8bit = np.frombuffer(audio_data, dtype='u1')
                samples_8bit_16 = ((samples_8bit.astype(np.int16) - 128) * 256)
                filename = get_timestamped_filename(DEBUG_DIR, f"{filename_prefix}_8bit_unsigned_{sr}Hz")
                with wave.open(filename, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sr)
                    wf.writeframes(samples_8bit_16.tobytes())
                debug_files_created.append(f"8bit_unsigned_{sr}Hz")
                
                samples_8bit_signed = np.frombuffer(audio_data, dtype='i1')
                samples_8bit_signed_16 = (samples_8bit_signed.astype(np.int16) * 256)
                filename = get_timestamped_filename(DEBUG_DIR, f"{filename_prefix}_8bit_signed_{sr}Hz")
                with wave.open(filename, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sr)
                    wf.writeframes(samples_8bit_signed_16.tobytes())
                debug_files_created.append(f"8bit_signed_{sr}Hz")
            except: pass
            
        import audioop
        for sr in sample_rates:
            # Try A-law first (call center format)
            try:
                pcm_data = audioop.alaw2lin(audio_data, 2)
                filename = get_timestamped_filename(DEBUG_DIR, f"{filename_prefix}_alaw_{sr}Hz")
                with wave.open(filename, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sr)
                    wf.writeframes(pcm_data)
                debug_files_created.append(f"alaw_{sr}Hz")
            except: pass
            
            # Try µ-law second (fallback)
            try:
                pcm_data = audioop.ulaw2lin(audio_data, 2)
                filename = get_timestamped_filename(DEBUG_DIR, f"{filename_prefix}_ulaw_{sr}Hz")
                with wave.open(filename, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sr)
                    wf.writeframes(pcm_data)
                debug_files_created.append(f"ulaw_{sr}Hz")
            except: pass
            
        header_skips = [1, 2, 4, 8, 16, 32, 64]
        for skip in header_skips:
            if len(audio_data) > skip + 1000:
                try:
                    data_no_header = audio_data[skip:]
                    
                    # Try A-law first (call center format)
                    pcm_data = audioop.alaw2lin(data_no_header, 2)
                    with wave.open(f"{filename_prefix}_alaw_skip{skip}_8kHz.wav", "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(8000)
                        wf.writeframes(pcm_data)
                    debug_files_created.append(f"alaw_skip{skip}")
                    
                    # Try µ-law fallback
                    pcm_data = audioop.ulaw2lin(data_no_header, 2)
                    with wave.open(f"{filename_prefix}_ulaw_skip{skip}_8kHz.wav", "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(8000)
                        wf.writeframes(pcm_data)
                    debug_files_created.append(f"ulaw_skip{skip}")
                    
                    # Try 8-bit unsigned
                    samples_8bit = np.frombuffer(data_no_header, dtype='u1')
                    samples_8bit_16 = ((samples_8bit.astype(np.int16) - 128) * 256)
                    with wave.open(f"{filename_prefix}_8bit_unsigned_skip{skip}_8kHz.wav", "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(8000)
                        wf.writeframes(samples_8bit_16.tobytes())
                    debug_files_created.append(f"8bit_unsigned_skip{skip}")
                    
                except: pass
                
        if len(audio_data) % 4 == 0:
            try:
                samples_f32 = np.frombuffer(audio_data, dtype='<f4')
                samples_clipped = np.clip(samples_f32, -1.0, 1.0)
                samples_int16 = (samples_clipped * 32767).astype(np.int16)
                with wave.open(f"{filename_prefix}_32bit_float_LE_8kHz.wav", "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(8000)
                    wf.writeframes(samples_int16.tobytes())
                debug_files_created.append("32bit_float_LE")
            except: pass
            
        print(f"Created {len(debug_files_created)} debug files")
        print(f"LISTEN TO EACH FILE - Find the one with clearest audio:")
        for i, filename in enumerate(debug_files_created[:20]):
            print(f"   {i+1:2d}. {filename}")
        if len(debug_files_created) > 20:
            print(f"   ... and {len(debug_files_created)-20} more files")
        
        return debug_files_created
        
    except Exception as e:
        print(f"Debug save error: {e}")
        return []

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
    """Transcribe using Whisper with concurrency control"""
    global active_transcriptions
    
    with transcription_lock:
        active_transcriptions += 1
        print(f"Starting transcription #{active_transcriptions}: {file_path}")
    
    try:
        result = model.transcribe(file_path, fp16=False, language="it")
        transcription = result["text"].strip()
        print(f"📝 Transcription: {transcription}")
        return transcription
    except Exception as e:
        print(f"Transcription error: {e}")
        return "Errore durante la trascrizione."
    finally:
        with transcription_lock:
            active_transcriptions -= 1

def handle_technical_client(client_socket, client_address):
    """Technical client handling with auto-format detection"""
    print(f"TECHNICAL CONNECTION: {client_address}")
    call_start_time = time.time()
    
    try:
        audio_data = b""
        last_data_time = time.time()
        client_socket.settimeout(CLIENT_TIMEOUT)
        
        print(f"Receiving audio data with auto-format detection (max {CF_DICTATION_TIME:.0f}s)...")
        
        while True:
            current_time = time.time()
            call_duration = current_time - call_start_time
            
            if call_duration > MAX_CALL_DURATION:
                print(f"⏰ Maximum duration reached")
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
                    print(f"Received ~{duration:.1f}s audio ({len(audio_data)} bytes) - expecting slin@8000...")
                
            except socket.timeout:
                time_since_data = current_time - last_data_time
                
                if time_since_data > CF_DICTATION_TIME:
                    print(f"⏰ Processing timeout - using {len(audio_data)} bytes")
                    break
                elif len(audio_data) == 0:
                    print(f"⏰ No initial data timeout")
                    break

        if len(audio_data) > 0:
            print(f"Processing {len(audio_data)} bytes with auto-format detection...")
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            os.makedirs("raw_evidence", exist_ok=True)
            raw_filename = f"raw_evidence/raw_audio_evidence_{timestamp}.bin"
            metadata_filename = f"raw_evidence/raw_audio_metadata_{timestamp}.json"
            
            with open(raw_filename, "wb") as raw_file:
                raw_file.write(audio_data)
            print(f"Raw audio evidence saved: {raw_filename} ({len(audio_data)} bytes)")
            
            metadata = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "client_address": str(client_address),
                "total_bytes": len(audio_data),
                "call_duration_seconds": time.time() - call_start_time,
                "estimated_duration_seconds": len(audio_data) // 2 / SAMPLE_RATE_ORIGINAL,
                "call_center_format_claimed": "slin@8000",
                "actual_format_detected": "unknown",
                "file_description": "Raw audio data exactly as received from call center",
                "purpose": "Evidence of actual audio format being sent"
            }
            
            with open(metadata_filename, "w") as meta_file:
                json.dump(metadata, meta_file, indent=2)
            print(f"Raw audio metadata saved: {metadata_filename}")
            
            output_file = get_timestamped_filename(AUDIO_DIR, "processed_audio")
            
            success, format_detected = process_slin_audio(audio_data, output_file)
            
            def update_raw_metadata(actual_format):
                try:
                    with open(metadata_filename, "r") as meta_file:
                        metadata = json.load(meta_file)
                    metadata["actual_format_detected"] = actual_format
                    metadata["processing_successful"] = success
                    with open(metadata_filename, "w") as meta_file:
                        json.dump(metadata, meta_file, indent=2)
                except:
                    pass
            
            update_raw_metadata(format_detected)
            
            if success:
                transcription = transcribe_audio(output_file)
                
                cf_parser = TechnicalCFParser()
                cf_result = cf_parser.parse_cf(transcription)
                
                call_duration = time.time() - call_start_time
                estimated_samples = len(audio_data) // 2
                audio_duration = estimated_samples / SAMPLE_RATE_ORIGINAL
                
                print("=" * 60)
                print("CF ANALYSIS RESULTS:")
                print("=" * 60)
                print(f"📝 Raw transcription: {transcription}")
                print(f"🆔 CF Code: {cf_result['cf_code']}")
                print(f"📏 Length: {cf_result['length']}/16 characters")
                print(f"Complete: {'YES' if cf_result['is_complete'] else 'NO'}")
                print(f"Confidence: {cf_result['confidence']:.2f}")
                print(f"Call duration: {call_duration:.1f}s")
                print(f"Audio duration: {audio_duration:.1f}s")
                print("=" * 60)
                
                response = {
                    "status": "success",
                    "transcription": transcription,
                    "cf_code": cf_result["cf_code"],
                    "length": cf_result["length"],
                    "is_complete": cf_result["is_complete"],
                    "confidence": cf_result["confidence"],
                    "cf_parts": cf_result["parts"],
                    "call_duration": call_duration,
                    "audio_duration": audio_duration,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
                # Append to results log for frontend listing
                try:
                    append_cf_result(response)
                except Exception as e:
                    print(f"⚠️ CF result logging failed: {e}")

                response_json = json.dumps(response, ensure_ascii=False, indent=2)
                client_socket.sendall(response_json.encode("utf-8"))

                # Also try to push to VM frontend if configured
                try:
                    post_cf_result_to_vm(response)
                except Exception as e:
                    print(f"⚠️ Push to VM failed: {e}")
                
            else:
                print("Multi-format audio processing failed completely")
                update_raw_metadata("FAILED - No format could be processed")
                error_response = {
                    "status": "error",
                    "message": "Multi-format audio processing failed",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                client_socket.sendall(json.dumps(error_response).encode("utf-8"))
        else:
            print("No audio data received")

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
        print(f"🔚 Technical call completed: {total_time:.1f}s")
        
        # Clean up files after processing
        cleanup_after_call()
        
        print("=" * 50)

def start_technical_server():
    """Start technical server with auto-format detection"""
    # Setup directories and cleanup
    setup_directories()
    cleanup_old_files(AUDIO_DIR)
    cleanup_old_files(RAW_EVIDENCE_DIR)
    
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((HOST, PORT))
            server_socket.listen(5)
            
            print("HIGH-QUALITY CF SERVER (Call Center Approach)")
            print("=" * 60)
            print(f"📍 Listening: {HOST}:{PORT}")
            print(f"🎯 FORMAT: A-law with call center quality processing")
            print(f"🔧 FRAME PROTOCOL: 3-byte headers with type 0x10 frames")
            print(f"🎵 CUSTOM DECODER: Call center's exact A-law decoder")
            print(f"🎛️ FFMPEG FILTERS: highpass=200, lowpass=3400, dynaudnorm")
            print(f"📊 Sample rate: 8kHz → 16kHz (quality resampling)")
            print(f"🧹 AUTO-CLEANUP: Removes files after each call")
            print(f"📁 Organized directories: processed_audio/, raw_evidence/")
            print(f"🔤 CF Parser: Simple first-letter extraction (no mapping)")
            
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                print(f"GPU: {gpu_name}")
            
            print("Ready for comprehensive audio format detection...")
            print("=" * 60)

            while True:
                try:
                    client_socket, client_address = server_socket.accept()
                    handle_technical_client(client_socket, client_address)
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