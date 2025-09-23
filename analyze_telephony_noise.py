#!/usr/bin/env python3
"""
WAV File Analysis for Telephony "Radio Robotic Voice" Issues
Run this on your saved WAV files to identify the exact noise problem
"""

import numpy as np
import matplotlib.pyplot as plt
import wave
import scipy.signal as signal
from scipy.fft import fft, fftfreq
import os
import glob

def analyze_wav_file(wav_file):
    """Analyze a WAV file for telephony noise issues"""
    print(f"\n🔍 ANALYZING: {wav_file}")
    print("=" * 60)
    
    try:
        # Load WAV file
        with wave.open(wav_file, 'rb') as wf:
            sample_rate = wf.getframerate()
            frames = wf.getnframes()
            audio_data = wf.readframes(frames)
            
        # Convert to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        duration = len(audio_array) / sample_rate
        
        print(f"📊 FILE INFO:")
        print(f"   Duration: {duration:.2f} seconds")
        print(f"   Sample Rate: {sample_rate} Hz")
        print(f"   Samples: {len(audio_array)}")
        
        # Basic audio characteristics
        rms_level = np.sqrt(np.mean(audio_array ** 2))
        peak_level = np.max(np.abs(audio_array))
        
        print(f"\n🎚️ AUDIO LEVELS:")
        print(f"   RMS Level: {rms_level:.4f}")
        print(f"   Peak Level: {peak_level:.4f}")
        print(f"   RMS (dB): {20 * np.log10(rms_level + 1e-10):.1f} dB")
        
        # Frequency analysis
        print(f"\n🔬 FREQUENCY ANALYSIS:")
        fft_size = min(8192, len(audio_array))
        fft_data = fft(audio_array[:fft_size])
        freqs = fftfreq(fft_size, 1/sample_rate)
        power_spectrum = np.abs(fft_data) ** 2
        
        # Only look at positive frequencies
        pos_freqs = freqs[:fft_size//2]
        pos_power = power_spectrum[:fft_size//2]
        
        # Find dominant frequencies
        peak_indices = signal.find_peaks(pos_power, height=np.max(pos_power) * 0.1)[0]
        dominant_freqs = pos_freqs[peak_indices]
        dominant_powers = pos_power[peak_indices]
        
        print(f"   Dominant Frequencies:")
        for freq, power in zip(dominant_freqs[:10], dominant_powers[:10]):
            if freq > 0:
                power_db = 10 * np.log10(power + 1e-10)
                print(f"     {freq:.0f} Hz: {power_db:.1f} dB")
        
        # Check for telephony artifacts
        print(f"\n📞 TELEPHONY ARTIFACT DETECTION:")
        
        # 1. Check for carrier frequencies (common in telephony)
        carrier_freqs = [50, 60, 400, 800, 1000, 1200, 1600, 2000, 2400, 2800, 3200]
        detected_carriers = []
        
        for carrier_freq in carrier_freqs:
            if carrier_freq < sample_rate / 2:
                freq_idx = np.argmin(np.abs(pos_freqs - carrier_freq))
                if freq_idx < len(pos_power):
                    carrier_power = pos_power[freq_idx]
                    avg_power = np.mean(pos_power)
                    
                    if carrier_power > avg_power * 5:  # 5x above average
                        power_db = 10 * np.log10(carrier_power + 1e-10)
                        detected_carriers.append((carrier_freq, power_db))
                        print(f"   ⚠️  CARRIER at {carrier_freq} Hz: {power_db:.1f} dB")
        
        if not detected_carriers:
            print(f"   ✅ No obvious carrier frequencies detected")
        
        # 2. Check for quantization noise patterns
        print(f"\n🔢 QUANTIZATION ANALYSIS:")
        audio_diff = np.diff(audio_array)
        small_steps = np.sum(np.abs(audio_diff) < 0.001) / len(audio_diff)
        print(f"   Small step percentage: {small_steps * 100:.1f}%")
        
        if small_steps > 0.15:
            print(f"   ⚠️  HIGH quantization noise detected!")
        else:
            print(f"   ✅ Quantization noise normal")
        
        # 3. Check for robotic voice patterns (regularity)
        print(f"\n🤖 ROBOTIC VOICE ANALYSIS:")
        frame_size = int(0.025 * sample_rate)  # 25ms frames
        frame_energies = []
        
        for i in range(0, len(audio_array) - frame_size, frame_size):
            frame = audio_array[i:i + frame_size]
            energy = np.sum(frame ** 2)
            frame_energies.append(energy)
        
        if len(frame_energies) > 10:
            energy_var = np.var(frame_energies)
            energy_mean = np.mean(frame_energies)
            regularity = energy_var / max(energy_mean, 1e-10)
            
            print(f"   Voice regularity: {regularity:.3f}")
            if regularity < 0.5:
                print(f"   ⚠️  ROBOTIC voice pattern detected!")
            else:
                print(f"   ✅ Natural voice variation detected")
        
        # 4. Check for GSM codec artifacts
        print(f"\n📱 GSM/CODEC ARTIFACT ANALYSIS:")
        gsm_freqs = [270, 400, 540, 800, 1080, 1200, 1600]  # Common GSM artifact frequencies
        gsm_artifacts = []
        
        for gsm_freq in gsm_freqs:
            if gsm_freq < sample_rate / 2:
                freq_idx = np.argmin(np.abs(pos_freqs - gsm_freq))
                if freq_idx < len(pos_power):
                    gsm_power = pos_power[freq_idx]
                    local_avg = np.mean(pos_power[max(0, freq_idx-10):min(len(pos_power), freq_idx+10)])
                    
                    if gsm_power > local_avg * 3:
                        power_db = 10 * np.log10(gsm_power + 1e-10)
                        gsm_artifacts.append((gsm_freq, power_db))
                        print(f"   ⚠️  GSM artifact at {gsm_freq} Hz: {power_db:.1f} dB")
        
        if not gsm_artifacts:
            print(f"   ✅ No GSM codec artifacts detected")
        
        # 5. High frequency digital noise
        print(f"\n💾 DIGITAL NOISE ANALYSIS:")
        high_freq_start = int(len(pos_freqs) * 0.7)  # Top 30% of frequencies
        high_freq_power = np.mean(pos_power[high_freq_start:])
        total_power = np.mean(pos_power)
        
        high_freq_ratio = high_freq_power / max(total_power, 1e-10)
        print(f"   High frequency ratio: {high_freq_ratio:.3f}")
        
        if high_freq_ratio > 0.3:
            print(f"   ⚠️  HIGH digital noise detected!")
        else:
            print(f"   ✅ Digital noise normal")
        
        # Generate recommendations
        print(f"\n🎯 RECOMMENDATIONS:")
        problems_found = []
        
        if detected_carriers:
            problems_found.append("carrier_interference")
            print(f"   🔧 Apply notch filters at: {[f[0] for f in detected_carriers]} Hz")
        
        if small_steps > 0.15:
            problems_found.append("quantization_noise")
            print(f"   🔧 Apply quantization noise smoothing")
        
        if len(frame_energies) > 10 and regularity < 0.5:
            problems_found.append("robotic_voice")
            print(f"   🔧 Apply voice naturalness enhancement")
        
        if gsm_artifacts:
            problems_found.append("gsm_artifacts")
            print(f"   🔧 Apply GSM artifact removal at: {[f[0] for f in gsm_artifacts]} Hz")
        
        if high_freq_ratio > 0.3:
            problems_found.append("digital_noise")
            print(f"   🔧 Apply high-frequency noise reduction")
        
        if not problems_found:
            print(f"   ✅ No major issues detected - audio quality looks good!")
        
        # Create spectrum plot
        plt.figure(figsize=(12, 8))
        
        # Spectrum plot
        plt.subplot(2, 1, 1)
        plt.plot(pos_freqs, 10 * np.log10(pos_power + 1e-10))
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power (dB)')
        plt.title(f'Power Spectrum: {os.path.basename(wav_file)}')
        plt.grid(True)
        plt.xlim(0, min(4000, sample_rate/2))  # Focus on speech range
        
        # Mark problem frequencies
        for freq, power_db in detected_carriers:
            plt.axvline(x=freq, color='red', linestyle='--', alpha=0.7, label=f'Carrier {freq}Hz')
        
        for freq, power_db in gsm_artifacts:
            plt.axvline(x=freq, color='orange', linestyle=':', alpha=0.7, label=f'GSM {freq}Hz')
        
        # Waveform plot
        plt.subplot(2, 1, 2)
        time_axis = np.linspace(0, duration, len(audio_array))
        plt.plot(time_axis[:min(len(time_axis), sample_rate * 5)], 
                audio_array[:min(len(audio_array), sample_rate * 5)])  # First 5 seconds
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Waveform (First 5 seconds)')
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = wav_file.replace('.wav', '_analysis.png')
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"\n📊 Analysis plot saved: {plot_filename}")
        plt.show()
        
        return {
            'problems_found': problems_found,
            'detected_carriers': detected_carriers,
            'gsm_artifacts': gsm_artifacts,
            'quantization_noise': small_steps,
            'robotic_voice': regularity if len(frame_energies) > 10 else 0,
            'high_freq_noise': high_freq_ratio
        }
        
    except Exception as e:
        print(f"❌ Error analyzing {wav_file}: {e}")
        return None

def batch_analyze_wavs(directory="."):
    """Analyze all WAV files in a directory"""
    wav_files = glob.glob(os.path.join(directory, "*.wav"))
    
    if not wav_files:
        print("❌ No WAV files found in current directory")
        return
    
    print(f"🔍 Found {len(wav_files)} WAV files to analyze...")
    
    all_results = {}
    
    for wav_file in wav_files:
        result = analyze_wav_file(wav_file)
        if result:
            all_results[wav_file] = result
    
    # Summary
    print(f"\n📋 BATCH ANALYSIS SUMMARY")
    print("=" * 60)
    
    problem_counts = {}
    for filename, result in all_results.items():
        for problem in result['problems_found']:
            problem_counts[problem] = problem_counts.get(problem, 0) + 1
    
    if problem_counts:
        print("🔍 Problems found across all files:")
        for problem, count in problem_counts.items():
            print(f"   {problem}: {count}/{len(all_results)} files")
    else:
        print("✅ No major problems detected in any files!")

if __name__ == "__main__":
    import sys
    
    print("🔬 TELEPHONY AUDIO ANALYZER")
    print("Detects 'radio robotic voice' and other telephony issues")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        # Analyze specific file
        wav_file = sys.argv[1]
        if os.path.exists(wav_file):
            analyze_wav_file(wav_file)
        else:
            print(f"❌ File not found: {wav_file}")
    else:
        # Analyze all WAV files in current directory
        batch_analyze_wavs()
    
    print(f"\n✅ Analysis complete!")