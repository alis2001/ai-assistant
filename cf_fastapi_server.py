from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import wave
import audioop
import numpy as np
from scipy.signal import resample
import whisper
import json
import time
import tempfile
import os
import sys

# Add CF parser (same as your existing FastAPI)
sys.path.append('api/app/services')
from cf_parser import CFParserService

# Same configuration as new.py
SAMPLE_RATE_ORIGINAL = 8000
SAMPLE_RATE_TARGET = 16000

# Initialize FastAPI
app = FastAPI(title="CF Audio Server", description="new.py converted to FastAPI with CF parsing")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services (same as new.py)
print("🤖 Loading Whisper model...")
model = whisper.load_model("medium")
print("✅ Whisper model loaded for Italian transcription.")

print("🇮🇹 Loading CF parser...")
cf_parser = CFParserService()
print(f"✅ CF parser loaded with {len(cf_parser.cities)} cities.")

def process_and_save_audio(ulaw_data, output_file):
    """Exact same function from new.py"""
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

        print(f"✅ Audio saved: {output_file}")
        return True
    except Exception as e:
        print(f"❌ Audio processing error: {e}")
        return False

def transcribe_audio(file_path):
    """Exact same function from new.py"""
    try:
        print(f"🎤 Transcribing: {file_path}")
        result = model.transcribe(file_path, fp16=False, language="it")
        transcription = result["text"].strip()
        print(f"📝 Transcription: {transcription}")
        return transcription
    except Exception as e:
        print(f"❌ Transcription error: {e}")
        return "Errore durante la trascrizione."

@app.get("/")
def root():
    """API info"""
    return {
        "message": "CF Audio Server - new.py with FastAPI + CF parsing",
        "status": "running",
        "cities_loaded": len(cf_parser.cities)
    }

@app.get("/health")
def health():
    """Health check"""
    return {"status": "healthy", "whisper": "loaded", "cities": len(cf_parser.cities)}

@app.post("/process-audio")
async def process_audio(file: UploadFile = File(...)):
    """
    Main endpoint: Upload audio file, get CF results
    Same workflow as new.py but with CF parsing added
    """
    try:
        # Validate audio file
        if not file.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="File must be audio format")

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        try:
            # Same workflow as new.py
            print(f"📊 Processing audio file: {file.filename}")
            
            # Generate output filename (same as new.py)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = f"received_audio_{timestamp}.wav"
            
            # For uploaded WAV files, just copy (new.py processes µ-law)
            if file.filename.endswith('.wav'):
                os.rename(temp_file_path, output_file)
            else:
                # If it's raw µ-law data, process it
                with open(temp_file_path, 'rb') as f:
                    ulaw_data = f.read()
                if not process_and_save_audio(ulaw_data, output_file):
                    raise HTTPException(status_code=500, detail="Audio processing failed")

            # Transcribe (same as new.py)
            transcription = transcribe_audio(output_file)
            
            # NEW: Add CF parsing
            print("🔍 Parsing CF from transcription...")
            cf_result = cf_parser.parse_transcription_to_cf(transcription)
            
            # Display results (same style as your other servers)
            print(f"🔍 CF ANALYSIS:")
            print(f"   Input: {transcription}")
            print(f"   CF Code: {cf_result['cf_code']}")
            print(f"   Length: {cf_result['length']}/16")
            print(f"   Complete: {cf_result['is_complete_cf']}")
            print(f"   Confidence: {cf_result['confidence']:.2f}")
            
            # Return same information as new.py would, but enhanced with CF
            return {
                "status": "success",
                "filename": file.filename,
                "transcription": transcription,
                "cf_code": cf_result["cf_code"],
                "cf_length": cf_result["length"],
                "is_complete_cf": cf_result["is_complete_cf"],
                "confidence": cf_result["confidence"],
                "cities_matched": cf_result["cities_matched"],
                "audio_file": output_file,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }

        finally:
            # Clean up temp file if it still exists
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/process-raw-audio")
async def process_raw_audio(request: dict):
    """
    Process raw µ-law audio data (like new.py TCP server)
    For direct integration with call center systems
    """
    try:
        import base64
        
        # Get audio data
        audio_b64 = request.get("audio_data", "")
        if not audio_b64:
            raise HTTPException(status_code=400, detail="No audio data")
        
        # Decode audio (same as new.py)
        ulaw_data = base64.b64decode(audio_b64)
        
        print(f"📊 Received {len(ulaw_data)} bytes of µ-law audio")
        
        # Process audio (exact same as new.py)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = f"received_audio_{timestamp}.wav"
        
        if process_and_save_audio(ulaw_data, output_file):
            # Transcribe (same as new.py)
            transcription = transcribe_audio(output_file)
            
            # Add CF parsing
            cf_result = cf_parser.parse_transcription_to_cf(transcription)
            
            # Display results
            print(f"🔍 CF ANALYSIS:")
            print(f"   Input: {transcription}")
            print(f"   CF Code: {cf_result['cf_code']}")
            print(f"   Confidence: {cf_result['confidence']:.2f}")
            
            return {
                "status": "success",
                "transcription": transcription,
                "cf_code": cf_result["cf_code"],
                "confidence": cf_result["confidence"],
                "is_complete_cf": cf_result["is_complete_cf"],
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        else:
            raise HTTPException(status_code=500, detail="Audio processing failed")
            
    except Exception as e:
        print(f"❌ Raw audio error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting CF Audio Server (new.py + FastAPI + CF parsing)")
    print("📍 Same functionality as new.py but with CF parsing!")
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)
