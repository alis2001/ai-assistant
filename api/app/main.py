"""
FastAPI Main Application - CF Automation API
Integrates with existing audio processing and CF parsing
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import sys
import os
import tempfile
import time
import json
from pathlib import Path
from typing import Dict, Any

# Add the services directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'services'))

# Import our CF parser
from services.cf_parser import CFParserService

# Import existing modules (your working code)
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import whisper

# Initialize FastAPI app
app = FastAPI(
    title="CF Automation API",
    description="Automated Codice Fiscale extraction from Italian speech for call centers",
    version="1.0.0"
)

# Add CORS middleware for web integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
cf_parser = CFParserService()
whisper_model = None

# Initialize Whisper model on startup
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global whisper_model
    print("🚀 Starting CF Automation API...")
    
    # Load Whisper model (same as your existing code)
    print("🤖 Loading Whisper model...")
    whisper_model = whisper.load_model("medium")
    print("✅ Whisper model loaded for Italian transcription.")
    
    print("🇮🇹 CF Parser ready with", len(cf_parser.cities), "Italian cities")
    print("🎯 API ready for Codice Fiscale processing!")

@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "message": "CF Automation API",
        "version": "1.0.0",
        "status": "running",
        "cities_loaded": len(cf_parser.cities),
        "endpoints": {
            "/transcribe": "POST - Upload audio file for CF extraction",
            "/parse-text": "POST - Parse text directly for CF extraction", 
            "/health": "GET - Health check",
            "/stats": "GET - Parser statistics"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "whisper_loaded": whisper_model is not None,
        "cf_parser_loaded": len(cf_parser.cities) > 0,
        "timestamp": time.time()
    }

@app.get("/stats")
async def get_parser_stats():
    """Get CF parser statistics."""
    return cf_parser.get_parser_info()

@app.post("/parse-text")
async def parse_text_for_cf(request: Dict[str, str]):
    """
    Parse text directly for CF extraction.
    
    Expected input: {"transcription": "roma napoli milano otto cinque"}
    """
    try:
        transcription = request.get("transcription", "")
        
        if not transcription.strip():
            raise HTTPException(status_code=400, detail="Empty transcription provided")
        
        # Use our CF parser (same logic as your working parser)
        result = cf_parser.parse_transcription_to_cf(transcription)
        
        return {
            "status": "success",
            "input": transcription,
            "cf_result": result,
            "timestamp": time.time()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CF parsing error: {str(e)}")

@app.post("/transcribe")
async def transcribe_audio_for_cf(file: UploadFile = File(...)):
    """
    Upload audio file, transcribe it, and extract CF.
    Supports WAV files (same format as your existing workflow).
    """
    try:
        # Validate file type
        if not file.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="File must be audio format")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Transcribe using Whisper (same as your existing new.py)
            print(f"🎤 Transcribing audio file: {file.filename}")
            result = whisper_model.transcribe(temp_file_path, fp16=False, language="it")
            transcription = result["text"].strip()
            print(f"📝 Transcription: {transcription}")
            
            # Parse CF from transcription
            cf_result = cf_parser.parse_transcription_to_cf(transcription)
            
            # Log the processing
            print(f"🔍 CF Analysis:")
            print(f"   Input: {transcription}")
            print(f"   CF Code: {cf_result['cf_code']}")
            print(f"   Cities Matched: {cf_result['cities_matched']}")
            print(f"   Confidence: {cf_result['confidence']:.2f}")
            
            # Prepare response (same structure as your existing server)
            response = {
                "status": "success",
                "filename": file.filename,
                "transcription": transcription,
                "cf_result": cf_result,
                "processing_info": {
                    "whisper_model": "medium",
                    "language": "it",
                    "timestamp": time.time()
                }
            }
            
            return response
            
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)
            
    except Exception as e:
        print(f"❌ Error processing audio: {e}")
        raise HTTPException(status_code=500, detail=f"Audio processing error: {str(e)}")

@app.post("/transcribe-raw")
async def transcribe_raw_audio_data(request: Dict[str, Any]):
    """
    Process raw audio data (for integration with your existing client.py).
    Expected input: {"audio_data": base64_encoded_audio, "format": "ulaw"}
    """
    try:
        import base64
        import wave
        import audioop
        import numpy as np
        from scipy.signal import resample
        
        # Get audio data
        audio_data_b64 = request.get("audio_data", "")
        audio_format = request.get("format", "ulaw")
        
        if not audio_data_b64:
            raise HTTPException(status_code=400, detail="No audio data provided")
        
        # Decode base64 audio data
        audio_bytes = base64.b64decode(audio_data_b64)
        
        # Process µ-law audio (same as your existing ccformat.py and new.py)
        if audio_format == "ulaw":
            # Convert µ-law to PCM (same logic as your existing code)
            pcm_data = audioop.ulaw2lin(audio_bytes, 2)
            pcm_array = np.frombuffer(pcm_data, dtype=np.int16)
            
            # Resample to 16kHz for Whisper
            num_samples = int(len(pcm_array) * (16000 / 8000))
            resampled_array = resample(pcm_array, num_samples).astype(np.int16)
            
            # Save as temporary WAV file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                with wave.open(temp_file.name, "wb") as wf:
                    wf.setnchannels(1)  # Mono
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(16000)  # 16kHz
                    wf.writeframes(resampled_array.tobytes())
                
                temp_file_path = temp_file.name
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported audio format: {audio_format}")
        
        try:
            # Transcribe and parse (same workflow as your existing servers)
            print("🎤 Transcribing raw audio data...")
            result = whisper_model.transcribe(temp_file_path, fp16=False, language="it")
            transcription = result["text"].strip()
            print(f"📝 Raw transcription: {transcription}")
            
            # Parse CF
            cf_result = cf_parser.parse_transcription_to_cf(transcription)
            
            # Display results (same as your existing server output)
            print(f"🔍 CF ANALYSIS:")
            print(f"   Input: {transcription}")
            print(f"   CF Code: {cf_result['cf_code']}")
            print(f"   Length: {cf_result['length']}/16")
            print(f"   Complete: {cf_result['is_complete_cf']}")
            print(f"   Confidence: {cf_result['confidence']:.2f}")
            
            response = {
                "status": "success",
                "transcription": transcription,
                "cf_result": cf_result,
                "audio_info": {
                    "format": audio_format,
                    "bytes_processed": len(audio_bytes),
                    "duration_seconds": len(resampled_array) / 16000
                },
                "timestamp": time.time()
            }
            
            return response
            
        finally:
            os.unlink(temp_file_path)
            
    except Exception as e:
        print(f"❌ Raw audio processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Raw audio processing error: {str(e)}")

@app.get("/test-cf")
async def test_cf_parser_endpoint():
    """Test endpoint to verify CF parsing works."""
    test_cases = [
        "roma napoli milano",
        "f come firenze r come roma", 
        "agliè airasca",
        "otto cinque torino"
    ]
    
    results = []
    for test in test_cases:
        cf_result = cf_parser.parse_transcription_to_cf(test)
        results.append({
            "input": test,
            "cf_code": cf_result["cf_code"],
            "confidence": cf_result["confidence"]
        })
    
    return {
        "status": "success",
        "test_results": results,
        "parser_info": cf_parser.get_parser_info()
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status": "error", "timestamp": time.time()}
    )

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting CF Automation FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)