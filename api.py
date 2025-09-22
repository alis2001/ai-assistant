from TTS.api import TTS
tts = TTS(model_name="coqui-ai/tts-english", gpu=False)
tts.tts_to_file("Ciao! Questo è un test del sistema di sintesi vocale.", "output.wav")
