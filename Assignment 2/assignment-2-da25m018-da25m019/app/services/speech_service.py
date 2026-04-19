import os
from gtts import gTTS
from datetime import datetime

# Ensure output directory exists
SPEECH_OUTPUT_DIR = "app/outputs/speech"
os.makedirs(SPEECH_OUTPUT_DIR, exist_ok=True)

def text_to_speech(text: str) -> str:
    tts = gTTS(text=text, lang="en")

    filename = f"speech_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
    filepath = os.path.join(SPEECH_OUTPUT_DIR, filename)

    tts.save(filepath)

    return {"saved_file": filepath}