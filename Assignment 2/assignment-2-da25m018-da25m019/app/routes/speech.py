from fastapi import APIRouter
from pydantic import BaseModel
from app.services.speech_service import text_to_speech

router = APIRouter()

class TextInput(BaseModel):
    text: str

@router.post("/speech")
def speech_endpoint(request: TextInput):
    audio_base64 = text_to_speech(request.text)
    return {"audio_base64": audio_base64}

