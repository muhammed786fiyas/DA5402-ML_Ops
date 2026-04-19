from fastapi import APIRouter
from pydantic import BaseModel
from app.services.translation_service import translate_text
from app.utils.container import get_container_id

router = APIRouter()

class TranslationRequest(BaseModel):
    text: str
    target_lang: str

@router.post("/translate")
def translate(request: TranslationRequest):
    result = translate_text(request.text, request.target_lang)
    return {
        "translated_text": result,
        "container_id": get_container_id()
    }
