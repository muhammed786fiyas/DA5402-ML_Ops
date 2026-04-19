from fastapi import APIRouter
from pydantic import BaseModel
from app.services.ner_service import extract_entities

router = APIRouter()

class TextInput(BaseModel):
    text: str

@router.post("/ner")
def ner_endpoint(request: TextInput):
    entities = extract_entities(request.text)
    return {"entities": entities}