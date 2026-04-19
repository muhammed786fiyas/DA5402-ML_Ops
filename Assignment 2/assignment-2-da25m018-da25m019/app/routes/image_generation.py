from fastapi import APIRouter, Response
from pydantic import BaseModel
from app.services.image_service import generate_image
from app.utils.container import get_container_id

router = APIRouter()

class ImageRequest(BaseModel):
    prompt: str

@router.post("/generate-image")
def generate(request: ImageRequest):
    image_bytes = generate_image(request.prompt)

    return Response(
        content=image_bytes,
        media_type="image/png",
        headers={"container_id": get_container_id()}
    )
