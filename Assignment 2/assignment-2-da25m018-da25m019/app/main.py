from fastapi import FastAPI
from app.routes import ner, speech
from app.routes.translation import router as translation_router
from app.routes.image_generation import router as image_router

app = FastAPI(title="Multi-Modal AI Service")

# Include routers
app.include_router(ner.router)
app.include_router(speech.router)
app.include_router(translation_router)
app.include_router(image_router)



@app.get("/")
def root():
    return {"message": "Multi-Modal AI Service is running"}

