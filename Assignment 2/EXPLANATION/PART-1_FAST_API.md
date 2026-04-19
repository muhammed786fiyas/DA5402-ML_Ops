# 🚀 DA5402 – Part 1: Collaborative AI Microservice Development

---

# 🎯 Objective

Build a single FastAPI-based multi-modal AI REST API with:

- 4 AI features
- Proper Git collaboration
- Branch-based development
- Pull Requests
- Intentional merge conflict resolution
- Secure API key handling

---

# 👨‍💻 Team Role Allocation

## 🔹 Developer A (You)
- Translation endpoint
- Image generation endpoint

## 🔹 Developer B
- NER endpoint (spaCy)
- Speech synthesis endpoint (gTTS)

---

# 🏗 Project Architecture

```
app/
 ├── main.py
 ├── routes/
 │    ├── translation.py
 │    ├── image_generation.py
 │    ├── ner.py
 │    └── speech.py
 ├── services/
 │    ├── translation_service.py
 │    ├── image_service.py
 │    ├── ner_service.py
 │    └── speech_service.py
 └── utils/
      └── container.py
```

---

# 🌐 Implemented Endpoints

| Endpoint | Description |
|-----------|------------|
| POST `/translate` | English → Target language (MyMemory API) |
| POST `/generate-image` | Text → Image (HuggingFace SDXL) |
| POST `/ner` | Named Entity Recognition (spaCy) |
| POST `/speech` | Text → Speech (gTTS) |
| GET `/` | Health check endpoint |

---

# 🧠 FastAPI Concepts Used

## 🔹 APIRouter
Used to modularize endpoints.

## 🔹 Pydantic Models
Used for automatic request validation.

Example:
```python
class TranslationRequest(BaseModel):
    text: str
    target_lang: str
```

## 🔹 JSON Response
Returning dictionary automatically converts to JSON.

## 🔹 Raw Response
Used `Response()` to return image bytes.

---

# 🔐 Security Implementation

- API keys stored in `.env`
- Loaded using `os.getenv()`
- `.env` added to `.gitignore`
- No secrets pushed to GitHub

---

# 📦 Dependency Management

Pinned versions in `requirements.txt`:

```
fastapi==0.129.0
uvicorn==0.41.0
requests==2.32.5
python-dotenv==1.2.1
gTTS==2.5.4
spacy==3.8.11
```

spaCy model installed separately:

```
python -m spacy download en_core_web_sm
```

---

# 🔀 Git Workflow Followed

## ✔ Feature Branches Used
- `feature/translation`
- `feature/image-generation`
- `feature/ner-tts-implementation`

## ✔ No Direct Commits to main

## ✔ Pull Requests Created

## ✔ Intentional Merge Conflict Created

Conflict occurred in:
- `main.py`
- `.gitignore`
- `requirements.txt`

## ✔ Conflict Resolved Manually

Documented in:

```
CONFLICT.md
```

---

# 🔁 Merge Strategy

- Translation merged into image-generation branch
- Final PR created to merge into main
- Clean branch graph maintained
- Reviewed and approved before merge

---

# 🧩 Separation of Concerns

| Layer | Responsibility |
|--------|---------------|
| routes/ | API request handling |
| services/ | External API logic |
| utils/ | Helper functions |
| main.py | Application entry point |

---

# 📌 Container Readiness for Part 2

Added:

```python
socket.gethostname()
```

Returned as:

```json
{
  "translated_text": "...",
  "container_id": "abc123"
}
```

Purpose:
- Required for load balancing proof in Part 2

---

# 🧪 Testing

- Tested via `/docs`
- Verified translation API
- Verified image generation API
- Verified NER extraction
- Verified speech synthesis

---

# 🏁 Part 1 Completion Checklist

✔ All 4 AI endpoints working  
✔ REST API functional  
✔ Proper Git collaboration  
✔ Pull Requests used  
✔ Merge conflict documented  
✔ Environment variables secured  
✔ Clean project structure  
✔ Ready for Dockerization  

---

# 🎯 Transition to Part 2

Part 1 = Build AI Service  
Part 2 = Containerize and Deploy at Scale  

You move from:

AI Feature Development  
→  
Containerized Distributed Deployment