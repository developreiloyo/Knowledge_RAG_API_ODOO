from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from fastapi import Depends
from auth import get_api_key

from services.retrieval_service import answer_question

# --------------------------------------------------
# App
# --------------------------------------------------
app = FastAPI(
    title="Knowledge Retrieval API",
    version="1.1.0",
)

# --------------------------------------------------
# Constantes de validación
# --------------------------------------------------
ALLOWED_DOMAINS = {"odoo", "wms", "legal", "finance"}
ALLOWED_LANGUAGES = {"en", "es", "pt"}

MAX_QUESTION_LENGTH = 500
MAX_TOP_K = 10

# --------------------------------------------------
# Schemas
# --------------------------------------------------
class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=5,
        max_length=MAX_QUESTION_LENGTH,
        description="Pregunta del usuario"
    )
    domain: str
    module: Optional[str] = None
    language: Optional[str] = "en"
    top_k: int = Field(default=5, ge=1, le=MAX_TOP_K)

    # 🔹 Validadores
    @validator("domain")
    def validate_domain(cls, v):
        if v not in ALLOWED_DOMAINS:
            raise ValueError(f"Dominio inválido. Permitidos: {ALLOWED_DOMAINS}")
        return v

    @validator("language")
    def validate_language(cls, v):
        if v not in ALLOWED_LANGUAGES:
            raise ValueError(f"Idioma inválido. Permitidos: {ALLOWED_LANGUAGES}")
        return v

    @validator("question")
    def clean_question(cls, v):
        if v.strip().lower() in {"hi", "hola", "hello", "test"}:
            raise ValueError("Pregunta demasiado vaga")
        return v.strip()

class Source(BaseModel):
    content: str
    similarity: float

class AskResponse(BaseModel):
    answer: str
    sources: List[Source]

# --------------------------------------------------
# Endpoint
# --------------------------------------------------
@app.post("/ask")
def ask(
    request: AskRequest,
    auth=Depends(get_api_key)
):
    return answer_question(
        question=request.question,
        domain=request.domain,
        module=request.module,
        language=request.language,
        top_k=request.top_k
    )


