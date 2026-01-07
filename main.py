from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

from services.retrieval_service import answer_question

# --------------------------------------------------
# App
# --------------------------------------------------
app = FastAPI(
    title="Knowledge Retrieval API",
    version="1.0.0",
    description="API RAG para consultas semánticas sobre conocimiento empresarial"
)

# --------------------------------------------------
# Schemas
# --------------------------------------------------
class AskRequest(BaseModel):
    question: str
    domain: str
    module: Optional[str] = None
    language: Optional[str] = "en"
    top_k: int = 5

class Source(BaseModel):
    content: str
    similarity: float

class AskResponse(BaseModel):
    answer: str
    sources: List[Source]

# --------------------------------------------------
# Endpoint
# --------------------------------------------------
@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """
    Endpoint principal de preguntas.

    Flujo:
    - Valida input
    - Llama al motor RAG
    - Devuelve respuesta + fuentes
    """

    result = answer_question(
        question=request.question,
        domain=request.domain,
        module=request.module,
        language=request.language,
        top_k=request.top_k
    )

    return result


