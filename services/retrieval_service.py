import os
import psycopg2
import math
import hashlib
from typing import List, Dict
from dotenv import load_dotenv
from openai import OpenAI
from pgvector.psycopg2 import register_vector
from pgvector import Vector

# --------------------------------------------------
# Configuración
# --------------------------------------------------
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4.1-mini"

load_dotenv()
_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --------------------------------------------------
# DB helpers
# --------------------------------------------------
def _get_connection():
    conn = psycopg2.connect(
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        host="localhost",
        port=os.getenv("POSTGRES_PORT"),
    )
    conn.autocommit = True
    register_vector(conn)
    return conn

# --------------------------------------------------
# Embedding
# --------------------------------------------------
def _embed(text: str) -> Vector:
    response = _client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return Vector(response.data[0].embedding)

# --------------------------------------------------
# Deduplicación + reranking
# --------------------------------------------------
def _normalize(text: str) -> str:
    return " ".join(text.lower().split())

def _hash(text: str) -> str:
    return hashlib.sha1(_normalize(text).encode()).hexdigest()

def _deduplicate(results: List[Dict]) -> List[Dict]:
    seen = {}
    for r in results:
        h = _hash(r["content"])
        if h not in seen or r["similarity"] > seen[h]["similarity"]:
            seen[h] = r
    return list(seen.values())

def _rerank(results: List[Dict], top_k: int) -> List[Dict]:
    ranked = []
    for r in results:
        length_score = math.log(max(len(r["content"]), 50))
        score = r["similarity"] * length_score
        ranked.append({**r, "_score": score})

    ranked.sort(key=lambda x: x["_score"], reverse=True)
    return ranked[:top_k]

# --------------------------------------------------
# Search (solo retrieval)
# --------------------------------------------------
def search(
    query_text: str,
    domain: str,
    module: str | None,
    language: str | None,
    top_k: int,
    similarity_threshold: float
) -> List[Dict]:

    query_vector = _embed(query_text)
    conn = _get_connection()

    SQL_LIMIT = max(top_k * 3, 10)

    try:
        with conn.cursor() as cur:
            sql = """
                SELECT
                    c.content,
                    1 - (e.embedding <=> %s) AS similarity
                FROM embeddings e
                JOIN chunks c ON c.id = e.chunk_id
                JOIN documents d ON d.id = c.document_id
                WHERE
                    e.model = %s
                    AND d.domain = %s
            """

            params = [query_vector, EMBEDDING_MODEL, domain]

            if module:
                sql += " AND d.module = %s"
                params.append(module)

            if language:
                sql += " AND d.language = %s"
                params.append(language)

            sql += """
                AND (1 - (e.embedding <=> %s)) >= %s
                ORDER BY e.embedding <=> %s
                LIMIT %s;
            """

            params.extend([
                query_vector,
                similarity_threshold,
                query_vector,
                SQL_LIMIT
            ])

            cur.execute(sql, params)
            rows = cur.fetchall()

        raw_results = [
            {"content": c, "similarity": float(s)}
            for c, s in rows
        ]

        deduped = _deduplicate(raw_results)
        reranked = _rerank(deduped, top_k)

        return reranked

    finally:
        conn.close()

# --------------------------------------------------
# Answering con citas numeradas
# --------------------------------------------------
def answer_question(
    question: str,
    domain: str,
    module: str | None,
    language: str,
    top_k: int = 5
) -> Dict:

    results = search(
        question, domain, module, language, top_k, similarity_threshold=0.35
    )
    mode = "strict"

    if not results:
        results = search(
            question, domain, module, language, top_k, similarity_threshold=0.25
        )
        mode = "fallback"

    if not results:
        return {
            "answer": "No tengo información suficiente para responder a esa pregunta.",
            "sources": []
        }

    # 🔢 Numerar fuentes
    numbered_results = [
        {**r, "ref_id": i + 1}
        for i, r in enumerate(results)
    ]

    # 📚 Contexto con referencias
    context = "\n".join(
        f"[{r['ref_id']}] {r['content']}"
        for r in numbered_results
    )

    system_prompt = (
        "Eres un asistente experto. "
        "Responde únicamente usando el contexto proporcionado. "
        "Cada afirmación DEBE incluir una cita en formato [n]. "
        "No inventes información. "
        "Si la respuesta no está en el contexto, responde exactamente: "
        "'No tengo información suficiente para responder a esa pregunta.'"
    )

    user_prompt = f"""
Contexto:
{context}

Pregunta:
{question}
"""

    response = _client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    _log_metrics(
        question, domain, module, language, mode, numbered_results
    )

    print("📊 LOGGING METRICS:", mode, len(numbered_results))

    return {
        "answer": response.choices[0].message.content.strip(),
        "sources": numbered_results
    }

# --------------------------------------------------
# Metrics
# --------------------------------------------------
def _log_metrics(
    question: str,
    domain: str,
    module: str | None,
    language: str,
    mode: str,
    results: List[Dict]
):
    similarity_avg = (
        sum(r["similarity"] for r in results) / len(results)
        if results else 0.0
    )

    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO query_metrics (
                    question, domain, module, language,
                    mode, similarity_avg, results_count
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s);
                """,
                (
                    question,
                    domain,
                    module,
                    language,
                    mode,
                    similarity_avg,
                    len(results),
                )
            )
    finally:
        conn.close()


