import os
import psycopg2
from typing import List, Dict
from dotenv import load_dotenv
from openai import OpenAI
from pgvector.psycopg2 import register_vector
from pgvector import Vector

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4.1-mini"

load_dotenv()

_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

def _embed(text: str) -> Vector:
    response = _client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return Vector(response.data[0].embedding)

def search(
    query_text: str,
    domain: str,
    module: str | None,
    language: str | None,
    top_k: int = 5,
    similarity_threshold: float = 0.35
) -> List[Dict]:

    query_vector = _embed(query_text)
    conn = _get_connection()

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

            params = [
                query_vector,
                EMBEDDING_MODEL,
                domain,
            ]

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
                top_k
            ])

            cur.execute(sql, params)
            rows = cur.fetchall()

        return [
            {"content": content, "similarity": float(similarity)}
            for content, similarity in rows
        ]

    finally:
        conn.close()


####
def answer_question(
    question: str,
    domain: str,
    module: str | None,
    language: str,
    top_k: int = 5
) -> Dict:

    # 🔹 Primer intento (estricto)
    results = search(
        query_text=question,
        domain=domain,
        module=module,
        language=language,
        top_k=top_k,
        similarity_threshold=0.35
    )
    mode = "strict"

    # 🔹 Fallback automático (más flexible)
    if not results:
        results = search(
            query_text=question,
            domain=domain,
            module=module,
            language=language,
            top_k=top_k,
            similarity_threshold=0.25,
            mode = "fallback"
        )

    # 🔹 Guard clause final (muy importante)
    if not results:
        return {
            "answer": "No tengo información suficiente para responder a esa pregunta.",
            "sources": []
        }

    context = "\n".join(f"- {r['content']}" for r in results)

    system_prompt = (
        "Eres un asistente experto. "
        "Responde únicamente usando la información del contexto proporcionado. "
        "Si la respuesta no está en el contexto, indícalo claramente."
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
        question=question,
        domain=domain,
        module=module,
        language=language,
        mode=mode,
        results=results
    )
    print("📊 LOGGING METRICS:", mode, len(results))


    return {
        "answer": response.choices[0].message.content.strip(),
        "sources": results
    }

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
