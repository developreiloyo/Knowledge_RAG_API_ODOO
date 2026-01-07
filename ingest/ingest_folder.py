import os
from pathlib import Path
from dotenv import load_dotenv
import psycopg2
from openai import OpenAI
from pgvector.psycopg2 import register_vector
from pgvector import Vector

from ingest.loaders import load_document


# ---------------- CONFIG ----------------
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "text-embedding-3-small"

# ----------------------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def split_text(text: str):
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start = end - CHUNK_OVERLAP
    return chunks

def get_connection():
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

def ingest_file(path: Path, domain: str, module: str, language: str):
    print(f"📄 Procesando: {path.name}")

    text = load_document(path)
    clean_text = text.strip()

    if len(clean_text) < 300:
        print(f"❌ Documento ignorado por poco texto útil: {path.name}")
        return
    
    chunks = split_text(clean_text)

    conn = get_connection()
    cur = conn.cursor()

    # 1️⃣ Insert document
    cur.execute(
        """
        INSERT INTO documents (title, source, domain, module, language)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id;
        """,
        (path.stem, str(path), domain, module, language)
    )
    document_id = cur.fetchone()[0]

    # 2️⃣ Insert chunks + embeddings
    for chunk in chunks:
        cur.execute(
            """
            INSERT INTO chunks (document_id, content)
            VALUES (%s, %s)
            RETURNING id;
            """,
            (document_id, chunk)
        )
        chunk_id = cur.fetchone()[0]

        embedding = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=chunk
        ).data[0].embedding

        cur.execute(
            """
            INSERT INTO embeddings (chunk_id, embedding, model)
            VALUES (%s, %s, %s);
            """,
            (chunk_id, Vector(embedding), EMBEDDING_MODEL)
        )

    cur.close()
    conn.close()
    print(f"✅ Ingesta completada: {path.name}")

def ingest_folder(base_folder: Path, domain: str, language: str):
    for module_dir in base_folder.iterdir():
        if module_dir.is_dir():
            module = module_dir.name
            for file in module_dir.iterdir():
                if file.is_file():
                    ingest_file(file, domain, module, language)


if __name__ == "__main__":
    ingest_folder(
        base_folder=Path("data/input/odoo"),
        domain="odoo",
        language="en"
    )
