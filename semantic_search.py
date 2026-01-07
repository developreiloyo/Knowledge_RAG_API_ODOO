import os
import psycopg2
from dotenv import load_dotenv
from openai import OpenAI
from pgvector.psycopg2 import register_vector
from pgvector import Vector

# --------------------------------------------------
# CONFIGURACIÓN
# --------------------------------------------------
EMBEDDING_MODEL = "text-embedding-3-small"
QUERY_TEXT = "¿Cómo funciona el proceso de picking en un WMS?"
DOMAIN = "wms"
LANGUAGE = "es"
TOP_K = 5

# --------------------------------------------------
# ENV
# --------------------------------------------------
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

conn = psycopg2.connect(
    dbname=os.getenv("POSTGRES_DB"),
    user=os.getenv("POSTGRES_USER"),
    password=os.getenv("POSTGRES_PASSWORD"),
    host="localhost",
    port=os.getenv("POSTGRES_PORT"),
)

conn.autocommit = True
register_vector(conn)

# --------------------------------------------------
# EMBEDDING DE LA CONSULTA
# --------------------------------------------------
response = client.embeddings.create(
    model=EMBEDDING_MODEL,
    input=QUERY_TEXT
)

query_vector = Vector(response.data[0].embedding)

# --------------------------------------------------
# BÚSQUEDA SEMÁNTICA CON FILTROS
# --------------------------------------------------
with conn.cursor() as cur:
    cur.execute(
        """
        SELECT
            c.content,
            1 - (e.embedding <=> %s) AS similarity
        FROM embeddings e
        JOIN chunks c ON c.id = e.chunk_id
        JOIN documents d ON d.id = c.document_id
        WHERE
            e.model = %s
            AND d.domain = %s
            AND d.language = %s
        ORDER BY e.embedding <=> %s
        LIMIT %s;
        """,
        (
            query_vector,
            EMBEDDING_MODEL,
            DOMAIN,
            LANGUAGE,
            query_vector,
            TOP_K,
        )
    )

    results = cur.fetchall()

# --------------------------------------------------
# OUTPUT
# --------------------------------------------------
print("\n🔎 Resultados semánticos (con filtros):\n")

if not results:
    print("⚠️ No se encontraron resultados con esos filtros.")
else:
    for content, similarity in results:
        print(f"• ({similarity:.4f}) {content}")

conn.close()


