import os
import psycopg2
from dotenv import load_dotenv
from openai import OpenAI

# 1️⃣ Cargar variables de entorno
load_dotenv()

# 2️⃣ Cliente OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 3️⃣ Conexión a PostgreSQL
conn = psycopg2.connect(
    dbname=os.getenv("POSTGRES_DB"),
    user=os.getenv("POSTGRES_USER"),
    password=os.getenv("POSTGRES_PASSWORD"),
    host="localhost",
    port=os.getenv("POSTGRES_PORT"),
)

# ⚠️ Pega aquí un chunk_id real
CHUNK_ID = "2d8bc514-b6a1-4a0d-99ae-7f72dda8308b"

with conn.cursor() as cur:
    # 4️⃣ Obtener el texto del chunk
    cur.execute(
        "SELECT content FROM chunks WHERE id = %s;",
        (CHUNK_ID,)
    )
    row = cur.fetchone()

    if not row:
        raise ValueError("Chunk no encontrado")

    text = row[0]

    # 5️⃣ Generar embedding real
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    embedding = response.data[0].embedding  # lista de 1536 floats

    # 6️⃣ Insertar embedding en PGVector
    cur.execute(
        """
        INSERT INTO embeddings (chunk_id, embedding, model)
        VALUES (%s, %s, %s);
        """,
        (CHUNK_ID, embedding, "text-embedding-3-small")
    )

    conn.commit()

print("✅ Embedding generado y guardado correctamente")
