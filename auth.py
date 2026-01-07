from fastapi import Header, HTTPException, status
import psycopg2
import os

def get_api_key(x_api_key: str = Header(...)):
    conn = psycopg2.connect(
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        host="localhost",
        port=os.getenv("POSTGRES_PORT"),
    )

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT key, is_active, domain
                FROM api_keys
                WHERE key = %s
                """,
                (x_api_key,)
            )
            row = cur.fetchone()

        if not row:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key"
            )

        key, is_active, domain = row

        if not is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="API key disabled"
            )

        return {
            "api_key": key,
            "domain": domain
        }

    finally:
        conn.close()
