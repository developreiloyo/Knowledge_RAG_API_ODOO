# 📘 Knowledge Retrieval API  
### Backend RAG (Retrieval-Augmented Generation)

---

## 1️⃣ Visión General

La **Knowledge Retrieval API** es un backend de tipo **RAG (Retrieval-Augmented Generation)** diseñado para responder preguntas utilizando exclusivamente conocimiento previamente ingerido y almacenado en una base vectorial.

El sistema:
- ❌ NO entrena modelos
- ❌ NO inventa respuestas
- ✅ SOLO responde con evidencia existente
- ✅ Optimizado para bajo costo de tokens y alta precisión

---

## 2️⃣ Arquitectura General
Cliente
│
▼
FastAPI (/ask)
│
▼
Retrieval Service
│
├─ PostgreSQL + pgvector (similarity search)
├─ Filtros (domain, module, language)
├─ Threshold dinámico + fallback
│
▼
LLM (OpenAI Chat)
│
▼
Respuesta final + fuentes
---

## 3️⃣ Tecnologías Utilizadas

| Capa | Tecnología |
|----|----|
| API | FastAPI |
| DB | PostgreSQL 16 |
| Vector DB | pgvector |
| Embeddings | `text-embedding-3-small` |
| LLM | `gpt-4.1-mini` |
| DB Driver | psycopg2 |
| Infraestructura | Docker |
| Parsing | Python scripts |
| Métricas | PostgreSQL |

---

## 4️⃣ Modelo de Datos

### 📄 `documents`
Documentos originales ingeridos.

| Campo | Tipo |
|----|----|
| id | UUID |
| title | TEXT |
| domain | TEXT |
| module | TEXT |
| language | TEXT |
| source | TEXT |
| created_at | TIMESTAMP |

---

### 🧩 `chunks`
Fragmentos de texto derivados de documentos.

| Campo | Tipo |
|----|----|
| id | UUID |
| document_id | UUID |
| content | TEXT |

---

### 🧠 `embeddings`
Vectores asociados a cada fragmento.

| Campo | Tipo |
|----|----|
| id | UUID |
| chunk_id | UUID |
| embedding | VECTOR(1536) |
| model | TEXT |
| created_at | TIMESTAMP |

---

### 📊 `query_metrics`
Métricas de uso y calidad del sistema.

| Campo | Tipo |
|----|----|
| question | TEXT |
| domain | TEXT |
| module | TEXT |
| language | TEXT |
| mode | strict / fallback |
| similarity_avg | FLOAT |
| results_count | INT |
| created_at | TIMESTAMP |

---

## 5️⃣ Flujo de Ingesta

### 📥 Ingesta manual y controlada

La ingesta se realiza mediante scripts Python:

```bash
python -m ingest.ingest_folder
