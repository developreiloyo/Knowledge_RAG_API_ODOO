from pathlib import Path
import PyPDF2

def load_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def load_pdf_file(path: Path) -> str:
    text = ""
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def load_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf_file(path)
    elif suffix in [".txt", ".md"]:
        return load_text_file(path)
    else:
        raise ValueError(f"Formato no soportado: {suffix}")
