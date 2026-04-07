import uuid
import fitz  # PyMuPDF
import re
from dataclasses import dataclass, field
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TextChunk:
    text: str
    page: int
    chunk_index: int
    doc_id: str
    filename: str
    chunk_type: str = "text"   # "text" | "table"
    char_start: int = 0


@dataclass
class ProcessedDocument:
    doc_id: str
    filename: str
    pages: int
    chunks: list[TextChunk] = field(default_factory=list)
    table_count: int = 0


class PDFProcessor:
    def __init__(self):
        settings = get_settings()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def process(self, file_bytes: bytes, filename: str) -> ProcessedDocument:
        doc_id = str(uuid.uuid4())
        logger.info("processing_pdf", filename=filename, doc_id=doc_id)
        pages_data = self._extract_all(file_bytes)
        chunks = self._build_chunks(pages_data, doc_id, filename)
        table_count = sum(1 for c in chunks if c.chunk_type == "table")
        logger.info("pdf_processed", filename=filename, doc_id=doc_id,
                    pages=len(pages_data), chunks=len(chunks), tables=table_count)
        return ProcessedDocument(doc_id=doc_id, filename=filename,
                                  pages=len(pages_data), chunks=chunks, table_count=table_count)

    def _extract_all(self, file_bytes: bytes) -> list[dict]:
        pages = []
        try:
            import pdfplumber, io
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    tables = []
                    for table in page.extract_tables():
                        if table:
                            fmt = self._format_table(table)
                            if fmt:
                                tables.append(fmt)
                    text = self._clean_text(page.extract_text() or "")
                    pages.append({"page": page_num, "text": text, "tables": tables})
        except ImportError:
            logger.warning("pdfplumber_not_found", msg="Falling back to PyMuPDF")
            with fitz.open(stream=file_bytes, filetype="pdf") as pdf:
                for page_num, page in enumerate(pdf, start=1):
                    text = self._clean_text(page.get_text("text"))
                    pages.append({"page": page_num, "text": text, "tables": []})
        return pages

    def _format_table(self, raw_table: list) -> str:
        if not raw_table or not any(any(cell for cell in row) for row in raw_table):
            return ""
        cleaned = [[str(cell).strip() if cell is not None else "" for cell in row] for row in raw_table]
        if not cleaned:
            return ""
        col_widths = [max((len(row[i]) if i < len(row) else 0) for row in cleaned) for i in range(len(cleaned[0]))]
        lines = []
        for i, row in enumerate(cleaned):
            padded = [(row[j] if j < len(row) else "").ljust(col_widths[j]) for j in range(len(col_widths))]
            lines.append("| " + " | ".join(padded) + " |")
            if i == 0:
                lines.append("|" + "|".join("-" * (w + 2) for w in col_widths) + "|")
        return "\n".join(lines)

    def _clean_text(self, text: str) -> str:
        text = re.sub(r"\s{3,}", "  ", text)
        text = re.sub(r"(\n\s*){3,}", "\n\n", text)
        return text.strip()

    def _build_chunks(self, pages: list[dict], doc_id: str, filename: str) -> list[TextChunk]:
        chunks = []
        chunk_index = 0
        for page_data in pages:
            page_num = page_data["page"]
            for table_text in page_data.get("tables", []):
                if table_text.strip():
                    chunks.append(TextChunk(
                        text=f"[TABLE from {filename}, Page {page_num}]\n{table_text}",
                        page=page_num, chunk_index=chunk_index,
                        doc_id=doc_id, filename=filename, chunk_type="table",
                    ))
                    chunk_index += 1
            text = page_data.get("text", "")
            if text.strip():
                for split in self.splitter.split_text(text):
                    if split.strip():
                        chunks.append(TextChunk(
                            text=split, page=page_num, chunk_index=chunk_index,
                            doc_id=doc_id, filename=filename, chunk_type="text",
                        ))
                        chunk_index += 1
        return chunks