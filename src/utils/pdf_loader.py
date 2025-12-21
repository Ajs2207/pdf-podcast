from pathlib import Path
from typing import List, Dict
from pypdf import PdfReader
import uuid
from datetime import datetime, timezone


class PDFLoader:
    """
    Responsible for loading PDFs and extracting text + metadata.
    """

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.doc_id = str(uuid.uuid4())

    def load(self) -> List[Dict]:
        """
        Extracts text from each page and attaches metadata.
        Returns a list of page-level documents.
        """
        reader = PdfReader(self.file_path)
        documents = []

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text()

            if not text or not text.strip():
                continue

            documents.append({
                "text": text.strip(),
                "metadata": {
                    "source": self.file_path.name,
                    "page_number": page_number,
                    "doc_id": self.doc_id,
                    "ingested_at": datetime.now(timezone.utc).isoformat()
                }
            })

        return documents
