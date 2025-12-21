from pathlib import Path
from src.utils.pdf_loader import PDFLoader


def test_pdf_loader_extracts_text():
    sample_pdf = Path("tests/sample_data/sample.pdf")
    loader = PDFLoader(sample_pdf)

    docs = loader.load()

    assert isinstance(docs, list)
    assert len(docs) > 0
    assert "text" in docs[0]
    assert "metadata" in docs[0]


def test_pdf_loader_metadata():
    sample_pdf = Path("tests/sample_data/sample.pdf")
    loader = PDFLoader(sample_pdf)

    docs = loader.load()
    metadata = docs[0]["metadata"]

    assert "source" in metadata
    assert "page_number" in metadata
    assert "doc_id" in metadata
    assert "ingested_at" in metadata


def test_pdf_loader_doc_id_consistency():
    sample_pdf = Path("tests/sample_data/sample.pdf")
    loader = PDFLoader(sample_pdf)

    docs = loader.load()
    doc_ids = {doc["metadata"]["doc_id"] for doc in docs}

    assert len(doc_ids) == 1  # same document
