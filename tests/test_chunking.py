from src.utils.chunking import Chunker


def test_chunker_splits_text():
    documents = [{
        "text": "This is a test document. " * 200,
        "metadata": {
            "source": "test.pdf",
            "page_number": 1,
            "doc_id": "doc-123"
        }
    }]

    chunker = Chunker(chunk_size=200, chunk_overlap=50)
    chunks = chunker.chunk(documents)

    assert len(chunks) > 1
    assert "text" in chunks[0]
    assert "metadata" in chunks[0]


def test_chunk_metadata_preserved():
    documents = [{
        "text": "Some sample text " * 100,
        "metadata": {
            "source": "test.pdf",
            "page_number": 2,
            "doc_id": "doc-xyz"
        }
    }]

    chunker = Chunker()
    chunks = chunker.chunk(documents)

    metadata = chunks[0]["metadata"]
    assert metadata["source"] == "test.pdf"
    assert metadata["doc_id"] == "doc-xyz"
    assert "chunk_id" in metadata
    assert "chunk_index" in metadata


def test_no_empty_chunks():
    documents = [{
        "text": "Valid text only.",
        "metadata": {"doc_id": "doc-1"}
    }]

    chunker = Chunker(chunk_size=50, chunk_overlap=10)
    chunks = chunker.chunk(documents)

    assert all(chunk["text"].strip() for chunk in chunks)
