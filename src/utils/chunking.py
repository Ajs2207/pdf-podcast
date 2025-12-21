from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
import uuid


class Chunker:
    """
    Splits documents into overlapping chunks suitable for embedding.
    """

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150
    ):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )

    def chunk(self, documents: List[Dict]) -> List[Dict]:
        """
        Takes page-level documents and returns chunk-level documents.
        """
        chunked_docs = []

        for doc in documents:
            chunks = self.text_splitter.split_text(doc["text"])

            for idx, chunk in enumerate(chunks):
                chunked_docs.append({
                    "text": chunk,
                    "metadata": {
                        **doc["metadata"],
                        "chunk_id": str(uuid.uuid4()),
                        "chunk_index": idx
                    }
                })

        return chunked_docs
