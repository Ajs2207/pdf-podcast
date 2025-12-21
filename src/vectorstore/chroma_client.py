from langchain_chroma import Chroma
from typing import List, Dict
from config.settings import CHROMA_DB_PATH
from src.utils.embedding import EmbeddingClient


class ChromaClient:
    def __init__(self, collection_name: str = "pdf_podcast"):
        self.embedding_client = EmbeddingClient()
        self.collection_name = collection_name

        self.vectordb = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_client.embeddings,
            persist_directory=str(CHROMA_DB_PATH)
        )

    def add_documents(self, chunks: List[Dict]):
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]

        ids = [
            f"{m['doc_id']}_{m['chunk_index']}_{m['chunk_id'][:8]}"
            for m in metadatas
        ]

        self.vectordb.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids
        )

    def similarity_search(self, query: str, k: int = 4):
        return self.vectordb.similarity_search(query, k=k)
