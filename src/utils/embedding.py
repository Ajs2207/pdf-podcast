from typing import List
from langchain_openai import OpenAIEmbeddings
from config.settings import FIREWORKS_API_KEY


class EmbeddingClient:
    def __init__(
        self,
        model: str = "fireworks/qwen3-embedding-8b"
    ):
        self.embeddings = OpenAIEmbeddings(
            model=model,
            api_key=FIREWORKS_API_KEY,
            base_url="https://api.fireworks.ai/inference/v1"
        )

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return self.embeddings.embed_documents(texts)

    def embed_query(self, query: str) -> List[float]:
        return self.embeddings.embed_query(query)
