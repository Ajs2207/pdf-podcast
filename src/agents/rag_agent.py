from typing import List
from pathlib import Path
import yaml

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from src.vectorstore.chroma_client import ChromaClient
from config.settings import FIREWORKS_API_KEY


class RAGAgent:
    def __init__(self, k: int = 4):
        self.k = k
        self.vectordb = ChromaClient()

        self.llm = ChatOpenAI(
            model="fireworks/gpt-oss-20b",
            api_key=FIREWORKS_API_KEY,
            base_url="https://api.fireworks.ai/inference/v1",
            temperature=0
        )

        self.prompt = self._load_prompt()

    def _load_prompt(self) -> ChatPromptTemplate:
        prompt_path = Path("config/prompts.yaml")

        with open(prompt_path, "r") as f:
            prompts = yaml.safe_load(f)

        rag_prompt = prompts["rag_qa"]

        return ChatPromptTemplate.from_messages([
            ("system", rag_prompt["system"]),
            ("human", rag_prompt["human"]),
        ])

    def retrieve(self, question: str):
        return self.vectordb.similarity_search(question, k=self.k)

    def _build_context(self, docs: List):
        return "\n\n".join(doc.page_content for doc in docs)

    def answer(self, question: str) -> str:
        docs = self.retrieve(question)
        context = self._build_context(docs)

        messages = self.prompt.format_messages(
            context=context,
            question=question
        )

        response = self.llm.invoke(messages)
        return response.content
