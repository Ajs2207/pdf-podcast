from typing import List
from pathlib import Path
import yaml

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from src.vectorstore.chroma_client import ChromaClient
from config.settings import FIREWORKS_API_KEY
from src.utils.memory import RedisChatMemory


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
        self.memory = RedisChatMemory()

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
        context_blocks = []

        for doc in docs:
            metadata = doc.metadata
            source = metadata.get("source", "unknown")
            page = metadata.get("page_number", "N/A")

            block = (
                f"[Source: {source} | page {page}]\n"
                f"{doc.page_content}"
            )
            context_blocks.append(block)

        return "\n\n".join(context_blocks)
    
    def _format_history(self, history):
        if not history:
            return "None"

        return "\n".join(
                f"{m['role'].capitalize()}: {m['content']}"
                for m in history
            )


    def answer(self, question: str, session_id: str):

        history = self.memory.get_history(session_id)
        formatted_history = self._format_history(history)

        docs = self.retrieve(question)
        context = self._build_context(docs)

        messages = self.prompt.format_messages(
            history=formatted_history,
            context=context,
            question=question
        )


        response = self.llm.invoke(messages)

        self.memory.append(session_id, "user", question)
        self.memory.append(session_id, "assistant", response.content)
        
        return response.content
