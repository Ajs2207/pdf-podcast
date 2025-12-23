from src.agents.rag_agent import RAGAgent

rag_agent = RAGAgent(use_memory=False)

def retrieve_node(state):
    docs = rag_agent.retrieve(state["question"])
    documents = [doc.page_content for doc in docs]

    return {
        "documents": documents
    }

def generate_node(state):
    if not state["documents"]:
        return {
            "answer": "I don't know based on the provided documents."
        }

    answer = rag_agent.answer(
        question=state["question"],
        session_id="graph"
    )

    return {
        "answer": answer
    }

def fallback_node(state):
    return {
        "answer": "I don't know based on the provided documents."
    }

def route_node(state):
    if state["documents"]:
        return {"route": "generate"}
    else:
        return {"route": "fallback"}
