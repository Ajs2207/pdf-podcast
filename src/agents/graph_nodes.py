from src.agents.rag_agent import RAGAgent

rag_agent = RAGAgent(use_memory=False)

def retrieve_node(state):
    docs = rag_agent.retrieve(state["question"])
    documents = [doc.page_content for doc in docs]

    return {
        "documents": documents
    }

def rag_agent_node(state):
    if not state["documents"]:
        return {
            "answer": "I don't know based on the provided documents."
        }

    answer = rag_agent.answer(
        question=state["question"],
        session_id="graph"
    )

    return {"answer": answer}


def fallback_node(state):
    return {
        "answer": "I don't know based on the provided documents."
    }

def route_node(state):
    if state["documents"]:
        return {"route": "generate"}
    else:
        return {"route": "fallback"}
    
def intent_router_node(state):
    q = state["question"].lower()

    if "podcast" in q or "dialogue" in q or "conversation" in q:
        return {"intent": "podcast"}
    elif "image" in q or "diagram" in q or "illustration" in q:
        return {"intent": "image"}
    else:
        return {"intent": "rag"}
    

def podcast_agent_node(state):
    return {
        "answer": (
            "Host: Today we discuss transformer models.\n"
            "Guest: Transformers use attention mechanisms to understand context..."
        )
    }


def image_agent_node(state):
    return {
        "answer": "Prompt: A clean diagram showing transformer architecture with self-attention blocks."
    }


