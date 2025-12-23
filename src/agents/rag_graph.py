from langgraph.graph import StateGraph, END
from src.agents.graph_state import GraphState
from src.agents.graph_nodes import (
    retrieve_node,
    route_node,
    intent_router_node,
    rag_agent_node,
    podcast_agent_node,
    image_agent_node,
    fallback_node,
    error_handler_node
)


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("route_docs", route_node)
    graph.add_node("intent_router", intent_router_node)

    graph.add_node("rag", rag_agent_node)
    graph.add_node("podcast", podcast_agent_node)
    graph.add_node("image", image_agent_node)
    graph.add_node("fallback", fallback_node)
    graph.add_node("error_handler", error_handler_node)


    graph.set_entry_point("retrieve")

    graph.add_edge("retrieve", "route_docs")

    graph.add_conditional_edges(
        "route_docs",
        lambda state: state["route"],
        {
            "generate": "intent_router",
            "fallback": "fallback",
        },
    )

    graph.add_conditional_edges(
        "intent_router",
        lambda state: state["intent"],
        {
            "rag": "rag",
            "podcast": "podcast",
            "image": "image",
        },
    )

    # graph.add_edge("rag", END)
    graph.add_conditional_edges(
    "rag",
    lambda state: "error" if state.get("error") else "ok",
    {
        "ok": END,
        "error": "error_handler",
    },
)

    graph.add_edge("error_handler", END)

    graph.add_edge("podcast", END)
    graph.add_edge("image", END)
    graph.add_edge("fallback", END)

    return graph.compile()
