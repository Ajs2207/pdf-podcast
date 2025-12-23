from langgraph.graph import StateGraph, END
from src.agents.graph_state import GraphState
from src.agents.graph_nodes import (
    retrieve_node,
    route_node,
    generate_node,
    fallback_node,
)


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("route", route_node)
    graph.add_node("generate", generate_node)
    graph.add_node("fallback", fallback_node)

    graph.set_entry_point("retrieve")

    graph.add_edge("retrieve", "route")

    graph.add_conditional_edges(
        "route",
        lambda state: state["route"],
        {
            "generate": "generate",
            "fallback": "fallback",
        },
    )

    graph.add_edge("generate", END)
    graph.add_edge("fallback", END)

    return graph.compile()
