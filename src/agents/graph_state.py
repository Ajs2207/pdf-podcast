from typing import TypedDict, List

## This is the shared memory of the graph.
class GraphState(TypedDict):
    question: str
    answer: str
    documents: List[str]
    route: str
