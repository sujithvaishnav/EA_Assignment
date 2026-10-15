# LangGraph Concepts and Architecture

## Overview of LangGraph
LangGraph is an orchestration framework by LangChain designed to build robust, stateful multi-actor applications with LLMs. Unlike linear chains (like classic LangChain Chains), LangGraph models agentic interactions as cyclical graphs. This cyclical nature is critical for agent architectures, iterative refinement, human-in-the-loop validation, and self-corrective RAG systems.

## Core Components: State, Nodes, and Edges

### 1. State
State is the central shared data structure that persists and evolves as execution flows through the graph.
- State is typically defined using Python `TypedDict` or Pydantic `BaseModel`.
- Every node receives the current state as input and returns a dictionary with updated fields.
- Reducers: LangGraph allows defining reducer functions (such as `operator.add` or custom merge functions) via typing annotations (e.g. `Annotated[list[str], operator.add]`) to specify how partial updates from nodes are combined into the global state.

```python
from typing import TypedDict, Annotated, List
import operator

class AgentState(TypedDict):
    question: str
    documents: List[str]
    generation: str
    revision_count: Annotated[int, operator.add]
```

### 2. Nodes
Nodes represent discrete Python functions or runnables that perform work.
- Each node takes the state as its argument and returns a dictionary of keys to update in the state.
- Special nodes:
  - `START`: The designated entry point of graph execution.
  - `END`: A terminal node signaling the conclusion of the workflow.

```python
def retrieve_node(state: AgentState) -> dict:
    query = state["question"]
    docs = retriever.invoke(query)
    return {"documents": docs}
```

### 3. Edges
Edges define the control flow transitions between nodes.
- **Normal Edges**: Direct, unconditional transitions from one node to another (e.g., `graph.add_edge("node_a", "node_b")`).
- **Conditional Edges**: Dynamic routing decisions based on the outcome of a routing function.
  - The routing function inspects the state and returns the name of the next node to visit.
  - Crucial for self-correction: determining whether retrieved documents are relevant, or whether an answer needs web search or regeneration.

```python
def route_documents(state: AgentState) -> str:
    if not state.get("documents"):
        return "web_search"
    return "generate"

graph.add_conditional_edges(
    "grade_documents",
    route_documents,
    {
        "web_search": "web_search_node",
        "generate": "generate_answer_node"
    }
)
```

## StateGraph Compilation and Execution
To execute a graph in LangGraph:
1. Initialize `StateGraph(AgentState)` with your state schema.
2. Add nodes via `graph.add_node("name", function)`.
3. Connect nodes using `graph.add_edge()` and `graph.add_conditional_edges()`.
4. Compile the graph using `app = graph.compile(checkpointer=memory)`.
5. Invoke with initial state: `result = app.invoke({"question": "Explain LangGraph"})`.

## Persistence and Memory (Checkpointers)
LangGraph provides built-in state persistence through checkpointers:
- `MemorySaver`: In-memory checkpointing for thread-level state and conversational continuity.
- `SqliteSaver` / `PostgresSaver`: Persistent storage for production environments.
- Threads: Execution runs can be partitioned by a `thread_id` in the `configurable` dictionary, enabling multi-session chat histories and time-travel debugging.

## Self-Corrective RAG Pattern in LangGraph
In self-corrective RAG (CRAG / Corrective RAG):
1. **Retrieve Node**: Queries vector database for top-k document chunks.
2. **Grade Documents Node**: LLM assesses whether each document is relevant to the user query.
3. **Decide Route (Conditional Edge)**:
   - If relevant documents exist, proceed to Generate.
   - If documents are irrelevant or insufficient, route to Query Rewrite / Web Search.
4. **Generate Node**: Produces the final answer using retrieved context.
5. **Hallucination / Answer Quality Check**: LLM checks if generation is grounded in retrieved facts and directly answers the question.
