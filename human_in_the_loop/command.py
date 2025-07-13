from langgraph.types import Command
from langgraph.graph import StateGraph, END
from typing import TypedDict



class BaseGraph(TypedDict):
    text: str

def node_a(state: BaseGraph):
    print("Node A")
    return Command(
        goto="node_b",
        update={
            "text": state["text"]+"a"
        }
    )

def node_b(state: BaseGraph):
    print("node_b")
    return Command(
        goto = "node_c",
        update = {
            "text": state["text"]+"b"
        }
    )

def node_c(state: BaseGraph):
    print("node_c")
    return Command(
        goto="node_d",
        update={
            "text": state["text"]+"c"
        }
    )

def node_d(state: BaseGraph):
    print("node_d")
    return Command(
        goto=END,
        update={
            "text": state["text"]+"d"
        }
    )

graph = StateGraph(BaseGraph)

graph.add_node("node_a",node_a)
graph.add_node("node_b",node_b)
graph.add_node("node_c",node_c)
graph.add_node("node_d",node_d)

graph.set_entry_point("node_a")
app = graph.compile()

response = app.invoke({
    "text": ""
})

print(response)




