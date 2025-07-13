from langgraph.types import Command
from langgraph.graph import StateGraph, END
from typing import TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

memory = MemorySaver()
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
    human_response = interrupt("Where do you wnat to go at C or D. Please type C/D")
    print(human_response)

    if human_response == "C":
        return Command(
            goto = "node_c",
            update = {
                "text": state["text"]+"c"
            }
        )
    else:
        return Command(
            goto="node_d",
            update={
                "text": state["text"]+"d"
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
app = graph.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}
initialstate = {
    "text": ""
}
# first_response = app.invoke(initialstate, config=config, stream_mode="updates")

# print(first_response)

second_result = app.invoke(Command(resume="D"), config=config)
print(second_result)




