from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END, add_messages
from typing import TypedDict, Annotated

class State(TypedDict):
    messages: Annotated[list, add_messages]

llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")

GENERATE_POST = "generate_post"
GET_REVIEW_DECISION = "get_review_decision"
POST = "post"
COLLECT_FEEDBACK = "collect_feedback"

def generate_post(state: State):
    return {
        "messages": [llm.invoke(state["messages"])]
    }

def get_review_decision(state: State):
    post_content = state["messages"][-1].content
    print("\n Current LinkedIn Post:")
    print(post_content)
    print("\n")

    decision = input("Should be posted on the LinkedIn (yes/no)? ")

    if decision.lower() == "yes":
        return POST
    else:
        return COLLECT_FEEDBACK

def post(state: State):
    final_post = state["messages"][-1].content
    print("The final post is \n")
    print(final_post)
    print("POSTED on LinkedIn")


def collect_feedback(state: State):
    feedback = input("How can I improve the post?")

    return {
        "messages": [HumanMessage(content=feedback)]
    }

graph = StateGraph(State)
graph.add_node(GENERATE_POST, generate_post)
graph.add_node(GET_REVIEW_DECISION, get_review_decision)
graph.add_node(COLLECT_FEEDBACK, collect_feedback)
graph.add_node(POST, post)

graph.set_entry_point(GENERATE_POST)

graph.add_conditional_edges(GENERATE_POST, get_review_decision)
graph.add_edge(POST, END)
graph.add_edge(COLLECT_FEEDBACK,GENERATE_POST)

app = graph.compile()
response = app.invoke({
    "messages": [HumanMessage(content="Write me a linkedIn post on how AI-Agents are changing the world")]
})

print(response)