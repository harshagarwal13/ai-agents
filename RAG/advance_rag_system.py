from dotenv import load_dotenv
from langchain.schema import Document
from langchain_nomic import NomicEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()


#Making a Document and a retriever
embedding_function = NomicEmbeddings(model="nomic-embed-text-v1")
docs = [
    Document(
        page_content="Peak Performance Gym was founded in 2015 by former Olympic athlete Marcus Chen. With over 15 years of experience in professional athletics, Marcus established the gym to provide personalized fitness solutions for people of all levels. The gym spans 10,000 square feet and features state-of-the-art equipment.",
        metadata={"source": "about.txt"}
    ),
    Document(
        page_content="Peak Performance Gym is open Monday through Friday from 5:00 AM to 11:00 PM. On weekends, our hours are 7:00 AM to 9:00 PM. We remain closed on major national holidays. Members with Premium access can enter using their key cards 24/7, including holidays.",
        metadata={"source": "hours.txt"}
    ),
    Document(
        page_content="Our membership plans include: Basic (₹1,500/month) with access to gym floor and basic equipment; Standard (₹2,500/month) adds group classes and locker facilities; Premium (₹4,000/month) includes 24/7 access, personal training sessions, and spa facilities. We offer student and senior citizen discounts of 15% on all plans. Corporate partnerships are available for companies with 10+ employees joining.",
        metadata={"source": "membership.txt"}
    ),
    Document(
        page_content="Group fitness classes at Peak Performance Gym include Yoga (beginner, intermediate, advanced), HIIT, Zumba, Spin Cycling, CrossFit, and Pilates. Beginner classes are held every Monday and Wednesday at 6:00 PM. Intermediate and advanced classes are scheduled throughout the week. The full schedule is available on our mobile app or at the reception desk.",
        metadata={"source": "classes.txt"}
    ),
    Document(
        page_content="Personal trainers at Peak Performance Gym are all certified professionals with minimum 5 years of experience. Each new member receives a complimentary fitness assessment and one free session with a trainer. Our head trainer, Neha Kapoor, specializes in rehabilitation fitness and sports-specific training. Personal training sessions can be booked individually (₹800/session) or in packages of 10 (₹7,000) or 20 (₹13,000).",
        metadata={"source": "trainers.txt"}
    ),
    Document(
        page_content="Peak Performance Gym's facilities include a cardio zone with 30+ machines, strength training area, functional fitness space, dedicated yoga studio, spin class room, swimming pool (25m), sauna and steam rooms, juice bar, and locker rooms with shower facilities. Our equipment is replaced or upgraded every 3 years to ensure members have access to the latest fitness technology.",
        metadata={"source": "facilities.txt"}
    )
]

db = Chroma.from_documents(docs,embedding_function)
retriever = db.as_retriever(search_type = 'mmr', search_kwards = {'k':3})

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


#Making template
llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")
template = """
Answer the question based on the following context and the chat history. Especially take the latest question into consideration:
Chathistory: {history}
Context: {context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template=template)

rag_chain = prompt | llm


#Making of Agent
from typing import TypedDict, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.schema import Document
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph,END

class AgentState(TypedDict):
    messages: List[BaseMessage]
    documents: List[Document]
    on_topic: str
    rephrased_question: str
    proceed_to_generate: bool
    rephrase_count: int
    question: HumanMessage


#Making a pydantic model to grade question

class GradeQuestion(BaseModel):
    score: str = Field(
        description="Question is about the specified topics. If yes-> 'Yes' if no-> 'No'"
    )

#first_node

def question_rewriter(state: AgentState):
    print(f"Entering question rewriter with following state: {state}")

    #Reset the state variables
    state["documents"] = []
    state["on_topic"] = ""
    state["rephrase_count"] = 0
    state["rephrased_question"] = []
    state["proceed_to_generate"] = False

    if "messages" not in state or state["messages"] is None:
        state["messages"] = []
    if state["question"] not in state["messages"]:
        state["messages"].append(state["question"])
    if len(state["messages"])>1:
        conversation = state["messages"][:-1]
        current_question = state["question"].content
        messages = [
            SystemMessage(
                content="You are an helpful assistant that rephrases the user's question to be a standalone question optimised for retrieval."
            )
        ]
        messages.extend(conversation)
        messages.append(HumanMessage(content=current_question))
        rephrase_prompt = ChatPromptTemplate.from_template(messages)
        llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")
        prompt = rephrase_prompt.format()
        response = llm.invoke(prompt)
        better_question = response.content.strip()
        print(f'Question rewriter: rephrased question: {better_question}')
        state["rephrased_question"] = better_question
    else:
        state["rephrased_question"] = state["question"].content
    return state


def question_classifier(state: AgentState):
    print("Entering question classifier")
    system_message =  SystemMessage(
        content='''You are a classifier that determines whether a user's question is about one of the following topics
        1. Gym History and founder
        2. Operating Hours
        3. Membership Plans
        4. Fitness Classes
        5. Personal Trainers
        6. Facilities and Equipments
        7. Anything else about Peak Performance Gym

        If the question is about any of these topics, repond with 'Yes' otherwise, respond with 'No'.
        '''
    )
    human_message = HumanMessage(content=f"User Question: {state['rephrased_question']}")
    grade_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
    llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")

    structured_llm = llm.with_structured_output(GradeQuestion)
    grader_llm = grade_prompt | structured_llm

    result = grader_llm.invoke({})
    state["on_topic"] = result.score.strip()
    print(f"question classifier: on_topic = {state['on_topic']}")
    return state

def on_topic_router(state: AgentState):
    print("Entering on_topic_router")
    on_topic = state.get("on_topic","").strip().lower()
    if on_topic == 'yes':
        print("Routing to retriever")
        return "retrieve"
    else:
        print("Routing to off_topic_response")
        return "off_topic_response"
    
def retrieve(state: AgentState):
    print("Entering retriever")
    documents = retriever.invoke(state["rephrased_question"])
    print(f"retrieve: retrieved {len(documents)} number of documents")
    state["documents"] = documents
    return state


class GradeDocument(BaseModel):
    score: str = Field(description="Document is relevant to the question? If yes -> 'yes' if not-> 'No'")


def retrieval_grader(state: AgentState):
    print("Entering Retrieval Grader")
    system_message = SystemMessage(
        content='''You are a grader assessing the relevance of the retrieved documents to a user question
        Only answer in 'Yes' or 'No'.
        If the document contains information relevant to user's question, reposnd with 'Yes'. Otherwise, respond with 'No'.
        '''
    )

    llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")
    structured_llm = llm.with_structured_output(GradeDocument)
    relevant_documents = []
    for doc in state['documents']:
        human_message = HumanMessage(
            content=f"User Question: {state['rephrased_question']}\n\nretrieved documents: {doc.page_content}"
        )
        grade_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
        grader_llm = grade_prompt | structured_llm
        result = grader_llm.invoke({})
        print(f"Grading document : {doc.page_content[:30]}... Result: {result.score.strip()}")
        if result.score.strip().lower() == 'yes':
            relevant_documents.append(doc)
    state["documents"] = relevant_documents
    state["proceed_to_generate"] = len(relevant_documents)>0
    print(f"retrieval_grader: proceed_to_generate = {state['proceed_to_generate']}")
    return state

def proceed_router(state: AgentState):
    print("Entering proceed router")
    rephrase_count = state.get("rephrase_count",0)
    if state.get("proceed_to_generate", False):
        print("Routing to generate_answer")
        return "generate_answer"
    elif rephrase_count>=2:
        print("Maximum retries have been done but no relevant document documents")
        return "cannot_answer"
    else:
        print("routing to refine_question")
        return "refine_question"
    
def refine_question(state: AgentState):
    print("Entering refine_question")
    rephrase_count = state.get("rephrase_count", 0)
    if rephrase_count >= 2:
        print("Maximum rephrase attempts reached")
        return state
    question_to_refine = state["rephrased_question"]
    system_message = SystemMessage(
        content="""You are a helpful assistant that slightly refines the user's question to improve retrieval results.
Provide a slightly adjusted version of the question."""
    )
    human_message = HumanMessage(
        content=f"Original question: {question_to_refine}\n\nProvide a slightly refined question."
    )
    refine_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
    llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")
    prompt = refine_prompt.format()
    response = llm.invoke(prompt)
    refined_question = response.content.strip()
    print(f"refine_question: Refined question: {refined_question}")
    state["rephrased_question"] = refined_question
    state["rephrase_count"] = rephrase_count + 1
    return state

def generate_answer(state: AgentState):
    print("Entering generate_answer")
    if "messages" not in state or state["messages"] is None:
        raise ValueError("State must include 'messages' before generating an answer.")

    history = state["messages"]
    documents = state["documents"]
    rephrased_question = state["rephrased_question"]

    response = rag_chain.invoke(
        {"history": history, "context": documents, "question": rephrased_question}
    )

    generation = response.content.strip()

    state["messages"].append(AIMessage(content=generation))
    print(f"generate_answer: Generated response: {generation}")
    return state

def cannot_answer(state: AgentState):
    print("Entering cannot_answer")
    if "messages" not in state or state["messages"] is None:
        state["messages"] = []
    state["messages"].append(
        AIMessage(
            content="I'm sorry, but I cannot find the information you're looking for."
        )
    )
    return state


def off_topic_response(state: AgentState):
    print("Entering off_topic_response")
    if "messages" not in state or state["messages"] is None:
        state["messages"] = []
    state["messages"].append(AIMessage(content="I'm sorry! I cannot answer this question!"))
    return state


from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()


workflow = StateGraph(AgentState)
workflow.add_node("question_rewriter", question_rewriter)
workflow.add_node("question_classifier", question_classifier)
workflow.add_node("off_topic_response", off_topic_response)
workflow.add_node("retrieve", retrieve)
workflow.add_node("retrieval_grader", retrieval_grader)
workflow.add_node("generate_answer", generate_answer)
workflow.add_node("refine_question", refine_question)
workflow.add_node("cannot_answer", cannot_answer)

workflow.add_edge("question_rewriter", "question_classifier")
workflow.add_conditional_edges(
    "question_classifier",
    on_topic_router,
    {
        "retrieve": "retrieve",
        "off_topic_response": "off_topic_response",
    },
)
workflow.add_edge("retrieve", "retrieval_grader")
workflow.add_conditional_edges(
    "retrieval_grader",
    proceed_router,
    {
        "generate_answer": "generate_answer",
        "refine_question": "refine_question",
        "cannot_answer": "cannot_answer",
    },
)
workflow.add_edge("refine_question", "retrieve")
workflow.add_edge("generate_answer", END)
workflow.add_edge("cannot_answer", END)
workflow.add_edge("off_topic_response", END)
workflow.set_entry_point("question_rewriter")
graph = workflow.compile(checkpointer=checkpointer)

from IPython.display import Image, display
from langchain_core.runnables.graph import MermaidDrawMethod

# display(
#     Image(
#         graph.get_graph().draw_mermaid_png(
#             draw_method=MermaidDrawMethod.API,
#         )
#     )
# )

#Off Topic:
# input_data = {"question": HumanMessage(content="What does the company Apple do?")}
# result = graph.invoke(input=input_data, config={"configurable": {"thread_id": 1}})

#No docs found
# input_data = {
#     "question": HumanMessage(
#         content="What is the cancelation policy for Peak Performance Gym memberships?"
#     )
# }
# result = graph.invoke(input=input_data, config={"configurable": {"thread_id": 2}})

#Rag with History

input_data = {
    "question": HumanMessage(content="Who founded Peak Performance Gym?")
}
result = graph.invoke(input=input_data, config={"configurable": {"thread_id": 3}})
# input_data = {"question": HumanMessage(content="When did he start it?")}
# graph.invoke(input=input_data, config={"configurable": {"thread_id": 3}})

print(result)