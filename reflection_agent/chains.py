from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_ollama import ChatOllama


generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            '''You are an twitter techie influencer assistant tasked with writing excellent twitter posts",
            "Generate the best twitter post for the user's request",
            "If the user provide critique, respond with a revised versionof your previous attempts.''',
        ),
        MessagesPlaceholder(variable_name='messages'),
    ]
)

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            '''You are an viral twitter influencer grading a tweet. Generate critique and recommendation for the user's tweet."
            "Always provide detailed recommendation, including request for length, virality, style, etc.''',
        ),
        MessagesPlaceholder(variable_name='messages'),
    ]
)

# llm = OllamaChatModel(model='gemma3:27b')
llm = ChatOllama(model = 'qwen3:8b')

generation_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm