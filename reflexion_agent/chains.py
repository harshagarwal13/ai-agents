import datetime
from langchain.prompts import MessagesPlaceholder, ChatPromptTemplate
# from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from urllib3 import response
from schema import AnswerQuestion, ReviseAnswer
import datetime
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.messages import HumanMessage
# pydantic_parser = PydanticToolsParser(tools = [AnswerQuestion, ReviseAnswer])
actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an expert AI researcher.
            Current time: {time}

            1. {first_instruction}
            2. Reflect and critique your answer by providing:
               - What is missing from the answer (required)
               - What is superfluous or could be removed (required)
            3. After the reflection, provide 1-3 search queries for researching improvements.

            Your response MUST include:
            - A detailed answer
            - A reflection with both missing and superfluous critiques
            - Search queries for improvement
            """,
        ),
        MessagesPlaceholder(variable_name='messages'),
        ("system","Answer the user's question using the required format")
    ]
).partial(time = lambda: datetime.datetime.now().isoformat())

first_responser_prompt_template = actor_prompt_template.partial(first_instruction = 'Provide a detailed ~250 word answer')

llm = ChatGroq(model = 'llama-3.1-8b-instant')
first_responder_chain = first_responser_prompt_template | llm.bind_tools([AnswerQuestion], tool_choice = 'AnswerQuestion')

validator = PydanticToolsParser(tools = [AnswerQuestion])

# response = first_responder_chain.invoke({'messages': [HumanMessage(content='Write me a blog post oh how small businesses can leverage AI to grow')]})
# print(response)

revise_instructions = '''
Revise your previous answer using the new information.
    - You should use previous critique to add important information to your answer.
        - You MUST include numerical citations in your revised answer to ensure it can be verified.
        - Add a 'References' section at the bottom of your answer (which doesnot count towards the word limit). IN form of:
            - [1] https://example.com
            - [2] https://example.com
    - You should use previous critique to remove superfluous information from your answer and make sure it is not more than 250 words.
'''

revisor_chain = actor_prompt_template.partial(first_instruction = revise_instructions) | llm.bind_tools([ReviseAnswer], tool_choice = 'ReviseAnswer')
