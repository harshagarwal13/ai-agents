# from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain.agents import tool, create_react_agent
import datetime
from langchain_community.tools import TavilySearchResults
from langchain import hub
from dotenv import load_dotenv
load_dotenv()


llm = ChatGroq(model = 'meta-llama/llama-4-scout-17b-16e-instruct')
search_tool = TavilySearchResults(search_depth = 'basic')

@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """ Returns the current date and time in the specified format"""

    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime(format)
    return formatted_time

tools = [get_system_time,search_tool]

react_prompt = hub.pull("hwchase17/react")
react_agent_runnable = create_react_agent(tools=tools, llm = llm, prompt = react_prompt)