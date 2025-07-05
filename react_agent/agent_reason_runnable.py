from langchain_ollama import ChatOllama
from langchain.agents import tool, create_react_agent
import datetime
from langchain_community.tools import TavilySearchResults
from langchain import hub


llm = ChatOllama(model = 'qwen3:8b')
search_tool = TavilySearchResults(search_depth = 'basic')
@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """ Returns the current date and time in the specified format"""

    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime(format)
    return formatted_time

tools = [get_system_time, search_tool]

react_prompt = hub.pull("hwchase17/react")
react_agent_runnable = create_react_agent(tools=tools, llm = llm, prompt = react_prompt)
'''So basically react_agent_runnable is giving you 2 types of things whcih can be either AgentFinish or AgentAction.
AgentFinish is the final output of the agent, while AgentAction is an intermediate steps.'''