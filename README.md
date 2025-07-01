# Self-Reflective AI System

A sophisticated AI system built with LangChain and LangGraph that implements self-reflective architecture for generating and improving responses.

## Features

### Core Components
- Multi-stage processing pipeline with automated search-augmented generation
- Self-critique mechanism for content evaluation and improvement
- Integrated LangSmith for comprehensive tracing and debugging
- Quality assurance framework for content optimization

### Technical Stack
- Python
- LangChain - LLM orchestration and prompt management
- LangGraph - Complex workflow orchestration
- Pydantic - Data validation
- LangSmith - Tracing and monitoring

## Project Structure
```
├── reflexion_agent/
│   ├── chains.py         # LLM chain definitions
│   ├── execute_tools.py  # Tool execution logic
│   ├── reflexion_graph.py# Workflow graph implementation
│   └── schema.py        # Data models and validation
└── reflection_agent/    # Basic implementation
    ├── basic.py
    └── chains.py
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install langchain langgraph pydantic python-dotenv
```

3. Set up environment variables in `.env`:
```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key
```

## Usage

The system implements a self-reflective architecture that:
1. Generates initial responses
2. Performs self-critique
3. Conducts automated research
4. Revises and improves responses

Example usage:
```python
from reflexion_agent.reflexion_graph import app
from langchain_core.messages import HumanMessage

response = app.invoke("Write about how small businesses can leverage AI")
print(response[-1].tool_calls[0]["args"]["answer"])
```

## Development

- Use LangSmith for monitoring and debugging chains
- Follow the existing pattern for adding new capabilities
- Ensure proper error handling and validation

## License

MIT