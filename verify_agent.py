from dotenv import load_dotenv
from langchain.messages import HumanMessage
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

@tool
def square_root(x: float) -> float:
    """Calculate the square root of a number."""
    return x ** 0.5

model = ChatGoogleGenerativeAI(model='gemini-3.1-flash-lite', temperature=0)
agent = create_agent(
    model=model,
    tools=[square_root],
    system_prompt='You are an arithmetic wizard. Use your tools to calculate the square root and square of any number.'
)

response = agent.invoke({'messages': [HumanMessage(content='What is the square root of 467?')]})
print(response['messages'][-1].content)
