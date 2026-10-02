from dotenv import load_dotenv

load_dotenv()

from langchain.tools import tool
from typing import Dict, Any
from tavily import TavilyClient
from langchain.agents import create_agent


# -----------------------------
# Tavily Web Search
# -----------------------------

tavily_client = TavilyClient()


@tool
def web_search(query: str) -> Dict[str, Any]:
    """Search the web for information"""
    return tavily_client.search(query)


# -----------------------------
# Agent instructions
# -----------------------------

system_prompt = """

You are a personal chef. The user will give you a list of ingredients
they have left over in their house.

Using the web search tool, search the web for recipes that can be made
with the ingredients they have.

Return recipe suggestions and eventually the recipe instructions to the
user, if requested.

"""


# -----------------------------
# Create the agent with Gemini
# -----------------------------

agent = create_agent(
    model="google_genai:gemini-3.1-flash-lite",
    tools=[web_search],
    system_prompt=system_prompt
)


# -----------------------------
# First test
# -----------------------------

response = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "I have chicken, tomatoes, onions and rice. What can I cook?"
        }
    ]
})

print(response)