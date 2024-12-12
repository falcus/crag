from tavily import TavilyClient
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
import os
from langchain.adapters.openai import convert_openai_messages
load_dotenv()

client=TavilyClient(api_key = os.getenv("TAVILY_API_KEY"))


llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        max_tokens=4096,
)

query = "What happened in the latest burning man floods?"
content = client.search(query, search_depth="advanced")["results"]

# Step 3. That's it! You've done a Tavily Search!

messages = [
        SystemMessage(content="You are an AI critical thinker research assistant. Your sole purpose is to write well written, critically acclaimed, objective and structured reports on given text."),
        HumanMessage(content=f"Based on the following search results, write a detailed report answering this question: {query}\n\nPlease use MLA format and markdown syntax. Do not include or repeat the search results in your report.\n\nSearch results: {content}")
    ]


report = llm.invoke(messages).content


print(report)