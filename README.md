Basic Corrective RAG application

Will answer questions based on documents given. Embeds into a Chroma database,
then uses Groq and LangGraph to search the vectorstore and return and grade relevant answers.
If no relevant answers are found, Tavily web search will be used to find a relevant answer. 

QA interface is achieved using streamlit.

**SETUP**<br/>
Required API keys:<br/>
Tavily (https://tavily.com/)<br/>
Groq (https://console.groq.com/keys)

Save these in a .env file

**RUN**<br/>
Install the requirements: pip install requirements.txt<br/>
Run the UI: streamlit run app.py

