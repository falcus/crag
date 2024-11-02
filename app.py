import streamlit as st
from ARAG import custom_graph  # Import your custom_graph from ARAG.py

def process_question(question: str):
    # Create the initial state
    initial_state = {
        "question": question,
        "generation": "",
        "search": "",
        "documents": [],
        "steps": []
    }
    
    # Run the graph
    result = custom_graph.invoke(initial_state)
    return result

# Set up the Streamlit interface
st.title("RAG Question-Answering System")

# Add a text input for the question
question = st.text_input("Enter your question:", "How do the types of agent memory work?")

# Add a submit button
if st.button("Get Answer"):
    with st.spinner("Processing your question..."):
        try:
            result = process_question(question)
            
            # Display the results
            st.subheader("Question:")
            st.write(result["question"])
            
            st.subheader("Answer:")
            st.write(result["generation"])
            
            st.subheader("Steps Taken:")
            for step in result["steps"]:
                st.write(f"- {step}")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Add some helpful information
st.sidebar.markdown("""
## About
This is a RAG (Retrieval-Augmented Generation) system that can answer questions based on:
- Retrieved documents from a knowledge base
- Web search results when needed
- LLM-generated responses

The system will:
1. Retrieve relevant documents
2. Grade their relevance
3. Perform web search if needed
4. Generate a comprehensive answer
""")