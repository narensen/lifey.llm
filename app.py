import streamlit as st
import numpy as np
import torch
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import SentenceTransformer, util

# Load CSV data
df = pd.read_csv('corpus/mhc.csv')
contexts = df['Context'].tolist()
responses = df['Response'].tolist()

st.set_page_config(page_title="Lifey LLM")

    
# Initialize Streamlit app
st.title("Lifey")

# Initialize Sentence Transformer model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode contexts
context_embeddings = embedding_model.encode(contexts, convert_to_tensor=True)

# Function to find most similar context
def find_most_similar_context(question, context_embeddings):
    question_embedding = embedding_model.encode(question, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(question_embedding, context_embeddings)
    most_similar_idx = torch.argmax(similarities).item()
    return contexts[most_similar_idx], responses[most_similar_idx], similarities[0][most_similar_idx].item()

# Initialize session state for conversation history
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# User input field for API key
gemini_api_key = "AIzaSyCM0tK3ljTw79tuMx_s4-afMxmOqNwPGRc"

if gemini_api_key:
    # Initialize Gemini chat object
    model_name = "gemini-pro"
    gemini_chat = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=gemini_api_key,
        temperature=0.7
    )

    # User input field
    user_question = st.text_area("How are you feeling today?")

    # Handle user input
    if user_question:
        similar_context, similar_response, similarity_score = find_most_similar_context(user_question, context_embeddings)
        
        # Construct prompt with similar context and response
        prompt = f"""You are an AI-powered chatbot named Lifey or virtual assistant that leverages natural language understanding and empathy to provide mental health and emotional support to students. You should not respond to any other kind of questions which are unrelated to mental health and life.

        User question: {user_question}
        Similar context from database: {similar_context}
        Suggested response: {similar_response}
        Similarity score: {similarity_score}

        Please provide a response to the user's question, taking into account the similar context and suggested response if they are relevant. If the similarity score is low, you may disregard the suggested context and response."""

        # Add user's question to conversation history
        st.session_state.conversation_history.append({"role": "user", "content": user_question})
        
        # Get AI response
        response = gemini_chat.invoke(st.session_state.conversation_history + [{"role": "user", "content": prompt}])
        ai_response = response.content

        # Add AI's response to conversation history
        st.session_state.conversation_history.append({"role": "assistant", "content": ai_response})

        # Display AI's response in a box
        st.text_area("AI's response:", value=ai_response, height=200, disabled=True)

# Show previous interactions in an expander (dropdown)
if st.session_state.conversation_history:
    with st.expander("Show Previous Interactions"):
        for idx, interaction in enumerate(st.session_state.conversation_history):
            if interaction['role'] == 'user':
                st.markdown(f"**You:** {interaction['content']}")
            else:
                st.markdown(f"**Lifey:** {interaction['content']}")
            if idx < len(st.session_state.conversation_history) - 1:
                st.markdown("---")
