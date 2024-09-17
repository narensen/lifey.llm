import streamlit as st
import numpy as np
import torch
import pandas as pd
from langchain.chains import ConversationChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from sentence_transformers import SentenceTransformer, util

# Load CSV data
df = pd.read_csv('/home/naren/Documents/Lifey/lifey.llm/mhc.csv')
contexts = df['Context'].tolist()
responses = df['Response'].tolist()

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

# Function to append conversation history
def append_to_history(user_input, ai_response):
    if 'history' not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append({'user_input': user_input, 'ai_response': ai_response})

# Initialize LangChain memory
memory = ConversationBufferWindowMemory(k=5)

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

    # Initialize conversation chain
    conversation = ConversationChain(
        llm=gemini_chat,
        memory=memory
    )

    # User input field
    user_question_ = "(You are an AI-powered chatbot named Lifey or virtual assistant that leverages Gemini's natural language understanding and empathy to provide mental health and emotional support to students\n you should not respond to any other kind of questions which are unrelated to mental health and life)"
    response = conversation(user_question_)
    user_question = st.text_area("How are you feeling today?")

    # Handle user input
    if user_question:
        similar_context, similar_response, similarity_score = find_most_similar_context(user_question, context_embeddings)
        
        # Construct prompt with similar context and response
        prompt = f"""User question: {user_question}
        Similar context from database: {similar_context}
        Suggested response: {similar_response}
        Similarity score: {similarity_score}

        Please provide a response to the user's question, taking into account the similar context and suggested response if they are relevant. If the similarity score is low, you may disregard the suggested context and response."""

        response = conversation(prompt)
        ai_response = response.get("response", "No response text found")

        append_to_history(user_question, ai_response)
        with st.expander("Click to see AI's response"):
            st.markdown(ai_response)

    # Show previous interactions
    if 'history' in st.session_state and st.session_state.history:
        if st.checkbox("Show Previous Interactions"):
            for idx, interaction in enumerate(reversed(st.session_state.history)):
                with st.expander(f"Interaction {idx + 1}"):
                    st.markdown(f"*User Input:* {interaction['user_input']}")
                    st.markdown(f"*Gears:* {interaction['ai_response']}")