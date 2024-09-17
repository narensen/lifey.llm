# Import necessary libraries
import streamlit as st
import numpy as np
import torch
from langchain.chains import ConversationChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from sentence_transformers import SentenceTransformer, util

# Set up Streamlit app UI
st.title("Lifey")

# Check if user's question relates to self-help topics using embeddings
def is_self_help_question(question, embedding_model, self_help_embeddings):
    threshold = 0.25  # Set similarity threshold
    question_embedding = embedding_model.encode(question, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(question_embedding, self_help_embeddings)
    max_similarity = torch.max(similarities).item()
    return max_similarity > threshold  # Return True if similarity is above threshold

# Function to store conversation history
def append_to_history(user_input, ai_response):
    if 'history' not in st.session_state:
        st.session_state.history = []  # Initialize history if not present
    st.session_state.history.append({'user_input': user_input, 'ai_response': ai_response})

# Initialize memory for conversation history (last 5 interactions)
memory = ConversationBufferWindowMemory(k=5)

gemini_api_key = "GOOGLEAPI_KEY"

if gemini_api_key:
    model_name = "gemini-pro"  
    gemini_chat = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=gemini_api_key,
        temperature=0.7  # Control response creativity
    )

    # Initialize conversation chain with memory
    conversation = ConversationChain(
        llm=gemini_chat,
        memory=memory
    )

    # Load a pre-trained embedding model
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Define self-help-related keywords
    self_help_keywords = [
        "self help", "mental health", "emotional well-being", "stress management",
        "personal growth", "positive thinking", "mindfulness", "motivation",
        "goal setting", "self-care", "relationships", "productivity", "confidence",
        "break up", "positive", "girlfriend", "boyfriend", "negative",
        "feeling", "attraction", "depression", "anxiety", "loneliness", 
        "happiness", "sadness", "anger", "joy", "fear", 
        "love", "heartbreak", "grief", "stress", "anxiety", 
        "forgiveness", "compassion", "empathy", "insecurity", "confidence", 
        "well-being", "therapy", "counseling", "support", "growth",
        "resilience", "coping", "emotional intelligence", "trauma", "healing",
        "self-esteem", "self-compassion", "self-awareness", "self-discovery",
        "emotional regulation", "emotional resilience", "emotional health",
        "mental resilience", "mental well-being", "stress relief", "stress coping",
        "stress reduction", "anxiety management", "anxiety relief", "anxiety coping",
        "loneliness coping", "loneliness relief", "loneliness management",
        "happiness pursuit", "happiness boosting", "sadness coping", "anger management",
        "joyful living", "fear management", "love advice", "heartbreak recovery",
        "grief support", "stress coping strategies", "anxiety coping techniques",
        "forgiveness therapy", "compassionate living", "empathetic listening",
        "confidence building", "self-worth", "self-validation", "self-acceptance",
        "mental clarity", "mental strength", "mind-body connection", "mindfulness practice",
        "therapy sessions", "counseling support", "support groups", "growth mindset",
        "personal development", "resilience training", "coping skills", "emotional intelligence",
        "trauma recovery", "healing process", "positive psychology",
        "feeling", "felt", "thoughts", "suicide"
    ]

    # Convert self-help keywords to embeddings
    self_help_embeddings = embedding_model.encode(self_help_keywords, convert_to_tensor=True)

    # Initial system message for Lifey AI
    user_question_ = "(You are an AI-powered chatbot named Lifey or virtual assistant that leverages Gemini's natural language understanding and empathy to provide mental health and emotional support to students)"
    response = conversation(user_question_)  # Initialize conversation
    user_question = st.text_area("How are you feeling today?")  # User input

    # If user submits a question, process it
    if user_question:
        response = conversation(user_question)  # Get AI response
        ai_response = response.get("response", "No response text found")

        # Check if question is self-help related
        if is_self_help_question(user_question, embedding_model, self_help_embeddings):
            append_to_history(user_question, ai_response)  # Save to history
            with st.expander("Click to see AI's response"):  # Show AI response
                st.markdown(ai_response)
        else:
            append_to_history(user_question, ai_response)  # Store response
            with st.expander("Click to see AI's response"):  # Show AI response
                st.markdown(ai_response)

    # Option to show conversation history if available
    if 'history' in st.session_state and st.session_state.history:
        if st.checkbox("Show Previous Interactions"):
            for idx, interaction in enumerate(reversed(st.session_state.history)):  # Show in reverse order
                with st.expander(f"Interaction {idx + 1}"):  # Expandable history
                    st.markdown(f"*User Input:* {interaction['user_input']}")
                    st.markdown(f"*Gears:* {interaction['ai_response']}")
