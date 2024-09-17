import streamlit as st
import numpy as np
import torch
from langchain.chains import ConversationChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from sentence_transformers import SentenceTransformer, util

# Initialize Streamlit app
st.title("Lifey")

# Function to check if the user's question is related to self-help using embeddings
def is_self_help_question(question, embedding_model, self_help_embeddings):
    threshold = 0.25  # Adjusted threshold
    question_embedding = embedding_model.encode(question, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(question_embedding, self_help_embeddings)
    max_similarity = torch.max(similarities).item()
    return max_similarity > threshold

# Function to append conversation history
def append_to_history(user_input, ai_response):
    if 'history' not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append({'user_input': user_input, 'ai_response': ai_response})

# Initialize LangChain memory
memory = ConversationBufferWindowMemory(k=5)

# User input field for API key
#st.subheader("Enter your Gemini API Key:")
gemini_api_key = #Enter your API KEY

if gemini_api_key:
    # Initialize Gemini chat object
    model_name = "gemini-pro"  # Use the appropriate Gemini model name
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

    # Initialize Sentence Transformer model
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Self-help topics
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

    # Encode the self-help keywords
    self_help_embeddings = embedding_model.encode(self_help_keywords, convert_to_tensor=True)

    # User input field
    user_question_ = "(You are an AI-powered chatbot named Lifey or virtual assistant that leverages Gemini's natural language understanding and empathy to provide mental health and emotional support to students)"
    response = conversation(user_question_)
    user_question = st.text_area("How are you feeling today?")

    # Handle user input
    if user_question:
        response = conversation(user_question)
        ai_response = response.get("response", "No response text found")

        if is_self_help_question(user_question, embedding_model, self_help_embeddings):
            append_to_history(user_question, ai_response)
            with st.expander("Click to see AI's response"):
                st.markdown(ai_response)
        else:     
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
