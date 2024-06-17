import streamlit as st
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
import pandas as pd

# Initialize Streamlit app
st.title("Gears: Your Self-Help Companion")

# Initialize LangChain memory
memory = ConversationBufferWindowMemory(k=5)

# Initialize Groq Langchain chat object
groq_api_key = "xxxxx"
model_name = "llama2-70b-4096"
groq_chat = ChatGroq(
    groq_api_key=groq_api_key,
    model_name=model_name
)

# Initialize conversation chain
conversation = ConversationChain(
    llm=groq_chat,
    memory=memory
)

# Keywords related to self-help topics
self_help_keywords = ["self help", "mental health", "emotional well-being", "stress management", 
                     "personal growth", "positive thinking", "mindfulness", "motivation", 
                     "goal setting", "self-care", "relationships", "productivity", "confidence"]

# Function to check if the user's question is related to self-help
def is_self_help_question(question):
    for keyword in self_help_keywords:
        if keyword in question.lower():
            return True
    return False

# Function to display messages
def display_message(user_input, ai_response):
    st.write("---")  # Add a line between input and output
    st.markdown(f"*User Input:* {user_input}")
    st.markdown(f"*Gears:* {ai_response}")

# Function to append conversation history
def append_to_history(user_input, ai_response):
    if 'history' not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append({'user_input': user_input, 'ai_response': ai_response})

# User input field
user_question = st.text_area("How are you feeling today?")

# Handle user input
if user_question:
    response = conversation(user_question)
    if is_self_help_question(response['response']):
        append_to_history(user_question, response['response'])
        display_message(user_question, response['response'])
    else:
        st.write("Sorry, the response is not relevant to self-help topics. Please ask another question related to mental health, personal growth, relationships, or productivity.")

# Show previous interactions
if 'history' in st.session_state and st.session_state.history:
    if st.checkbox("Show Previous Interactions"):
        for interaction in st.session_state.history:
            display_message(interaction['user_input'], interaction['ai_response'])
            st.write("---")
