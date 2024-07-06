import streamlit as st
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Initialize Streamlit app
st.title("gears.llm")

# Initialize LangChain memory
memory = ConversationBufferWindowMemory(k=5)

# Initialize Groq Langchain chat object with updated API key and model name
groq_api_key = ""
model_name = "llama3-70b-8192"
groq_chat = ChatGroq(
    groq_api_key=groq_api_key,
    model_name=model_name
)

# Initialize conversation chain
conversation = ConversationChain(
    llm=groq_chat,
    memory=memory
)

# Initialize Sentence Transformer model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Self-help topics
self_help_keywords = ["self help", "mental health", "emotional well-being", "stress management", 
                     "personal growth", "positive thinking", "mindfulness", "motivation", 
                     "goal setting", "self-care", "relationships", "productivity", "confidence"]

# Encode the self-help keywords
self_help_embeddings = embedding_model.encode(self_help_keywords, convert_to_tensor=True)

# Function to check if the user's question is related to self-help using embeddings
def is_self_help_question(question):
    question_embedding = embedding_model.encode(question, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(question_embedding, self_help_embeddings)
    max_similarity = np.max(similarities.numpy())
    return max_similarity > 0.5  # Adjust threshold as needed

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
    st.write(response)  # Debugging: Print out the response structure
    ai_response = response.get("text", "No response text found")  # Adjust according to the actual response structure
    if is_self_help_question(user_question):
        append_to_history(user_question, ai_response)
        display_message(user_question, ai_response)
    else:
        st.write("Sorry, the response is not relevant to self-help topics. Please ask another question related to mental health, personal growth, relationships, or productivity.")

# Show previous interactions
if 'history' in st.session_state and st.session_state.history:
    if st.checkbox("Show Previous Interactions"):
        for interaction in st.session_state.history:
            display_message(interaction['user_input'], interaction['ai_response'])
            st.write("---")
