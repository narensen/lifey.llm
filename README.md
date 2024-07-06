# Gears

Gears is a chatbot application built using Streamlit, LangChain, and Groq. It leverages natural language processing to provide assistance on various self-help topics, including mental health, personal growth, relationships, and productivity.

## Features

- **Interactive Chat Interface**: Engage in conversations with the chatbot via a Streamlit interface.
- **Memory**: The chatbot remembers the last five interactions to provide context-aware responses.
- **Topic Relevance Check**: Ensures the chatbot's responses are relevant to self-help topics using sentence embeddings and cosine similarity.
- **Conversation History**: View previous interactions for better context and continuity.

## Requirements

- Python 3.x
- Streamlit
- LangChain
- Groq API
- SentenceTransformers
- NumPy
- PyTorch (for SentenceTransformers)

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/gears-self-help-companion.git
   cd gears-self-help-companion
Install Dependencies:
pip install streamlit langchain langchain_groq sentence-transformers numpy torch
Set Up Groq API Key:
Replace the groq_api_key variable in the script with your Groq API key.

Usage
Run the Streamlit App:

streamlit run gears_self_help.py

Interact with the Chatbot:

Enter your question or statement in the text area and get responses related to self-help topics.
(The Similarity Cosine has some issues so even if the prompt is off topic it might still answer)

# Initialize Streamlit app
`st.title("Gears: Your Self-Help Companion")`
`memory = ConversationBufferWindowMemory(k=5)`
   ```bash
   groq_api_key = "https://console.groq.com/keys"
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
   
   # Function to check if a question or response is related to self-help using embeddings
   def is_self_help_related(text):
       text_embedding = embedding_model.encode(text, convert_to_tensor=True)
       similarities = util.pytorch_cos_sim(text_embedding, self_help_embeddings)
       max_similarity = np.max(similarities.cpu().numpy())
       return max_similarity > 0.5  # Adjust threshold as needed
   
   # User input field
   user_question = st.text_area("How are you feeling today?")
   
   # Handle user input
   if user_question:
       response = conversation(user_question)
       ai_response = response.get("response", "No response text found")  # Adjust according to the actual response structure
       st.write("Gears:", ai_response)
   
       # Check if either the user question or the AI response is relevant to self-help topics
       if is_self_help_related(user_question) or is_self_help_related(ai_response):
           append_to_history(user_question, ai_response)
           display_message(user_question, ai_response)
       else:
           st.write("Sorry, the response is not relevant to self-help topics. Please ask another question related to mental health, personal growth, relationships, or productivity.")

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any changes or improvements.

## Acknowledgments
Built with Streamlit, LangChain, and Groq.
Sentence embeddings provided by SentenceTransformers.
Used Llama-3 for interaction


Feel free to adjust the content to better fit your specific project needs and details.
