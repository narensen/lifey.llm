# lifey.llm

lifey.llm is a Streamlit web application that acts as a self-help companion using a language model and conversation memory to respond to user queries related to mental health, personal growth, relationships, and productivity.

## Features

- **API Integration**: Utilizes the Gemini API based on user input.
- **Self-Help Topic Detection**: Determines if user queries are related to self-help topics using embeddings and predefined keywords.
- **Conversation History**: Maintains a history of user interactions and AI responses using Streamlit session state.
- **Dynamic Response Display**: Expands AI responses for user review and interaction history exploration.

## Requirements

- Python
- Streamlit
- PyTorch
- TensorFlow
- Sentence Transformers
- Gemini API Key (required for functionality)

## Installation and Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/narensen/lifey.llm.git
   cd your-repository
Install Dependencies:
`pip install streamlit torch tensorflow sentence-transformers`
Run the Application:
`streamlit run gears.py`


## Usage
Enter Gemini API Key: Paste your Gemini API Key into the provided text input field.

`Ask a Question: Type a question related to mental health, personal growth, relationships, or productivity into the text area labeled "How are you feeling today?"` and press Enter.

`View AI Response: If the response is relevant to self-help topics (detected using embeddings and keywords), it will be displayed under an expander labeled` "Click to see AI's response".

Explore Interaction History: Use the checkbox labeled "Show Previous Interactions" to expand and review previous user inputs and AI responses.

## Project Purpose
The purpose of Gears.llm is to provide a supportive AI-based tool for users seeking information and advice on topics related to mental well-being, personal development, and relationship management. By leveraging advanced language models and conversation memory, the application aims to enhance user engagement and provide valuable insights into self-help areas.

## Acknowledgments
Built using Streamlit, PyTorch, TensorFlow, and Sentence Transformers.
Inspired by the need for accessible and responsive tools in the realm of mental health and personal growth.
