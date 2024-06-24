# Self-Help Language Model (LLM)

## Project Overview
The Self-Help Language Model (LLM) is a specialized large language model designed to provide targeted responses on self-help topics. Initially focused on university-related content, it has now shifted to self-help themes in response to the evolving project goals.

## Features
- **API**: Utilizes the Llama-2 G API for generating responses.
- **GUI**: Implemented using Streamlit for a user-friendly interface.
- **Language Chain**: Incorporates Python's language processing capabilities.
- **Keyword Filtering**: Enhances responses by filtering out profane language and racial slurs.
- **Topic Relevance Check**: Ensures responses are relevant by checking keywords in both the question and generated answer. If no relevant keywords are found, a predefined message indicates the answer is off-topic.

## How It Works
When a user requests an answer from Llama-2 G on a self-help topic:
- The API generates a response.
- Python scripts filter out inappropriate language.
- Keywords specific to self-help topics are checked in both the user's question and the API's answer.
- If the response aligns with the topic, it is displayed; otherwise, a message indicates the response is off-topic.

## Updates
Updated on May 20, 2024, to enhance API integration and improve response quality for better user experience.

## Technologies Used
- **API**: Llama-2 G
- **GUI**: Streamlit
- **Language Processing**: Python with custom keyword filters

## Future Development
Future updates may include expanding the keyword database for more nuanced topic detection and refining the filtering mechanisms to improve response accuracy and relevance.

