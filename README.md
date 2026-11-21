\# Company Document AI Assistant



\## Project Overview

The Company Document AI Assistant is a local AI-powered application that answers questions based on company documents.



Users can upload or use company policy documents and ask questions in natural language. The application retrieves relevant information from the document and generates answers using a Large Language Model (LLM).



\## Technologies Used

\- Python

\- Streamlit

\- LangChain concepts (RAG workflow)

\- Sentence Transformers

\- FAISS Vector Database

\- Ollama

\- Qwen 2.5 (3B)

\- Jupyter Notebook



\## Project Workflow

1\. PDF Text Extraction

2\. Text Preprocessing

3\. Text Chunking

4\. Text Embeddings Generation

5\. FAISS Vector Database Creation

6\. Semantic Search

7\. Ollama Model Integration

8\. Retrieval-Augmented Generation (RAG)

9\. Streamlit Web Application



\## Key Features

\- Ask questions about company documents.

\- Retrieve relevant document content using semantic search.

\- Generate answers using a locally running LLM.

\- Avoid unsupported answers when information is unavailable.

\- Simple chat-based user interface.



\## How It Works

The system converts PDF content into smaller text chunks and generates embeddings for those chunks.



FAISS stores the embeddings and helps retrieve relevant information when a user asks a question.



The retrieved content is passed to the Qwen 2.5 model through Ollama. The model generates an answer based only on the retrieved document context.



\## Local Setup

1\. Install Python.

2\. Install Ollama.

3\. Download the model using:



&#x20;  `ollama pull qwen2.5:3b`



4\. Install the required Python libraries.

5\. Run the Ollama service.

6\. Navigate to the `Exprements` folder.

7\. Start the application:



&#x20;  `streamlit run app.py`



\## Project Status

Successfully implemented and tested locally with company leave policy questions.



\## Future Improvements

\- Support multiple PDF documents.

\- Add document upload functionality.

\- Improve answer accuracy.

\- Add chat history.



