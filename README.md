# Company Document AI Assistant

An AI-powered document question-answering application that allows users to upload PDF documents and ask questions using natural language.

The application uses **Retrieval-Augmented Generation (RAG)** to retrieve relevant information from uploaded documents and generate answers using the Gemini API.

## Project Overview

Reading and searching through lengthy company documents manually can be time-consuming.

This project provides a simple solution where users can upload PDF documents, ask questions, and receive answers based on the document content, along with source page references.

## Key Features

- **PDF Upload:** Upload PDF documents through the Streamlit interface.
- **Text Extraction:** Extract text and page information from PDF files.
- **Text Chunking:** Split extracted content into smaller chunks.
- **Embeddings:** Convert text chunks into numerical representations using Sentence Transformers.
- **Vector Search:** Use FAISS to retrieve relevant document chunks.
- **AI-Powered Answers:** Generate answers using the Gemini API.
- **Source References:** Display document names and page numbers.
- **Grounded Responses:** Instruct the assistant not to invent answers when information is unavailable in the uploaded documents.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Application development |
| Streamlit | Web application interface |
| PyPDF | PDF text extraction |
| Sentence Transformers | Text embeddings |
| FAISS | Vector similarity search |
| Gemini API | Answer generation |
| NumPy | Numerical operations |
| Requests | API communication |

## How It Works

The application follows a Retrieval-Augmented Generation (RAG) workflow:

1. User uploads a PDF document.
2. Text is extracted from the PDF.
3. Extracted text is divided into smaller chunks.
4. Sentence Transformer generates embeddings.
5. FAISS stores and searches the vector representations.
6. Relevant chunks are retrieved based on the user's question.
7. Retrieved context is sent to the Gemini API.
8. Gemini generates an answer using the provided document context.
9. The application displays the answer with source references.

## Project Structure

```text
Company_Document_AI_Assistant/
│
├── Exprements/
│   └── app.py
│
├── .streamlit/
│   └── secrets.toml
│
├── requirements.txt
├── .gitignore
└── README.md
```

**Note:** The `.streamlit/secrets.toml` file contains the private API key and must not be uploaded to GitHub.

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/jeeva4100jk/Company-Document-AI-Assistant.git
```

### 2. Navigate to the Project

```bash
cd Company-Document-AI-Assistant
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Gemini API Key

Create a `.streamlit/secrets.toml` file and add:

```toml
GEMINI_API_KEY = "YOUR_API_KEY"
```

Never share or upload your API key publicly.

### 5. Run the Application

```bash
streamlit run .\Exprements\app.py
```

## Application Deployment

The application is deployed using Streamlit Community Cloud.

**GitHub Repository:**  
https://github.com/jeeva4100jk/Company-Document-AI-Assistant

## Future Enhancements

- Support for additional document formats.
- Improve retrieval accuracy.
- Add conversation memory.
- Support multiple document collections.
- Improve error handling and response performance.

## Project Objective

To build a practical AI-powered document assistant that combines document processing, semantic search, vector databases, and Large Language Models to answer questions using uploaded documents.
