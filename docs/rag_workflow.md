flowchart TD
    A[PDF Upload] --> B[PDF Text Extraction]
    B --> C[Text Chunking]
    C --> D[Sentence Transformer]
    D --> E[Generate Embeddings]
    E --> F[FAISS Vector Index]

    G[User Question] --> H[Question Embedding]
    H --> I[FAISS Similarity Search]
    F --> I

    I --> J[Retrieve Relevant Chunks]
    J --> K[Build Prompt with Context]
    K --> L[Gemini API]
    L --> M[Generate Answer]
    M --> N[Answer with Source Pages]
