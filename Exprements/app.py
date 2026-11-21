import streamlit as st
import requests
import json
import faiss
import numpy as np

from pathlib import Path
from sentence_transformers import SentenceTransformer


# --------------------------------
# 1. Page Configuration
# --------------------------------

st.set_page_config(
    page_title="Company Document AI Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Company Document AI Assistant")

st.write(
    "Ask questions about your company documents."
)


# --------------------------------
# 2. Load Project Files
# --------------------------------

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent / "data&doc"

with open(
    DATA_DIR / "chunks.json",
    "r",
    encoding="utf-8"
) as file:
    chunks = json.load(file)

index = faiss.read_index(
    str(DATA_DIR / "faiss_index.bin")
)


# --------------------------------
# 3. Load Embedding Model
# --------------------------------

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


embedding_model = load_embedding_model()


# --------------------------------
# 4. User Question
# --------------------------------

question = st.chat_input(
    "Ask your question here..."
)


# --------------------------------
# 5. RAG Pipeline
# --------------------------------

if question:

    st.chat_message("user").write(question)

    with st.spinner("Searching company documents..."):

        # Convert question into embedding

        question_embedding = embedding_model.encode(
            [question]
        )

        question_embedding = np.array(
            question_embedding
        ).astype("float32")


        # Retrieve relevant chunks

        distances, indices = index.search(
            question_embedding,
            k=2
        )

        retrieved_context = ""

        for idx in indices[0]:

            if idx != -1:
                retrieved_context += (
                    chunks[idx] + "\n\n"
                )


        # Prepare prompt

        prompt = f"""
You are a company document assistant.

Answer using ONLY the provided context.

If the answer is not available in the context,
say:
"The information is not available in the document."

Keep the answer short and simple.
Use maximum 3 bullet points.

Context:
{retrieved_context}

Question:
{question}

Answer:
"""


        # Generate answer using Ollama

        try:

            response = requests.post(
                "http://localhost:11434/api/generate",

                json={
                    "model": "qwen2.5:3b",
                    "prompt": prompt,
                    "stream": False,

                    "options": {
                        "temperature": 0.2,
                        "num_predict": 120
                    }
                },

                timeout=180
            )

            response.raise_for_status()

            result = response.json()

            answer = result["response"]

            st.chat_message("assistant").write(
                answer
            )

        except requests.exceptions.RequestException as error:

            st.error(
                f"Unable to connect to Ollama: {error}"
            )
