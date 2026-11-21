import streamlit as st
import requests
import faiss
import numpy as np
import io
import hashlib

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# --------------------------------
# 1. Page Configuration
# --------------------------------

st.set_page_config(
    page_title="Company Document AI Assistant",
    page_icon="🤖",
    layout="wide"
)


# --------------------------------
# 2. Professional Styling
# --------------------------------

st.markdown("""
<style>
.stApp {
    background-color: #f7f9fc;
}
.block-container {
    padding-top: 2rem;
    max-width: 1100px;
}
.main-title {
    font-size: 30px;
    font-weight: 700;
    color: #183153;
}
.sub-title {
    color: #64748b;
    font-size: 15px;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------
# 3. Load Embedding Model
# --------------------------------

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


# IMPORTANT: Load model before processing PDFs
embedding_model = load_embedding_model()


# --------------------------------
# 4. Text Chunking
# --------------------------------

def create_chunks(text, chunk_size=800, overlap=120):

    text = " ".join(text.split())

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# --------------------------------
# 5. Process Uploaded PDFs
# --------------------------------

def process_pdfs(uploaded_files):

    all_chunks = []

    for uploaded_file in uploaded_files:

        try:

            pdf = PdfReader(
                io.BytesIO(uploaded_file.getvalue())
            )

            for page_number, page in enumerate(
                pdf.pages,
                start=1
            ):

                text = page.extract_text() or ""

                page_chunks = create_chunks(text)

                for chunk in page_chunks:

                    all_chunks.append({
                        "text": chunk,
                        "source": uploaded_file.name,
                        "page": page_number
                    })

        except Exception as error:

            st.warning(
                f"Could not read {uploaded_file.name}: {error}"
            )

    if not all_chunks:
        return None

    # Generate embeddings

    texts = [
        item["text"] for item in all_chunks
    ]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    # Create FAISS index

    index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    index.add(embeddings)

    return {
        "chunks": all_chunks,
        "index": index
    }


# --------------------------------
# 6. Session State
# --------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "document_data" not in st.session_state:
    st.session_state.document_data = None

if "document_hash" not in st.session_state:
    st.session_state.document_hash = ""


# --------------------------------
# 7. Sidebar - PDF Upload
# --------------------------------

with st.sidebar:

    st.title("📁 Documents")

    if st.button("➕ New Chat", use_container_width=True):

        st.session_state.messages = []

        st.rerun()

    uploaded_files = st.file_uploader(
        "Upload your PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    st.caption(
        "You can upload one or multiple PDF documents."
    )

    if st.button("🗑️ Clear Chat", use_container_width=True):

        st.session_state.messages = []

        st.rerun()


# --------------------------------
# 8. Process New Documents
# --------------------------------

if uploaded_files:

    current_hash = hashlib.sha256()

    for file in uploaded_files:

        current_hash.update(file.name.encode())
        current_hash.update(file.getvalue())

    current_hash = current_hash.hexdigest()

    if current_hash != st.session_state.document_hash:

        with st.spinner("Processing your PDF documents..."):

            document_data = process_pdfs(uploaded_files)

            st.session_state.document_data = document_data

            st.session_state.document_hash = current_hash

            st.session_state.messages = []

        if document_data:

            st.sidebar.success(
                f"{len(uploaded_files)} PDF(s) processed successfully!"
            )

        else:

            st.sidebar.warning(
                "No readable text found in the uploaded PDFs."
            )

else:

    if st.session_state.document_hash:

        st.session_state.document_data = None
        st.session_state.document_hash = ""
        st.session_state.messages = []


# --------------------------------
# 9. Main Application Interface
# --------------------------------

st.markdown(
    '<div class="main-title">🤖 Company Document AI Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Upload your company documents and ask questions in simple English.</div>',
    unsafe_allow_html=True
)

st.caption(
    "Powered by RAG • FAISS • Ollama • Qwen 2.5"
)

st.divider()


if st.session_state.document_data:

    st.success("Your documents are ready. Ask your questions!")

else:

    st.info(
        "👈 Upload a PDF document from the sidebar to get started."
    )


# --------------------------------
# 10. Display Chat History
# --------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.write(message["content"])

        if message["role"] == "assistant":

            for source in message.get("sources", []):

                with st.expander(
                    f"📄 Source: {source['source']} | Page {source['page']}"
                ):

                    st.write(source["text"])


# --------------------------------
# 11. User Question
# --------------------------------

question = st.chat_input(
    "Ask a question about your documents..."
)


# --------------------------------
# 12. RAG Pipeline
# --------------------------------

if question:

    if not st.session_state.document_data:

        st.warning(
            "Please upload a PDF document first."
        )

    else:

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):

            with st.spinner("Searching documents and generating answer..."):

                try:

                    document_data = st.session_state.document_data

                    index = document_data["index"]

                    chunks = document_data["chunks"]

                    # Convert question into embedding

                    question_embedding = embedding_model.encode(
                        [question],
                        convert_to_numpy=True,
                        normalize_embeddings=True
                    ).astype("float32")

                    # Search relevant chunks

                    k = min(4, index.ntotal)

                    scores, indices = index.search(
                        question_embedding,
                        k
                    )

                    retrieved_chunks = []

                    for idx in indices[0]:

                        if idx != -1:

                            retrieved_chunks.append(
                                chunks[idx]
                            )

                    # Prepare context

                    context = ""

                    for item in retrieved_chunks:

                        context += f"""
Source: {item['source']}
Page: {item['page']}
Content: {item['text']}

"""

                    # Prepare prompt

                    prompt = f"""
You are a helpful Company Document AI Assistant.

Answer using ONLY the provided document context.

Do not use outside knowledge.
Do not guess or invent information.

If the answer is not available in the context, say:
"The information is not available in the uploaded documents."

Keep the answer simple and clear.
Use maximum 4 bullet points.

Document Context:
{context}

User Question:
{question}

Answer:
"""

                    # Generate answer using Ollama

                    response = requests.post(
                        "http://localhost:11434/api/generate",

                        json={
                            "model": "qwen2.5:3b",
                            "prompt": prompt,
                            "stream": False,
                            "options": {
                                "temperature": 0.2,
                                "num_predict": 200
                            }
                        },

                        timeout=180
                    )

                    response.raise_for_status()

                    answer = response.json().get(
                        "response",
                        "No answer generated."
                    ).strip()

                    st.write(answer)

                    # Display source information

                    unique_sources = []

                    seen = set()

                    for item in retrieved_chunks:

                        key = (
                            item["source"],
                            item["page"]
                        )

                        if key not in seen:

                            seen.add(key)

                            unique_sources.append(item)

                    for source in unique_sources:

                        with st.expander(
                            f"📄 {source['source']} | Page {source['page']}"
                        ):

                            st.write(source["text"])

                    # Save conversation

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": unique_sources
                    })

                except requests.exceptions.RequestException:

                    st.error(
                        "Unable to connect to Ollama. "
                        "The cloud app cannot directly access Ollama "
                        "running on your personal laptop."
                    )

                except Exception as error:

                    st.error(
                        f"Something went wrong: {error}"
                    )
