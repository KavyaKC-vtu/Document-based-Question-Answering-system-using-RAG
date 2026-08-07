import streamlit as st
import fitz  # PyMuPDF
import faiss
import numpy as np
import os

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq


load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)


st.set_page_config(page_title="Document QA System")

st.title("📄 AI-Powered Document Question Answering System")


# Chat memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# Upload multiple PDFs
uploaded_files = st.file_uploader(
    "Upload PDF files",
    type="pdf",
    accept_multiple_files=True
)


if uploaded_files:

    st.success(f"{len(uploaded_files)} PDF(s) uploaded successfully!")

    with st.spinner("📄 Processing documents..."):

        # Extract text from all PDFs
        text = ""
        chunk_metadata=[]

        text = ""
chunk_metadata = []

for uploaded_file in uploaded_files:

    pdf = fitz.open(
        stream=uploaded_file.read(),
        filetype="pdf"
    )

    for page_number, page in enumerate(pdf):

        page_text = page.get_text()

        text += page_text

        if not text.strip():
            st.error("❌ No text found in PDF")
            st.stop()


        # Text splitting
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )

        chunks = []

for uploaded_file in uploaded_files:
    uploaded_file.seek(0)
    pdf = fitz.open(
        stream=uploaded_file.read(),
        filetype="pdf"
    )

    for page_number, page in enumerate(pdf):

        page_text = page.get_text()

        page_chunks = splitter.split_text(page_text)

        for chunk in page_chunks:

            chunks.append(chunk)

            chunk_metadata.append({
                "pdf": uploaded_file.name,
                "page": page_number + 1
            })


        # Embeddings
        model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        embeddings = model.encode(chunks)

        embeddings = np.array(embeddings).astype("float32")


        # FAISS Vector Database
        index = faiss.IndexFlatL2(
            embeddings.shape[1]
        )

        index.add(embeddings)


    st.success("✅ Document processed successfully!")


    # Display information

    st.subheader("📌 Vector Database")

    st.write(
        f"Total vectors stored: {index.ntotal}"
    )


    st.subheader("🔢 Embeddings")

    st.write(
        f"Total embeddings: {len(embeddings)}"
    )

    st.write(
        f"Embedding dimension: {len(embeddings[0])}"
    )


    st.subheader("📄 Extracted Text")

    st.write(text[:1000])


    st.subheader("📚 Text Chunks")

    st.write(
        f"Total chunks: {len(chunks)}"
    )


    # Ask question

    st.subheader("💬 Ask a Question")

    query = st.text_input(
        "Enter your question"
    )


    if query:

        query_embedding = model.encode([query])

        query_embedding = np.array(
            query_embedding
        ).astype("float32")


        distance, indices = index.search(
            query_embedding,
            k=3
        )


        context = ""

        for i in indices[0]:
            context += chunks[i] + "\n"
        sources = []

        for i in indices[0]:
            sources.append(chunk_metadata[i])


        response = llm.invoke(
            f"""
Context:
{context}

Question:
{query}

Answer the question using only the context.
"""
        )


        st.subheader("🤖 AI Answer")

        st.write(response.content)
        st.subheader("📚 Sources")

        shown_sources = set()

        for source in sources:
            source_text = f"📄 {source['pdf']} | Page {source['page']}"

            if source_text not in shown_sources:
                st.write(source_text)
                shown_sources.add(source_text)


        # Save chat

        st.session_state.chat_history.append(
            {
                "question": query,
                "answer": response.content
            }
        )


    # Show history

    st.subheader("💬 Chat History")


    for chat in st.session_state.chat_history:

        st.write(
            "**You:**",
            chat["question"]
        )

        st.write(
            "**AI:**",
            chat["answer"]
        )

        st.divider()