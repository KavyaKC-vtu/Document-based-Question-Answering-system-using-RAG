# Document-based Question Answering System using RAG

An AI-powered document Q&A system that lets you upload PDFs and ask natural-language questions about their content. Built on Retrieval-Augmented Generation (RAG) architecture.

## 🚀 Features
- Upload PDF documents and query their content conversationally
- Retrieval-Augmented Generation for accurate, grounded answers
- Fast inference using Groq's Llama 3.1
- Simple web interface built with Streamlit

## 🛠️ Tech Stack
- **LangChain** – RAG pipeline orchestration
- **Hugging Face** – embeddings
- **FAISS** – vector similarity search
- **Groq (Llama 3.1)** – LLM inference
- **Streamlit** – web UI

## ⚙️ How It Works
1. The uploaded PDF is split into chunks
2. Each chunk is embedded and stored in a FAISS vector index
3. On a user query, the most relevant chunks are retrieved
4. Retrieved context + question is passed to Llama 3.1 via Groq for a grounded answer

## 📦 Setup

```bash
git clone https://github.com/KavyaKC-vtu/Document-based-Question-Answering-system-using-RAG.git
cd Document-based-Question-Answering-system-using-RAG
pip install -r requirements.txt
```

Create a `.env` file in the project root with your Groq API key:
```
GROQ_API_KEY=your_key_here
```

Run the app:
```bash
streamlit run app.py
```
