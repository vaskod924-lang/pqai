"""
Query a local PDF using Hugging Face models + FAISS retriever.
No RetrievalQA class used — manual pipeline.
Compatible with LangChain 1.1.3.
"""

import os
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate

# ---------- Configuration ----------
PDF_PATH = os.environ.get("PDF_PATH", "example.pdf")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL_ID = os.environ.get("LLM_MODEL_ID", "google/flan-t5-base")  # free, CPU-friendly

# ---------- Load PDF ----------
def load_pdf(path: str) -> List:
    if not os.path.exists(path):
        raise FileNotFoundError(f"PDF not found at: {path}")
    loader = PyPDFLoader(path)
    docs = loader.load()
    if not docs:
        raise ValueError("No pages extracted from PDF.")
    return docs

# ---------- Build vector store ----------
def build_vectorstore(docs):
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return FAISS.from_documents(docs, embeddings)

# ---------- Build LLM ----------
def build_llm():
    return HuggingFacePipeline.from_model_id(
        model_id=LLM_MODEL_ID,
        task="text2text-generation",
        model_kwargs={"max_length": 512}
    )

# ---------- Ask a question ----------
def answer_query(query: str, retriever, llm):
    # Retrieve relevant chunks
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs[:4]])

    # Build a simple prompt
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Use the context below to answer the question.\n\n"
        "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    formatted = prompt.format(context=context, question=query)

    # Run the LLM
    return llm.invoke(formatted)

# ---------- CLI entry ----------
if __name__ == "__main__":
    print("🚀 Building PDF QA pipeline (no RetrievalQA)...")
    docs = load_pdf(PDF_PATH)
    vectorstore = build_vectorstore(docs)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = build_llm()

    print(f"✅ Ready. PDF: {PDF_PATH}\n")

    queries = [
        "Summarize the main topic of the PDF.",
        "List the key points mentioned in the document.",
        "What does the document say about the introduction?"
    ]

    for q in queries:
        print(f"🟣 Query: {q}")
        answer = answer_query(q, retriever, llm)
        print(f"🟢 Answer:\n{answer}\n{'-'*60}\n")
