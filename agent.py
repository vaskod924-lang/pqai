"""
Query a local PDF using an open-source Hugging Face model + FAISS via RetrievalQA.
Compatible with LangChain 1.1.3 (no proprietary APIs).

"""

import os
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.chains.qa import RetrievalQA

# ---------- Configuration ----------
PDF_PATH = os.environ.get("PDF_PATH", "example.pdf")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL_ID = os.environ.get("LLM_MODEL_ID", "google/flan-t5-base")  # small, free, CPU-friendly

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
    # Flan-T5 is a seq2seq model; HuggingFacePipeline wraps it for generation
    return HuggingFacePipeline.from_model_id(
        model_id=LLM_MODEL_ID,
        task="text2text-generation",
        model_kwargs={"max_length": 512}
    )

# ---------- Build RetrievalQA ----------
def build_qa_chain(pdf_path: str) -> RetrievalQA:
    docs = load_pdf(pdf_path)
    vectorstore = build_vectorstore(docs)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = build_llm()
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
    return qa

# ---------- CLI entry ----------
if __name__ == "__main__":
    print("🚀 Building RetrievalQA pipeline with open-source components...")
    qa = build_qa_chain(PDF_PATH)
    print(f"✅ Ready. PDF: {PDF_PATH}\n")

    # Demo queries
    queries = [
        "Summarize the main topic of the PDF.",
        "List the key points mentioned in the document.",
        "What does the document say about the introduction?"
    ]

    for q in queries:
        print(f"🟣 Query: {q}")
        answer = qa.run(q)
        print(f"🟢 Answer:\n{answer}\n{'-'*60}\n")
