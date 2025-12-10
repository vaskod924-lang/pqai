"""
Agent that uses open-source Hugging Face models and can query a local PDF.
LangChain 1.1.3 compatible.
"""

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

# 1. Load and embed PDF
pdf_path = "USAW036A0N25.pdf"  # replace with your local PDF
loader = PyPDFLoader(pdf_path)
docs = loader.load()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()

# 2. Define a retrieval tool
@tool
def query_pdf(query: str) -> str:
    """Query the local PDF for relevant information."""
    results = retriever.get_relevant_documents(query)
    return "\n".join([doc.page_content for doc in results[:3]])

# 3. Initialize an open-source LLM
llm = HuggingFacePipeline.from_model_id(
    model_id="google/flan-t5-base",  # free model
    task="text2text-generation",
    model_kwargs={"max_length": 512}
)

# 4. Create the agent
agent = create_agent(llm, [query_pdf])

# 5. Run the agent
if __name__ == "__main__":
    print("🚀 Agent ready. Querying PDF...\n")
    result = agent.invoke({"input": "Summarize the main topic of the PDF."})
    print("✅ Agent response:")
    print(result)
