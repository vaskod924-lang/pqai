"""
Test script for verifying all imports used in agent.py (LangChain 1.1.3 + community).
Run: python test_imports.py
"""

def main():
    try:
        from langchain_openai import ChatOpenAI
        print("✅ ChatOpenAI imported")

        from langchain.agents import create_agent, AgentState
        print("✅ create_agent and AgentState imported")

        from langchain.tools import tool
        print("✅ tool decorator imported")

        from langchain_core.prompts import ChatPromptTemplate
        print("✅ ChatPromptTemplate imported")

        from langchain_core.runnables import Runnable
        print("✅ Runnable imported")

        from langchain_core.tools import Tool
        print("✅ Tool imported")

        from langchain_core.messages import AIMessage, HumanMessage
        print("✅ AIMessage and HumanMessage imported")

        from langchain_core.output_parsers import StrOutputParser
        print("✅ StrOutputParser imported")

        from langchain_community.document_loaders import PyPDFLoader
        print("✅ PyPDFLoader imported")

        from langchain_community.vectorstores import FAISS
        print("✅ FAISS imported")

        from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
        print("✅ HuggingFaceEmbeddings and HuggingFacePipeline imported")

        # RetrievalQA location in v1.1.3 + community 0.4.x
        try:
            from langchain.chains.question_answering import RetrievalQA
            print("✅ RetrievalQA imported from langchain.chains")
        except ImportError:
            print("❌ RetrievalQA not found in langchain_community.qa.retrieval")

        print("\n🎉 All agent.py imports tested!")

    except ImportError as e:
        print(f"❌ Import failed: {e}")

if __name__ == "__main__":
    main()
