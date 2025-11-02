from langchain_cohere.embeddings import CohereEmbeddings
from langchain_community.vectorstores import Chroma
from document_loader import WellnessDocumentLoader  
from utils.config import Config
import os

def build_vectorstore():
    
    loader = WellnessDocumentLoader(guides_path="data/guides")
    chunks = loader.process_documents()
    
    if not chunks:
        print("No documents found. Add PDFs to data/guides/ and try again.")
        return

    
    embeddings = CohereEmbeddings(cohere_api_key=Config.COHERE_API_KEY)

    
    persist_directory = "data/vectorstore"
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    
    vectorstore.persist()
    print(f"✓ Vector store built successfully with {len(chunks)} chunks!")

if __name__ == "__main__":
    build_vectorstore()
