from langchain_cohere import CohereEmbeddings
from utils.config import Config
from utils.document_loader import WellnessDocumentLoader
from langchain.vectorstores import Chroma
import os

class VectorStoreManager:
    """Manage vector store for wellness guides"""

    def __init__(self):
        self.embeddings = CohereEmbeddings(
            cohere_api_key=Config.COHERE_API_KEY,
            model=Config.EMBEDDING_MODEL
        )
        self.persist_directory = Config.VECTOR_STORE_PATH
        self.vectorstore = None

    def create_vectorstore(self, force_reload: bool = False):
        """Create or load existing vector store"""
        if os.path.exists(self.persist_directory) and not force_reload:
            print("📚 Loading existing vector store...")
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            return self.vectorstore

        # Load documents and create new vectorstore
        print("🔨 Creating new vector store from documents...")
        loader = WellnessDocumentLoader()
        chunks = loader.process_documents()

        if not chunks:
            print("⚠️ No documents found. Vector store will be empty.")
            print("   Add PDF guides to data/guides/ directory")
            return None

        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        print(f"✓ Vector store created with {len(chunks)} chunks")
        return self.vectorstore

    def get_retriever(self, k: int = 3):
        """Get retriever for RAG chain"""
        if not self.vectorstore:
            self.create_vectorstore()
        if not self.vectorstore:
            return None
        return self.vectorstore.as_retriever(search_kwargs={"k": k})

    def similarity_search(self, query: str, k: int = 3):
        """Direct similarity search"""
        if not self.vectorstore:
            return []
        return self.vectorstore.similarity_search(query, k=k)
