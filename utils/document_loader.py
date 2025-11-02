from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
import os


class WellnessDocumentLoader:
    """Load and process wellness guide PDFs"""

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..")) 
GUIDES_PATH = os.path.join(ROOT_DIR, "data", "guides")



    def __init__(self, guides_path: str = None, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.guides_path = guides_path or self.GUIDES_PATH
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def load_documents(self):
        """Load all PDF documents from guides directory"""
        if not os.path.exists(self.guides_path):
            os.makedirs(self.guides_path)
            print(f"Created {self.guides_path} - add PDF wellness guides here")
            return []

        loader = DirectoryLoader(
            self.guides_path,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )

        try:
            documents = loader.load()
            print(f"✓ Loaded {len(documents)} document pages")
            return documents
        except Exception as e:
            print(f"Error loading documents: {e}")
            return []

    def split_documents(self, documents):
        """Split documents into chunks"""
        if not documents:
            return []

        chunks = self.text_splitter.split_documents(documents)
        print(f"✓ Split into {len(chunks)} chunks")
        return chunks

    def process_documents(self):
        """Complete document processing pipeline"""
        docs = self.load_documents()
        return self.split_documents(docs)

