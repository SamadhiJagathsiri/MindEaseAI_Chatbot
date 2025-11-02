import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import Chroma
from utils.config import Config

class VectorStoreManager:
    """Manages vector store creation and operations for wellness guide PDFs"""
    
    def __init__(self, guides_dir: str = "data/guides", persist_dir: str = "./chroma_db"):
        """
        Initialize VectorStoreManager
        
        Args:
            guides_dir: Directory containing PDF guides
            persist_dir: Directory to persist vector store
        """
        self.guides_dir = Path(guides_dir)
        self.persist_dir = Path(persist_dir)
        self.vectorstore = None
        self.embeddings = None
        
        print(f"📁 Guides directory: {self.guides_dir.absolute()}")
        print(f"📁 Persist directory: {self.persist_dir.absolute()}")
        
       
        try:
            self.embeddings = CohereEmbeddings(
                cohere_api_key=Config.COHERE_API_KEY,
                model="embed-english-v3.0"
            )
            print(" Cohere embeddings initialized")
        except Exception as e:
            print(f" Failed to initialize embeddings: {e}")
            raise
    
    def _get_pdf_files(self):
        """Get all PDF files from guides directory"""
        
        
        if not self.guides_dir.exists():
            print(f"⚠️ Creating guides directory: {self.guides_dir}")
            self.guides_dir.mkdir(parents=True, exist_ok=True)
            return []
        
        
        pdf_files = list(self.guides_dir.glob("*.pdf"))
        
        print(f"📄 Found {len(pdf_files)} PDF files in {self.guides_dir}")
        for pdf in pdf_files:
            print(f"  - {pdf.name}")
        
        return pdf_files
    
    def _load_documents(self, pdf_files):
        """Load and split PDF documents"""
        
        if not pdf_files:
            print("⚠️ No PDF files to load")
            return []
        
        all_documents = []
        
        for pdf_path in pdf_files:
            try:
                print(f" Loading: {pdf_path.name}")
                
                
                loader = PyPDFLoader(str(pdf_path))
                documents = loader.load()
                
                print(f"  Loaded {len(documents)} pages from {pdf_path.name}")
               
                for doc in documents:
                    doc.metadata['source'] = pdf_path.name
                
                all_documents.extend(documents)
                
            except Exception as e:
                print(f"  ❌ Error loading {pdf_path.name}: {e}")
                continue
        
        if not all_documents:
            print("❌ No documents loaded successfully")
            return []
        
        print(f"✓ Total documents loaded: {len(all_documents)}")
        
        
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            splits = text_splitter.split_documents(all_documents)
            print(f"✓ Split into {len(splits)} chunks")
            
            return splits
            
        except Exception as e:
            print(f" Error splitting documents: {e}")
            return []
    
    def create_vectorstore(self, force_rebuild: bool = False):
        """
        Create or load vector store
        
        Args:
            force_rebuild: Force rebuild even if persisted store exists
        """
        
        
        if not force_rebuild and self.persist_dir.exists() and list(self.persist_dir.glob("*")):
            try:
                print(f"📦 Loading existing vectorstore from {self.persist_dir}")
                
                self.vectorstore = Chroma(
                    persist_directory=str(self.persist_dir),
                    embedding_function=self.embeddings
                )
                
               
                try:
                    collection = self.vectorstore._collection
                    count = collection.count()
                    print(f" Loaded vectorstore with {count} documents")
                    
                    if count > 0:
                        return self.vectorstore
                    else:
                        print("⚠️ Vectorstore is empty, rebuilding...")
                        
                except Exception as e:
                    print(f"⚠️ Error checking vectorstore: {e}")
                    print("Rebuilding vectorstore...")
                    
            except Exception as e:
                print(f"⚠️ Error loading persisted vectorstore: {e}")
                print("Building new vectorstore...")
        
        # Build new vectorstore
        print("🔨 Building new vectorstore...")
        
        # Get PDF files
        pdf_files = self._get_pdf_files()
        
        if not pdf_files:
            print("⚠️ No PDF files found. RAG will be disabled.")
            print(f"   Add PDF files to: {self.guides_dir.absolute()}")
            return None
        
        
        documents = self._load_documents(pdf_files)
        
        if not documents:
            print(" Failed to load documents. RAG will be disabled.")
            return None
        
        try:
            
            print(f" Creating vectorstore with {len(documents)} chunks...")
            
            
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=str(self.persist_dir)
            )
            
            
            collection = self.vectorstore._collection
            count = collection.count()
            print(f" Vectorstore created successfully with {count} documents")
            
            return self.vectorstore
            
        except Exception as e:
            print(f" Error creating vectorstore: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_retriever(self, k: int = 3, search_type: str = "similarity"):
        """
        Get retriever for the vectorstore
        
        Args:
            k: Number of documents to retrieve
            search_type: Type of search ("similarity" or "mmr")
            
        Returns:
            Retriever object or None
        """
        if not self.vectorstore:
            print("⚠️ No vectorstore available for retriever")
            return None
        
        try:
            retriever = self.vectorstore.as_retriever(
                search_type=search_type,
                search_kwargs={"k": k}
            )
            
            print(f"Retriever created (k={k}, type={search_type})")
            return retriever
            
        except Exception as e:
            print(f"Error creating retriever: {e}")
            return None
    
    def similarity_search(self, query: str, k: int = 3):
        """
        Perform similarity search
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of relevant documents
        """
        if not self.vectorstore:
            print("⚠️ No vectorstore available for search")
            return []
        
        try:
            results = self.vectorstore.similarity_search(query, k=k)
            print(f"✓ Found {len(results)} results for: {query[:50]}...")
            return results
        except Exception as e:
            print(f" Error in similarity search: {e}")
            return []
    
    def add_documents(self, documents):
        """Add new documents to existing vectorstore"""
        if not self.vectorstore:
            print("⚠️ No vectorstore available to add documents")
            return False
        
        try:
            self.vectorstore.add_documents(documents)
            print(f"✓ Added {len(documents)} documents to vectorstore")
            return True
        except Exception as e:
            print(f" Error adding documents: {e}")
            return False
    
    def delete_vectorstore(self):
        """Delete the persisted vectorstore"""
        import shutil
        
        if self.persist_dir.exists():
            try:
                shutil.rmtree(self.persist_dir)
                print(f" Deleted vectorstore at {self.persist_dir}")
                self.vectorstore = None
                return True
            except Exception as e:
                print(f" Error deleting vectorstore: {e}")
                return False
        else:
            print("⚠️ No vectorstore to delete")
            return False
