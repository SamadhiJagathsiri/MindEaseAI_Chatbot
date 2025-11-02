from langchain_cohere import ChatCohere
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from chatbot.prompts.templates import RAG_PROMPT
from chatbot.memory.conversation_memory import MindEaseMemory
from utils.config import Config
from utils.vectorstore_manager import VectorStoreManager

class RAGChain:
    """Retrieval-Augmented Generation chain for wellness guide PDFs"""
    
    def __init__(self, memory: MindEaseMemory = None, vectorstore_manager: VectorStoreManager = None):
        """
        Initialize RAG chain
        
        Args:
            memory: Conversation memory manager
            vectorstore_manager: VectorStoreManager instance with loaded documents
        """
        self.llm = ChatCohere(
            cohere_api_key=Config.COHERE_API_KEY,
            model=Config.COHERE_MODEL,
            temperature=Config.TEMPERATURE,
            max_tokens=Config.MAX_TOKENS
        )
        
        self.memory = memory or MindEaseMemory()
        self.vectorstore_manager = vectorstore_manager
        self.chain = None
        self.retriever = None
        
        print("🔧 Initializing RAG chain...")
        self._initialize_chain()
    
    def _initialize_chain(self):
        """Initialize the RAG chain with retriever"""
        
        
        if not self.vectorstore_manager:
            print("⚠️ No vector store manager provided. RAG disabled.")
            return
        
        
        if not hasattr(self.vectorstore_manager, 'vectorstore') or self.vectorstore_manager.vectorstore is None:
            print("⚠️ No vectorstore found in manager. RAG disabled.")
            return
        
        try:
            
            self.retriever = self.vectorstore_manager.get_retriever(k=3)
            
            if not self.retriever:
                print("⚠️ Failed to create retriever. RAG disabled.")
                return
            
            
            try:
                test_results = self.retriever.get_relevant_documents("wellness")
                print(f"✓ Retriever test successful: Found {len(test_results)} documents")
            except Exception as e:
                print(f"⚠️ Retriever test failed: {e}")
                self.retriever = None
                return
            
            
            doc_chain = create_stuff_documents_chain(
                llm=self.llm, 
                prompt=RAG_PROMPT
            )
            
            
            self.chain = create_retrieval_chain(
                retriever=self.retriever,
                combine_docs_chain=doc_chain
            )
            
            print(" RAG chain initialized successfully")
            
        except Exception as e:
            print(f" Error initializing RAG chain: {e}")
            import traceback
            traceback.print_exc()
            self.chain = None
            self.retriever = None
    
    def is_available(self) -> bool:
        """Check if RAG chain is available and ready"""
        is_ready = (
            self.chain is not None and 
            self.retriever is not None and
            self.vectorstore_manager is not None
        )
        
        if not is_ready:
            print(f"RAG availability check: chain={self.chain is not None}, "
                  f"retriever={self.retriever is not None}, "
                  f"vectorstore_manager={self.vectorstore_manager is not None}")
        
        return is_ready
    
    def generate_response(self, user_input: str) -> str:
        """
        Generate response using RAG
        
        Args:
            user_input: User's question or message
            
        Returns:
            Generated response string or None if RAG fails
        """
        if not self.is_available():
            print("⚠️ RAG not available, returning None")
            return None
        
        
        chat_history = self.memory.get_chat_history()
        
        try:
            print(f"🔍 Retrieving documents for: {user_input[:50]}...")
            
            
            result = self.chain.invoke({
                "input": user_input,
                "chat_history": chat_history
            })
            
            
            answer = result.get("answer", "").strip()
            
            
            context_docs = result.get("context", [])
            print(f"RAG response generated using {len(context_docs)} documents")
            
            if not answer:
                print("⚠️ RAG returned empty answer")
                return None
            
            return answer
            
        except Exception as e:
            print(f" Error in RAG generation: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def search_guides(self, query: str, k: int = 3):
        """
        Search wellness guides directly
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of relevant document chunks
        """
        if not self.retriever:
            print("⚠️ No retriever available for search")
            return []
        
        try:
            results = self.vectorstore_manager.similarity_search(query, k=k)
            print(f"✓ Found {len(results)} documents for query: {query[:50]}...")
            return results
        except Exception as e:
            print(f" Error searching guides: {e}")
            return []
    
    def get_relevant_context(self, query: str, k: int = 3) -> str:
        """
        Get relevant context as formatted string
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            Formatted string of relevant context
        """
        docs = self.search_guides(query, k=k)
        
        if not docs:
            return "No relevant wellness guide information found."
        
        context_parts = []
        for i, doc in enumerate(docs, 1):
            content = doc.page_content.strip()
            source = doc.metadata.get('source', 'Unknown')
            context_parts.append(f"[Source {i}: {source}]\n{content}\n")
        
        return "\n".join(context_parts)
