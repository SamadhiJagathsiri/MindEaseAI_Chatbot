from langchain_cohere import ChatCohere
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from chatbot.prompts.templates import RAG_PROMPT
from chatbot.memory.conversation_memory import MindEaseMemory
from utils.config import Config
from utils.vectorstore_manager import VectorStoreManager

class RAGChain:
    """Retrieval-Augmented Generation chain for wellness guide integration"""
    
    def __init__(self, memory: MindEaseMemory = None):
        self.llm = ChatCohere(
            cohere_api_key=Config.COHERE_API_KEY,
            model=Config.COHERE_MODEL,
            temperature=Config.TEMPERATURE,
            max_tokens=Config.MAX_TOKENS
        )
        
        self.memory = memory or MindEaseMemory()
        self.vectorstore_manager = VectorStoreManager()
        self.retriever = None
        self.chain = None
        
        self._initialize_chain()
    
    def _initialize_chain(self):
        """Initialize RAG chain with retriever"""
        try:
            self.retriever = self.vectorstore_manager.get_retriever(k=3)
            
            if not self.retriever:
                print("⚠️ No vector store available. RAG chain disabled.")
                return
            
            # Create document chain
            doc_chain = create_stuff_documents_chain(
                llm=self.llm,
                prompt=RAG_PROMPT
            )
            
            # Create retrieval chain
            self.chain = create_retrieval_chain(
                retriever=self.retriever,
                combine_docs_chain=doc_chain
            )
            
            print("✓ RAG chain initialized successfully")
            
        except Exception as e:
            print(f"Error initializing RAG chain: {e}")
            self.chain = None
    
    def is_available(self) -> bool:
        """Check if RAG functionality is available"""
        return self.chain is not None
    
    def generate_response(self, user_input: str) -> str:
        """Generate response using wellness guides context"""
        if not self.is_available():
            return None
        
        try:
            # Get chat history for context
            chat_history = self.memory.get_chat_history()
            
            result = self.chain.invoke({
                "input": user_input,
                "chat_history": chat_history
            })
            
            return result["answer"].strip()
            
        except Exception as e:
            print(f"Error in RAG generation: {e}")
            return None
    
    def search_guides(self, query: str, k: int = 3):
        """Search wellness guides directly"""
        if not self.retriever:
            return []
        
        return self.vectorstore_manager.similarity_search(query, k=k)