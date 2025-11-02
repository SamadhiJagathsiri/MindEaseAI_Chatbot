from chatbot.chains.conversation_chain import ConversationChain
from chatbot.chains.rag_chain import RAGChain
from chatbot.memory.conversation_memory import MindEaseMemory
from chatbot.crisis_detection import CrisisDetector
from chatbot.sentiment_analysis import SentimentAnalyzer
from utils.config import Config

class MindEaseAI:
    """
    Main orchestrator for MindEase AI
    Coordinates conversation, RAG, crisis detection, and sentiment analysis
    """
    
    def __init__(self):
        
        Config.validate()
        
        
        self.memory = MindEaseMemory()
        
        
        self.conversation_chain = ConversationChain(memory=self.memory)
        self.rag_chain = RAGChain(memory=self.memory)
        
        
        self.crisis_detector = CrisisDetector()
        self.sentiment_analyzer = SentimentAnalyzer()
        
        
        self.rag_enabled = self.rag_chain.is_available()
        
        print(" MindEase AI initialized successfully")
        if self.rag_enabled:
            print(" RAG mode: Enabled (wellness guides loaded)")
        else:
            print(" RAG mode: Disabled (add PDFs to data/guides/)")
    
    def process_message(self, user_input: str) -> dict:
        """
        Process user message through the complete pipeline
        Returns: {response, sentiment, crisis_detected, used_rag}
        """
        
        
        crisis_detected, crisis_response = self.crisis_detector.check_crisis(user_input)
        
        if crisis_detected:
            return {
                "response": crisis_response,
                "sentiment": self.sentiment_analyzer.analyze(user_input),
                "crisis_detected": True,
                "used_rag": False
            }
        
        
        sentiment = self.sentiment_analyzer.analyze(user_input)
        
        
        use_rag = self._should_use_rag(user_input, sentiment)
        
        
        if use_rag and self.rag_enabled:
            response = self.rag_chain.generate_response(user_input)
            
            if not response:
                response = self.conversation_chain.generate_response(user_input)
                use_rag = False
        else:
            response = self.conversation_chain.generate_response(user_input)
        
        
        self.memory.add_interaction(user_input, response, sentiment)
        
        return {
            "response": response,
            "sentiment": sentiment,
            "crisis_detected": False,
            "used_rag": use_rag
        }
    
    def _should_use_rag(self, user_input: str, sentiment: dict) -> bool:
        """Determine if RAG should be used based on query type"""
        if not self.rag_enabled:
            return False
        
        
        rag_triggers = [
            "how to", "what is", "help with", "strategies for",
            "tips", "advice", "techniques", "exercises",
            "cope", "manage", "deal with", "overcome"
        ]
        
        user_lower = user_input.lower()
        
        
        wellness_match = any(topic in user_lower for topic in Config.WELLNESS_TOPICS)
        trigger_match = any(trigger in user_lower for trigger in rag_triggers)
        
        return wellness_match or trigger_match
    
    def get_chat_history(self):
        """Retrieve conversation history"""
        return self.memory.get_chat_history()
    
    def get_emotional_summary(self):
        """Get emotional context summary"""
        return self.memory.get_emotional_summary()
    
    def clear_conversation(self):
        """Reset conversation state"""
        self.memory.clear()
        print("Conversation history cleared")
    
    def search_wellness_guides(self, query: str):
        """Search wellness guides directly"""
        if not self.rag_enabled:
            return []
        return self.rag_chain.search_guides(query)