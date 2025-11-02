from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import AIMessage, HumanMessage
from utils.config import Config

class MindEaseMemory:
    """Enhanced conversation memory with emotional context tracking"""
    
    def __init__(self, max_length: int = None):
        self.max_length = max_length or Config.MAX_MEMORY_LENGTH
        
        self.memory = ConversationBufferWindowMemory(
            k=self.max_length,
            memory_key=Config.MEMORY_KEY,
            return_messages=True,
            output_key="output"
        )
        
        # Track emotional patterns
        self.emotional_history = []
    
    def add_interaction(self, user_input: str, ai_response: str, sentiment: dict = None):
        """Add user-AI interaction to memory"""
        self.memory.save_context(
            {"input": user_input},
            {"output": ai_response}
        )
        
        # Track emotional context
        if sentiment:
            self.emotional_history.append({
                "input": user_input,
                "sentiment": sentiment,
                "response": ai_response
            })
            
            
            if len(self.emotional_history) > self.max_length:
                self.emotional_history = self.emotional_history[-self.max_length:]
    
    def get_chat_history(self):
        """Retrieve formatted chat history"""
        return self.memory.load_memory_variables({}).get(Config.MEMORY_KEY, [])
    
    def get_emotional_summary(self):
        """Get summary of emotional patterns"""
        if not self.emotional_history:
            return "No emotional context yet"
        
        recent_sentiments = [entry["sentiment"] for entry in self.emotional_history[-5:]]
        
        
        avg_polarity = sum(s.get("polarity", 0) for s in recent_sentiments) / len(recent_sentiments)
        
        if avg_polarity > 0.3:
            return "predominantly positive"
        elif avg_polarity < -0.3:
            return "predominantly negative"
        else:
            return "mixed"
    
    def clear(self):
        """Clear conversation memory"""
        self.memory.clear()
        self.emotional_history = []
    
    def get_memory_object(self):
        """Return the LangChain memory object for chains"""
        return self.memory
