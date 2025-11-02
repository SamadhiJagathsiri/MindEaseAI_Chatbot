import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Central configuration for MindEase AI"""
    
    
    COHERE_API_KEY = os.getenv("COHERE_API_KEY")
    
    
    COHERE_MODEL = "command-r-08-2024"  
    TEMPERATURE = 0.7
    MAX_TOKENS = 500
    

    EMBEDDING_MODEL = "embed-english-v3.0"
    VECTOR_STORE_PATH = "data/vectorstore"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    
    MEMORY_KEY = "chat_history"
    MAX_MEMORY_LENGTH = 10  
    
    
    CRISIS_KEYWORDS = [
        "suicide", "kill myself", "end it all", "don't want to live",
        "self-harm", "hurt myself", "overdose", "goodbye world"
    ]
    
    CRISIS_RESOURCES = {
    "Sri Lanka": {
        "Lifeline Foundation": "13 113",
        
        "Suicide Prevention Hotline": "011 269 9999",
        
        "National Institute of Mental Health": "011 269 8000",
        
        "International Help": "https://findahelpline.com"
    }
}
    
    
    WELLNESS_TOPICS = [
        "anxiety", "depression", "stress", "sleep",
        "meditation", "breathing", "mindfulness"
    ]

    @classmethod
    def validate(cls):
        """Validate required configurations"""
        if not cls.COHERE_API_KEY:
            raise ValueError("COHERE_API_KEY not found in environment")
        return True