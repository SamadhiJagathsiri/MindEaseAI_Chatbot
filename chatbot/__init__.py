"""
MindEase AI Chatbot Package

Main components:
- MindEaseAI: Core orchestrator
- ConversationChain: Standard dialogue
- RAGChain: Knowledge-enhanced responses
- ReflectionChain: Self-awareness prompts
- CrisisDetector: Safety system
- SentimentAnalyzer: Emotional analysis
"""

from chatbot.mindease_ai import MindEaseAI
from chatbot.crisis_detection import CrisisDetector
from chatbot.sentiment_analysis import SentimentAnalyzer

__all__ = [
    'MindEaseAI',
    'CrisisDetector',
    'SentimentAnalyzer',
]

__version__ = '2.0.0'