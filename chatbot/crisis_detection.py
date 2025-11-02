import re
from typing import Tuple
from utils.config import Config
from chatbot.prompts.templates import CRISIS_RESPONSE_TEMPLATE

class CrisisDetector:
    """
    Detects potential crisis situations in user messages.
    Multi-layered approach: keyword detection + pattern matching + severity scoring.
    """
    
    def __init__(self):
        self.crisis_keywords = Config.CRISIS_KEYWORDS
        self.crisis_resources = Config.CRISIS_RESOURCES
        
        
        self.high_severity_patterns = [
            r'\b(want to|going to|plan to|will)\s+(die|kill myself|end (it|my life))\b',
            r'\b(suicide|suicidal)\s+(plan|thoughts?|ideation)\b',
            r'\bno (reason|point) (to|in) (live|living)\b',
            r'\b(goodbye|farewell).{0,20}(world|everyone|forever)\b',
            r'\btake my (own )?life\b',
        ]
        
        
        self.medium_severity_patterns = [
            r'\b(can\'?t|cannot) (go on|take (it|this) anymore)\b',
            r'\bbetter off (dead|without me)\b',
            r'\beveryone would be better\b',
            r'\b(hurt|harm) myself\b',
            r'\b(cutting|burning) myself\b',
        ]
        
        
        self.intensity_words = [
            'really', 'very', 'extremely', 'seriously', 'desperately',
            'truly', 'absolutely', 'completely', 'totally'
        ]
    
    def check_crisis(self, message: str) -> Tuple[bool, str]:
        """
        Analyze message for crisis indicators.
        Returns: (is_crisis, response_message)
        """
        if not message or len(message.strip()) < 3:
            return False, ""
        
        message_lower = message.lower()
        
        
        severity_score = 0
        
        
        for pattern in self.high_severity_patterns:
            if re.search(pattern, message_lower):
                severity_score += 10
                break
        
        
        for pattern in self.medium_severity_patterns:
            if re.search(pattern, message_lower):
                severity_score += 5
                break
        
    
        keyword_matches = sum(1 for keyword in self.crisis_keywords 
                             if keyword.lower() in message_lower)
        severity_score += keyword_matches * 2
        
    
        intensity_count = sum(1 for word in self.intensity_words 
                             if word in message_lower)
        if intensity_count > 0 and keyword_matches > 0:
            severity_score += intensity_count
        
        
        first_person = any(pronoun in message_lower for pronoun in ['i ', 'i\'m', 'im ', 'my '])
        negative_future = any(phrase in message_lower for phrase in [
            'can\'t go', 'won\'t make', 'give up', 'no hope', 'no way out'
        ])
        
        if first_person and negative_future:
            severity_score += 3
        
        
        is_crisis = severity_score >= 8
        
        if is_crisis:
            response = self._generate_crisis_response(severity_score)
            return True, response
        
        return False, ""
    
    def _generate_crisis_response(self, severity_score: int) -> str:
        """Generate appropriate crisis response based on severity"""
        
        
        resources_text = "\n".join([
            f"**{region}:** {contact}" 
            for region, contact in self.crisis_resources.items()
        ])
        
        
        base_response = CRISIS_RESPONSE_TEMPLATE.format(resources=resources_text)
        
        
        if severity_score >= 15:
        
            urgent_addition = "\n\n⚠️ **Please reach out right now.** These services have trained counselors who want to help you through this moment. You deserve support."
            return base_response + urgent_addition
        
        return base_response
    
    def is_follow_up_to_crisis(self, message: str) -> bool:
        """
        Check if message is a follow-up to a crisis conversation.
        Used to maintain supportive mode.
        """
        follow_up_patterns = [
            r'\b(called|contacted|talked to|reached out)\b',
            r'\b(feeling (a little )?better|calmer now)\b',
            r'\b(thank you|thanks) (for|about)',
            r'\bstill (here|struggling|hard)\b',
        ]
        
        message_lower = message.lower()
        return any(re.search(pattern, message_lower) for pattern in follow_up_patterns)
    
    def get_safety_resources(self) -> dict:
        """Return crisis resources for display"""
        return self.crisis_resources