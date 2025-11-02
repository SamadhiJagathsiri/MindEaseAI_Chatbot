from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import Dict

class SentimentAnalyzer:
    """
    Analyzes emotional tone of user messages using dual approach:
    - TextBlob: Polarity and subjectivity
    - VADER: Social media optimized sentiment (better for informal text)
    """
    
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        
        
        self.emotion_keywords = {
            'joy': ['happy', 'joy', 'excited', 'great', 'wonderful', 'amazing', 'love', 'grateful'],
            'sadness': ['sad', 'depressed', 'down', 'unhappy', 'miserable', 'heartbroken', 'lonely'],
            'anxiety': ['anxious', 'worried', 'nervous', 'scared', 'afraid', 'panic', 'overwhelmed', 'stressed'],
            'anger': ['angry', 'mad', 'furious', 'frustrated', 'irritated', 'annoyed', 'rage'],
            'fear': ['fear', 'terrified', 'frightened', 'afraid', 'scared', 'worried'],
            'hope': ['hope', 'hopeful', 'optimistic', 'better', 'improve', 'forward', 'try'],
            'confusion': ['confused', 'lost', 'don\'t know', 'uncertain', 'unsure', 'mixed'],
        }
    
    def analyze(self, text: str) -> Dict:
        """
        Comprehensive sentiment analysis.
        Returns dict with polarity, subjectivity, label, emotions, and confidence.
        """
        if not text or len(text.strip()) < 2:
            return self._neutral_result()
        
        
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity  
        subjectivity = blob.sentiment.subjectivity  
        
        
        vader_scores = self.vader.polarity_scores(text)
        compound_score = vader_scores['compound']  
        
        
        combined_polarity = (polarity * 0.4) + (compound_score * 0.6)
        
    
        label = self._get_sentiment_label(combined_polarity)
        
        
        emotions = self._detect_emotions(text.lower())
        
        
        confidence = self._calculate_confidence(polarity, compound_score, subjectivity)
        
        return {
            'polarity': round(combined_polarity, 3),
            'subjectivity': round(subjectivity, 3),
            'label': label,
            'emotions': emotions,
            'confidence': confidence,
            'vader_breakdown': {
                'positive': vader_scores['pos'],
                'negative': vader_scores['neg'],
                'neutral': vader_scores['neu']
            }
        }
    
    def _get_sentiment_label(self, polarity: float) -> str:
        """Convert polarity score to human-readable label"""
        if polarity >= 0.3:
            return 'positive'
        elif polarity <= -0.3:
            return 'negative'
        elif polarity <= -0.1:
            return 'somewhat negative'
        elif polarity >= 0.1:
            return 'somewhat positive'
        else:
            return 'neutral'
    
    def _detect_emotions(self, text: str) -> list:
        """Detect specific emotions from keywords"""
        detected = []
        
        for emotion, keywords in self.emotion_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    detected.append(emotion)
                    break
        
        return list(set(detected))  
    
    def _calculate_confidence(self, polarity: float, compound: float, subjectivity: float) -> str:
        """
        Calculate confidence level in sentiment analysis.
        Higher when TextBlob and VADER agree, and text is more subjective.
        """
        
        agreement = abs(polarity - compound) < 0.3
        
        
        strong_sentiment = abs(compound) > 0.5
        
        
        is_subjective = subjectivity > 0.5
        
        if agreement and strong_sentiment and is_subjective:
            return 'high'
        elif agreement and (strong_sentiment or is_subjective):
            return 'medium'
        else:
            return 'low'
    
    def _neutral_result(self) -> Dict:
        """Return neutral sentiment for empty/invalid text"""
        return {
            'polarity': 0.0,
            'subjectivity': 0.0,
            'label': 'neutral',
            'emotions': [],
            'confidence': 'low',
            'vader_breakdown': {'positive': 0, 'negative': 0, 'neutral': 1}
        }
    
    def get_emoji_for_sentiment(self, sentiment: Dict) -> str:
        """Return emoji representation of sentiment"""
        polarity = sentiment.get('polarity', 0)
        
        if polarity >= 0.5:
            return '😊'
        elif polarity >= 0.2:
            return '🙂'
        elif polarity >= -0.2:
            return '😐'
        elif polarity >= -0.5:
            return '😔'
        else:
            return '😢'
    
    def is_high_distress(self, sentiment: Dict) -> bool:
        """Check if sentiment indicates high emotional distress"""
        polarity = sentiment.get('polarity', 0)
        emotions = sentiment.get('emotions', [])
        
        
        distress_emotions = {'sadness', 'anxiety', 'fear', 'anger'}
        has_distress = bool(set(emotions) & distress_emotions)
        
        return polarity < -0.5 and has_distress