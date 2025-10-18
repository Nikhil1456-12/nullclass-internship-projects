"""
Sentiment Analysis Integration for Chatbot

This module implements sentiment analysis to detect and respond appropriately to customer emotions
during interactions. It integrates with the existing knowledge_updater system to provide
emotionally aware responses.
"""

import os
import time
import pickle
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone
import re
import streamlit as st

# ML and NLP libraries
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np

# Local imports - integrate with existing knowledge_updater system
import sys
import os

# Add project root to path for direct execution
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from knowledge_updater.core.config import get_config
    from knowledge_updater.core.logging import get_logger, LogContext
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure you're running from the project root directory")
    print("Try: python -m streamlit run internship_projects/sentiment_analysis/sentiment_analyzer.py")
    sys.exit(1)

logger = get_logger(__name__)

# Download required NLTK data
def download_nltk_data():
    """Download required NLTK data packages"""
    try:
        # Check if punkt tokenizer data exists
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            logger.info("Downloading NLTK punkt tokenizer data...")
            nltk.download('punkt', quiet=True)

        # Check if stopwords corpus exists
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            logger.info("Downloading NLTK stopwords corpus...")
            nltk.download('stopwords', quiet=True)

        # Check if wordnet corpus exists
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            logger.info("Downloading NLTK wordnet corpus...")
            nltk.download('wordnet', quiet=True)

        logger.info("NLTK data download completed")
    except Exception as e:
        logger.error(f"Failed to download NLTK data: {e}")
        # Continue anyway - some functionality might still work

# Download NLTK data on import
download_nltk_data()


class SentimentAnalyzer:
    """Advanced sentiment analysis with emotion detection"""

    def __init__(self):
        """Initialize sentiment analyzer"""
        self.config = get_config()
        self.logger = get_logger(__name__)

        # Initialize NLTK components
        try:
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
        except:
            nltk.download('wordnet')
            nltk.download('stopwords')
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))

        # Sentiment lexicons
        self.positive_words = self._load_sentiment_lexicon('positive')
        self.negative_words = self._load_sentiment_lexicon('negative')

        # Emotion keywords
        self.emotion_keywords = {
            'joy': ['happy', 'excited', 'pleased', 'delighted', 'thrilled', 'wonderful', 'great', 'amazing', 'love', 'fantastic'],
            'sadness': ['sad', 'disappointed', 'upset', 'unhappy', 'depressed', 'miserable', 'heartbroken', 'sorrow', 'grief'],
            'anger': ['angry', 'furious', 'mad', 'annoyed', 'irritated', 'frustrated', 'outraged', 'hate', 'disgusted'],
            'fear': ['scared', 'afraid', 'worried', 'anxious', 'nervous', 'terrified', 'panic', 'frightened'],
            'surprise': ['surprised', 'shocked', 'amazed', 'astonished', 'unexpected', 'wow', 'oh my'],
            'disgust': ['disgusted', 'revolted', 'gross', 'awful', 'terrible', 'horrible', 'repulsive']
        }

        # Intensity modifiers
        self.intensifiers = ['very', 'extremely', 'really', 'so', 'absolutely', 'completely', 'totally']
        self.negations = ['not', 'never', 'no', 'nothing', 'nowhere', 'nobody', 'none']

        # Configuration
        self.sentiment_threshold = self.config.get('sentiment_analysis', {}).get('threshold', 0.1)
        self.emotion_threshold = self.config.get('sentiment_analysis', {}).get('emotion_threshold', 0.3)

        self.logger.info(
            "Sentiment analyzer initialized",
            positive_words=len(self.positive_words),
            negative_words=len(self.negative_words),
            emotion_categories=len(self.emotion_keywords)
        )

    def _load_sentiment_lexicon(self, sentiment_type: str) -> set:
        """Load sentiment lexicon from file or create default"""
        lexicon_file = f"data/sentiment_{sentiment_type}_lexicon.txt"

        if os.path.exists(lexicon_file):
            with open(lexicon_file, 'r') as f:
                return set(word.strip().lower() for word in f if word.strip())

        # Default lexicons
        if sentiment_type == 'positive':
            return {
                'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
                'love', 'like', 'awesome', 'brilliant', 'perfect', 'outstanding',
                'superb', 'marvelous', 'splendid', 'fabulous', 'terrific', 'nice',
                'pleasant', 'delightful', 'happy', 'joyful', 'excited', 'pleased'
            }
        else:  # negative
            return {
                'bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike', 'worst',
                'ugly', 'stupid', 'ridiculous', 'pathetic', 'disappointing', 'annoying',
                'frustrating', 'boring', 'sad', 'angry', 'mad', 'furious', 'disgusted'
            }

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment and emotions in text

        Args:
            text: Input text to analyze

        Returns:
            Sentiment analysis results with emotions and confidence scores
        """
        start_time = time.time()

        try:
            # Preprocess text
            cleaned_text = self._preprocess_text(text)

            # Calculate sentiment scores
            positive_score = self._calculate_sentiment_score(cleaned_text, self.positive_words)
            negative_score = self._calculate_sentiment_score(cleaned_text, self.negative_words)

            # Determine overall sentiment
            compound_score = positive_score - negative_score
            overall_sentiment = self._classify_sentiment(compound_score)

            # Detect emotions
            emotion_scores = self._detect_emotions(cleaned_text)
            dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0] if emotion_scores else 'neutral'

            # Calculate confidence
            confidence = min(abs(compound_score) * 2, 1.0)  # Scale to 0-1

            processing_time = time.time() - start_time

            result = {
                'text': text,
                'overall_sentiment': overall_sentiment,
                'compound_score': compound_score,
                'positive_score': positive_score,
                'negative_score': negative_score,
                'emotions': emotion_scores,
                'dominant_emotion': dominant_emotion,
                'confidence': confidence,
                'processing_time': processing_time,
                'word_count': len(cleaned_text.split()),
                'analyzed_at': datetime.now(timezone.utc).isoformat()
            }

            self.logger.debug(
                "Sentiment analysis completed",
                sentiment=overall_sentiment,
                confidence=confidence,
                dominant_emotion=dominant_emotion,
                processing_time=f"{processing_time:.3f}s"
            )

            return result

        except Exception as e:
            self.logger.error(f"Sentiment analysis failed: {e}")
            return self._get_fallback_sentiment_result(text, str(e))

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for analysis"""
        # Convert to lowercase
        text = text.lower()

        # Remove URLs, emails, and mentions
        text = re.sub(r'http\S+|www\S+|@\w+|#\w+', '', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text

    def _calculate_sentiment_score(self, text: str, sentiment_words: set) -> float:
        """Calculate sentiment score for given word set"""
        words = word_tokenize(text)
        score = 0.0

        for i, word in enumerate(words):
            # Check for negation
            negation_multiplier = 1.0
            if i > 0 and words[i-1] in self.negations:
                negation_multiplier = -1.0

            # Check for intensifiers
            intensifier_multiplier = 1.0
            if i > 0 and words[i-1] in self.intensifiers:
                intensifier_multiplier = 1.5

            # Lemmatize word for better matching
            lemma = self.lemmatizer.lemmatize(word)

            if lemma in sentiment_words:
                score += 1.0 * negation_multiplier * intensifier_multiplier

        # Normalize by text length
        word_count = len([w for w in words if w not in self.stop_words])
        if word_count > 0:
            score = score / word_count

        return score

    def _classify_sentiment(self, compound_score: float) -> str:
        """Classify sentiment based on compound score"""
        if compound_score > self.sentiment_threshold:
            return 'positive'
        elif compound_score < -self.sentiment_threshold:
            return 'negative'
        else:
            return 'neutral'

    def _detect_emotions(self, text: str) -> Dict[str, float]:
        """Detect specific emotions in text"""
        words = word_tokenize(text)
        emotion_scores = {emotion: 0.0 for emotion in self.emotion_keywords}

        for i, word in enumerate(words):
            # Check for negation
            negation_multiplier = 1.0
            if i > 0 and words[i-1] in self.negations:
                negation_multiplier = -0.5

            # Check for intensifiers
            intensifier_multiplier = 1.0
            if i > 0 and words[i-1] in self.intensifiers:
                intensifier_multiplier = 1.5

            # Check each emotion category
            for emotion, keywords in self.emotion_keywords.items():
                if word in keywords:
                    emotion_scores[emotion] += 1.0 * negation_multiplier * intensifier_multiplier

        # Normalize scores
        total_score = sum(emotion_scores.values())
        if total_score > 0:
            emotion_scores = {k: v/total_score for k, v in emotion_scores.items()}

        # Filter out low-confidence emotions
        emotion_scores = {k: v for k, v in emotion_scores.items() if v >= self.emotion_threshold}

        return emotion_scores

    def _get_fallback_sentiment_result(self, text: str, error: str) -> Dict[str, Any]:
        """Return fallback sentiment result on error"""
        return {
            'text': text,
            'overall_sentiment': 'neutral',
            'compound_score': 0.0,
            'positive_score': 0.0,
            'negative_score': 0.0,
            'emotions': {},
            'dominant_emotion': 'neutral',
            'confidence': 0.0,
            'processing_time': 0.0,
            'word_count': len(text.split()),
            'analyzed_at': datetime.now(timezone.utc).isoformat(),
            'error': error,
            'fallback': True
        }


class SentimentAwareResponseGenerator:
    """Generate responses that consider user sentiment"""

    def __init__(self, sentiment_analyzer: SentimentAnalyzer):
        """Initialize sentiment-aware response generator"""
        self.sentiment_analyzer = sentiment_analyzer
        self.config = get_config()

        # Response templates for different sentiments
        self.response_templates = {
            'positive': {
                'acknowledgment': "I'm glad to hear that! ",
                'continuation': "That's wonderful! How else can I help you today?",
                'enthusiastic': "Excellent! I'm thrilled we could help!"
            },
            'negative': {
                'empathy': "I'm sorry to hear that you're feeling this way. ",
                'help': "I want to help make this better. ",
                'apology': "I apologize for any inconvenience. "
            },
            'neutral': {
                'acknowledgment': "I understand. ",
                'clarification': "Could you provide more details? ",
                'help': "How can I assist you further?"
            }
        }

        # Emotion-specific responses
        self.emotion_responses = {
            'joy': "I'm happy to see you're feeling positive! ",
            'sadness': "I sense you might be feeling down. I'm here to help if you'd like to talk about it. ",
            'anger': "I can tell you're frustrated. Let me help resolve this issue for you. ",
            'fear': "I understand this might be concerning. Let me help clarify things for you. ",
            'surprise': "That's interesting! I didn't expect that. ",
            'disgust': "I understand this isn't pleasant. Let me help find a better solution. "
        }

    def generate_response(self, user_message: str, base_response: str) -> Dict[str, Any]:
        """
        Generate sentiment-aware response

        Args:
            user_message: Original user message
            base_response: Base response from chatbot

        Returns:
            Enhanced response considering sentiment
        """
        # Analyze sentiment
        sentiment_result = self.sentiment_analyzer.analyze_sentiment(user_message)

        # Generate sentiment-aware response
        enhanced_response = self._enhance_response_with_sentiment(
            base_response,
            sentiment_result
        )

        return {
            'response': enhanced_response,
            'sentiment_analysis': sentiment_result,
            'base_response': base_response,
            'sentiment_aware': True
        }

    def _enhance_response_with_sentiment(self, base_response: str, sentiment_result: Dict[str, Any]) -> str:
        """Enhance base response with sentiment awareness"""
        sentiment = sentiment_result['overall_sentiment']
        dominant_emotion = sentiment_result['dominant_emotion']
        confidence = sentiment_result['confidence']

        # Start with sentiment acknowledgment if confidence is high
        if confidence > 0.6:
            response_parts = []

            # Add emotion-specific acknowledgment
            if dominant_emotion in self.emotion_responses and dominant_emotion != 'neutral':
                response_parts.append(self.emotion_responses[dominant_emotion])

            # Add sentiment-specific response
            if sentiment in self.response_templates:
                if sentiment == 'positive' and confidence > 0.8:
                    response_parts.append(self.response_templates[sentiment]['enthusiastic'])
                elif sentiment == 'negative' and confidence > 0.7:
                    response_parts.append(self.response_templates[sentiment]['empathy'])
                    response_parts.append(self.response_templates[sentiment]['help'])
                else:
                    response_parts.append(self.response_templates[sentiment]['acknowledgment'])

            # Add base response
            response_parts.append(base_response)

            # Add continuation based on sentiment
            if sentiment == 'positive':
                response_parts.append(self.response_templates[sentiment]['continuation'])
            elif sentiment == 'negative':
                response_parts.append("Is there anything specific I can do to help?")
            else:
                response_parts.append(self.response_templates[sentiment]['help'])

            return ''.join(response_parts)
        else:
            # Return base response if confidence is low
            return base_response


class SentimentAnalysisDashboard:
    """Dashboard for visualizing sentiment analysis"""

    def __init__(self, sentiment_analyzer: SentimentAnalyzer):
        self.sentiment_analyzer = sentiment_analyzer

    def create_dashboard(self):
        """Create Streamlit dashboard for sentiment analysis"""
        st.title("😊 Sentiment Analysis Dashboard")
        st.markdown("Analyze sentiment and emotions in text using advanced NLP techniques")

        # Sidebar controls
        with st.sidebar:
            st.subheader("Analysis Settings")
            confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.1
            )

            st.subheader("Sample Texts")
            sample_texts = {
                "Positive": "I'm so happy with the service! Everything is working perfectly and I'm really impressed.",
                "Negative": "This is terrible and frustrating. I'm very disappointed with the poor quality.",
                "Neutral": "I need information about the product specifications and pricing details.",
                "Mixed": "The product is good but the customer service was disappointing."
            }

            selected_sample = st.selectbox(
                "Try Sample Text",
                options=list(sample_texts.keys())
            )

        # Main content
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Text Input")
            user_input = st.text_area(
                "Enter text to analyze:",
                value=sample_texts[selected_sample],
                height=150
            )

            if st.button("🔍 Analyze Sentiment", type="primary"):
                # Real-time analysis with progress
                progress_bar = st.progress(0)
                status_text = st.empty()

                for i in range(4):
                    progress_bar.progress((i + 1) * 25)
                    if i == 0:
                        status_text.text("📝 Preprocessing text...")
                    elif i == 1:
                        status_text.text("🧠 Analyzing emotions...")
                    elif i == 2:
                        status_text.text("📊 Calculating scores...")
                    else:
                        status_text.text("✨ Finalizing results...")
                    time.sleep(0.3)

                result = self.sentiment_analyzer.analyze_sentiment(user_input)
                progress_bar.empty()
                status_text.empty()

                # Display enhanced results
                self._display_sentiment_results(result)

            # Real-time analysis option
            if st.button("⚡ Real-time Analysis", help="See analysis as you type"):
                st.session_state.realtime_mode = not st.session_state.get('realtime_mode', False)

        # Real-time analysis display
        if st.session_state.get('realtime_mode', False) and user_input:
            with st.container():
                st.write("**Real-time Preview:**")
                realtime_result = self.sentiment_analyzer.analyze_sentiment(user_input)

                # Quick sentiment indicator
                sentiment = realtime_result['overall_sentiment']
                confidence = realtime_result['confidence']

                sentiment_colors = {
                    'positive': '🟢',
                    'negative': '🔴',
                    'neutral': '🟡'
                }

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Current Sentiment",
                        f"{sentiment_colors.get(sentiment, '⚪')} {sentiment.title()}",
                        delta=f"{confidence:.2f} confidence"
                    )
                with col2:
                    st.metric(
                        "Positive Score",
                        f"{realtime_result['positive_score']:.3f}"
                    )
                with col3:
                    st.metric(
                        "Negative Score",
                        f"{realtime_result['negative_score']:.3f}"
                    )

                if realtime_result.get('emotions'):
                    st.write("**Detected Emotions:**")
                    emotions_cols = st.columns(min(len(realtime_result['emotions']), 3))
                    for i, (emotion, score) in enumerate(realtime_result['emotions'].items()):
                        if i < 3:
                            emotions_cols[i].metric(
                                emotion.title(),
                                f"{score:.3f}"
                            )

        with col2:
            st.subheader("Quick Analysis")

            if st.button("Quick Test"):
                test_result = self.sentiment_analyzer.analyze_sentiment(user_input)
                self._display_quick_stats(test_result)

    def _display_sentiment_results(self, result: Dict[str, Any]):
        """Display detailed sentiment analysis results"""
        # Overall sentiment
        sentiment = result['overall_sentiment']
        confidence = result['confidence']

        # Color coding for sentiment
        color_map = {
            'positive': '🟢',
            'negative': '🔴',
            'neutral': '🟡'
        }

        st.markdown(f"### Overall Sentiment: {color_map.get(sentiment, '⚪')} {sentiment.title()}")
        st.progress(confidence)
        st.write(f"**Confidence:** {confidence:.2f}")

        # Sentiment scores
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Positive Score",
                f"{result['positive_score']:.3f}"
            )

        with col2:
            st.metric(
                "Negative Score",
                f"{result['negative_score']:.3f}"
            )

        with col3:
            st.metric(
                "Compound Score",
                f"{result['compound_score']:.3f}"
            )

        # Emotions detected
        if result.get('emotions'):
            st.subheader("🎭 Emotions Detected")
            emotions = result['emotions']

            # Create emotion bars
            for emotion, score in sorted(emotions.items(), key=lambda x: x[1], reverse=True):
                st.write(f"**{emotion.title()}:** {score:.2f}")
                st.progress(score)

                # Show keywords that triggered this emotion
                keywords = [word for word in self.sentiment_analyzer.emotion_keywords[emotion] if word in result['text'].lower()]
                if keywords:
                    st.caption(f"Triggered by: {', '.join(keywords[:3])}")

        # Processing details
        with st.expander("Processing Details"):
            st.write(f"**Processing Time:** {result['processing_time']:.3f}s")
            st.write(f"**Word Count:** {result['word_count']}")
            st.write(f"**Analyzed At:** {result['analyzed_at']}")

    def _display_quick_stats(self, result: Dict[str, Any]):
        """Display quick sentiment statistics"""
        st.write("**Quick Stats:**")
        st.info(f"Sentiment: {result['overall_sentiment'].title()} (Confidence: {result['confidence']:.2f})")

        if result.get('emotions'):
            dominant_emotion = result['dominant_emotion']
            st.info(f"Dominant Emotion: {dominant_emotion.title()}")

        st.caption(f"Processing time: {result['processing_time']:.3f}s")


def integrate_sentiment_with_chatbot(chatbot_response: str, user_message: str) -> str:
    """
    Integrate sentiment analysis with existing chatbot response

    Args:
        chatbot_response: Original response from chatbot
        user_message: User's message

    Returns:
        Sentiment-enhanced response
    """
    try:
        # Initialize sentiment analyzer
        analyzer = SentimentAnalyzer()
        response_generator = SentimentAwareResponseGenerator(analyzer)

        # Generate sentiment-aware response
        enhanced_response = response_generator.generate_response(user_message, chatbot_response)

        return enhanced_response['response']

    except Exception as e:
        logger.error(f"Sentiment integration failed: {e}")
        return chatbot_response  # Fallback to original response


def create_sentiment_analysis_ui():
    """Create Streamlit UI for sentiment analysis"""
    st.set_page_config(
        page_title="Sentiment Analysis Tool",
        page_icon="😊",
        layout="wide"
    )

    st.title("😊 Advanced Sentiment Analysis")
    st.markdown("Analyze text sentiment and emotions with detailed insights")

    # Initialize components
    if 'sentiment_analyzer' not in st.session_state:
        st.session_state.sentiment_analyzer = SentimentAnalyzer()

    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []

    # Create dashboard
    dashboard = SentimentAnalysisDashboard(st.session_state.sentiment_analyzer)
    dashboard.create_dashboard()


if __name__ == "__main__":
    create_sentiment_analysis_ui()