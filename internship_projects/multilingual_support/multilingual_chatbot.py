"""
Multi-language Support for Chatbot

This module implements multi-language support for the chatbot system, including:
- Automatic language detection
- Translation between languages
- Culturally appropriate responses
- Support for 3+ additional languages beyond English
"""

import os
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone
import streamlit as st

# Translation libraries
from deep_translator import GoogleTranslator
from langdetect import detect

# Local imports - integrate with existing knowledge_updater system
import sys
import os

# Add project root to path for direct execution
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from knowledge_updater.core.config import get_config, ConfigManager
    from knowledge_updater.core.logging import get_logger, LogContext
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure you're running from the project root directory")
    print("Try: python -m streamlit run internship_projects/multilingual_support/multilingual_chatbot.py")
    sys.exit(1)

logger = get_logger(__name__)


class LanguageDetector:
    """Advanced language detection with confidence scoring"""

    def __init__(self):
        self.supported_languages = {
            'en': 'English',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'ja': 'Japanese',
            'ko': 'Korean',
            'zh': 'Chinese',
            'ar': 'Arabic',
            'hi': 'Hindi'
        }

        # Language-specific common words for better detection
        self.language_indicators = {
            'es': ['hola', 'gracias', 'por favor', 'muy', 'como', 'que'],
            'fr': ['bonjour', 'merci', 's\'il vous plaît', 'très', 'comment', 'que'],
            'de': ['hallo', 'danke', 'bitte', 'sehr', 'wie', 'was'],
            'it': ['ciao', 'grazie', 'per favore', 'molto', 'come', 'che'],
            'pt': ['olá', 'obrigado', 'por favor', 'muito', 'como', 'que']
        }

    def detect_language(self, text: str) -> Dict[str, Any]:
        """Detect language with confidence scoring"""
        try:
            # Use langdetect for primary detection
            detected_lang = detect(text)

            # Calculate confidence based on text characteristics
            confidence = self._calculate_detection_confidence(text, detected_lang)

            # Validate against supported languages
            if detected_lang not in self.supported_languages:
                detected_lang = 'en'  # Default to English
                confidence = 0.5

            return {
                'language_code': detected_lang,
                'language_name': self.supported_languages.get(detected_lang, 'Unknown'),
                'confidence': confidence,
                'supported': detected_lang in self.supported_languages,
                'text_length': len(text),
                'word_count': len(text.split())
            }

        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return {
                'language_code': 'en',
                'language_name': 'English',
                'confidence': 0.0,
                'supported': True,
                'error': str(e),
                'fallback': True
            }

    def _calculate_detection_confidence(self, text: str, detected_lang: str) -> float:
        """Calculate confidence in language detection"""
        base_confidence = 0.7  # Base confidence from langdetect

        # Boost confidence for longer texts
        if len(text) > 100:
            base_confidence += 0.1
        elif len(text) < 20:
            base_confidence -= 0.2

        # Check for language-specific indicators
        indicators = self.language_indicators.get(detected_lang, [])
        text_lower = text.lower()

        indicator_matches = sum(1 for indicator in indicators if indicator in text_lower)
        if indicator_matches > 0:
            base_confidence += min(indicator_matches * 0.1, 0.2)

        return min(base_confidence, 1.0)


class TranslationManager:
    """Manage translations between multiple languages"""

    def __init__(self):
        self.google_translator = GoogleTranslator()

        # Supported language pairs
        self.supported_languages = {
            'en': 'English',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'ja': 'Japanese',
            'ko': 'Korean',
            'zh': 'Chinese',
            'ar': 'Arabic',
            'hi': 'Hindi'
        }

        # Translation cache for performance
        self.translation_cache = {}

    def translate_text(self, text: str, target_lang: str, source_lang: str = 'auto') -> Dict[str, Any]:
        """Translate text between languages"""
        cache_key = f"{source_lang}:{target_lang}:{hash(text)}"

        # Check cache first
        if cache_key in self.translation_cache:
            return self.translation_cache[cache_key]

        try:
            # Use deep_translator for better reliability
            translated_text = self.google_translator.translate(
                text=text,
                target=target_lang,
                source=source_lang
            )

            result = {
                'original_text': text,
                'translated_text': translated_text,
                'source_language': source_lang,
                'target_language': target_lang,
                'success': True,
                'cached': False
            }

            # Cache the result
            self.translation_cache[cache_key] = result

            return result

        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return {
                'original_text': text,
                'translated_text': text,  # Return original on failure
                'source_language': source_lang,
                'target_language': target_lang,
                'success': False,
                'error': str(e),
                'cached': False
            }

    def get_supported_languages(self) -> Dict[str, str]:
        """Get list of supported languages"""
        return self.supported_languages.copy()


class CulturalResponseAdapter:
    """Adapt responses for cultural appropriateness"""

    def __init__(self):
        self.cultural_contexts = {
            'es': {
                'greeting': '¡Hola! ¿En qué puedo ayudarte?',
                'farewell': '¡Que tengas un excelente día!',
                'polite_form': 'formal',
                'time_format': '24h'
            },
            'fr': {
                'greeting': 'Bonjour! Comment puis-je vous aider?',
                'farewell': 'Passez une excellente journée!',
                'polite_form': 'formal',
                'time_format': '24h'
            },
            'de': {
                'greeting': 'Guten Tag! Wie kann ich Ihnen helfen?',
                'farewell': 'Haben Sie einen wunderbaren Tag!',
                'polite_form': 'formal',
                'time_format': '24h'
            },
            'ja': {
                'greeting': 'こんにちは！どのようにお手伝いできますか？',
                'farewell': '素晴らしい一日をお過ごしください！',
                'polite_form': 'very_formal',
                'time_format': '24h'
            },
            'ko': {
                'greeting': '안녕하세요! 어떻게 도와드릴까요?',
                'farewell': '좋은 하루 보내세요!',
                'polite_form': 'formal',
                'time_format': '24h'
            }
        }

    def adapt_response(self, response: str, target_language: str) -> str:
        """Adapt response for cultural context"""
        if target_language not in self.cultural_contexts:
            return response

        context = self.cultural_contexts[target_language]

        # Add culturally appropriate greeting if response is starting a conversation
        if any(word in response.lower() for word in ['hello', 'hi', 'welcome']):
            return f"{context['greeting']} {response}"

        return response


class MultilingualChatbot:
    """Multi-language chatbot with translation capabilities"""

    def __init__(self):
        """Initialize multilingual chatbot"""
        # Explicitly specify config path to ensure correct file is loaded
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config.yaml")
        config_manager = ConfigManager(config_path)
        self.config = config_manager.get_config()
        self.logger = get_logger(__name__)

        # Initialize components
        self.language_detector = LanguageDetector()
        self.translation_manager = TranslationManager()
        self.cultural_adapter = CulturalResponseAdapter()

        # Chatbot settings
        self.primary_language = self.config.get('multilingual', {}).get('primary_language', 'en')
        self.supported_languages = self.config.get('multilingual', {}).get('supported_languages', ['en', 'es', 'fr', 'de'])
        self.auto_translate = self.config.get('multilingual', {}).get('auto_translate', True)
        self.preserve_formatting = self.config.get('multilingual', {}).get('preserve_formatting', True)

        # Conversation context
        self.current_language = self.primary_language
        self.conversation_history = []

        self.logger.info(
            "Multilingual chatbot initialized",
            primary_language=self.primary_language,
            supported_languages=self.supported_languages,
            auto_translate=self.auto_translate
        )

    def process_message(self, message: str, target_language: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a multilingual message

        Args:
            message: User message in any supported language
            target_language: Target language for response (optional)

        Returns:
            Processed response with translation metadata
        """
        start_time = time.time()

        try:
            # Detect message language
            language_info = self.language_detector.detect_language(message)

            # Update current language context
            if language_info['supported'] and language_info['confidence'] > 0.7:
                self.current_language = language_info['language_code']

            # Determine target language for response
            if not target_language:
                target_language = self.current_language if language_info['confidence'] > 0.8 else self.primary_language

            # Translate message to English for processing (if not already English)
            english_message = message
            if language_info['language_code'] != 'en':
                translation_result = self.translation_manager.translate_text(
                    message,
                    target_lang='en',
                    source_lang=language_info['language_code']
                )
                if translation_result['success']:
                    english_message = translation_result['translated_text']

            # Process the English version (integrate with existing chatbot)
            # For now, generate a simple response - in real implementation,
            # this would call the existing chatbot systems
            base_response = self._generate_base_response(english_message)

            # Translate response back to target language
            final_response = base_response
            translation_metadata = {}

            if target_language != 'en':
                response_translation = self.translation_manager.translate_text(
                    base_response,
                    target_lang=target_language,
                    source_lang='en'
                )

                if response_translation['success']:
                    final_response = response_translation['translated_text']
                    translation_metadata = {
                        'response_translated': True,
                        'translation_confidence': 0.9,  # Could be improved
                        'original_response': base_response
                    }

                # Adapt for cultural context
                final_response = self.cultural_adapter.adapt_response(final_response, target_language)

            processing_time = time.time() - start_time

            result = {
                'user_message': message,
                'detected_language': language_info,
                'response': final_response,
                'target_language': target_language,
                'base_response': base_response,
                'processing_time': processing_time,
                'translation_metadata': translation_metadata,
                'conversation_context': {
                    'current_language': self.current_language,
                    'message_count': len(self.conversation_history)
                },
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

            # Add to conversation history
            self.conversation_history.append(result)

            # Limit history size
            if len(self.conversation_history) > 50:
                self.conversation_history = self.conversation_history[-50:]

            self.logger.info(
                "Multilingual message processed",
                detected_language=language_info['language_code'],
                target_language=target_language,
                processing_time=f"{processing_time:.3f}s"
            )

            return result

        except Exception as e:
            self.logger.error(f"Multilingual processing failed: {e}")
            return self._generate_error_response(message, str(e))

    def _generate_base_response(self, message: str) -> str:
        """Generate base response in English"""
        # This would integrate with existing chatbot systems
        # For demonstration, provide simple responses based on keywords

        message_lower = message.lower()

        if any(word in message_lower for word in ['hello', 'hi', 'greetings']):
            return "Hello! I'm a multilingual chatbot that can help you in many languages. How can I assist you today?"
        elif any(word in message_lower for word in ['help', 'support', 'assist']):
            return "I can help you with various topics and questions. I support multiple languages including English, Spanish, French, German, and more. What would you like to know?"
        elif any(word in message_lower for word in ['language', 'languages', 'idioma']):
            return "I support many languages including English, Spanish, French, German, Italian, Portuguese, Russian, Japanese, Korean, Chinese, Arabic, and Hindi. I can detect your language automatically and respond in the same language!"
        else:
            return f"I understand you're asking about: '{message}'. I'm here to help! I can respond in multiple languages and understand various cultural contexts."

    def _generate_error_response(self, message: str, error: str) -> Dict[str, Any]:
        """Generate error response"""
        return {
            'user_message': message,
            'detected_language': {'language_code': 'en', 'confidence': 0.0},
            'response': "I'm sorry, I encountered an error processing your message. Please try again.",
            'target_language': self.primary_language,
            'base_response': "Error processing message",
            'processing_time': 0.0,
            'error': error,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

    def switch_language(self, target_language: str) -> bool:
        """Switch chatbot's primary language"""
        if target_language in self.supported_languages:
            self.current_language = target_language
            self.logger.info(f"Switched to language: {target_language}")
            return True
        return False

    def get_supported_languages(self) -> Dict[str, str]:
        """Get list of supported languages"""
        return self.translation_manager.get_supported_languages()

    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get conversation statistics"""
        languages_used = []
        for msg in self.conversation_history:
            lang = msg.get('detected_language', {}).get('language_code', 'unknown')
            if lang not in languages_used:
                languages_used.append(lang)

        return {
            'total_messages': len(self.conversation_history),
            'languages_used': languages_used,
            'current_language': self.current_language,
            'primary_language': self.primary_language,
            'auto_translate': self.auto_translate
        }


def create_multilingual_chat_ui():
    """Create Streamlit UI for multilingual chatbot"""
    st.set_page_config(
        page_title="Multilingual Chatbot",
        page_icon="🌐",
        layout="wide"
    )

    st.title("🌐 Multilingual Chatbot")
    st.markdown("Chat with me in any supported language! I can detect your language automatically and respond appropriately.")

    # Initialize chatbot
    if 'multilingual_bot' not in st.session_state:
        st.session_state.multilingual_bot = MultilingualChatbot()
        st.session_state.chat_history = []

    # Language selection sidebar
    with st.sidebar:
        st.subheader("🌍 Language Settings")

        # Get supported languages
        supported_langs = st.session_state.multilingual_bot.get_supported_languages()

        # Language selector
        current_lang = st.session_state.multilingual_bot.current_language
        selected_lang = st.selectbox(
            "Response Language",
            options=list(supported_langs.keys()),
            format_func=lambda x: supported_langs[x],
            index=list(supported_langs.keys()).index(current_lang) if current_lang in supported_langs else 0
        )

        if selected_lang != current_lang:
            if st.session_state.multilingual_bot.switch_language(selected_lang):
                st.success(f"Switched to {supported_langs[selected_lang]}")
                st.rerun()

        # Display supported languages
        st.subheader("Supported Languages")
        for code, name in supported_langs.items():
            if code == current_lang:
                st.write(f"✅ **{name}** (Current)")
            else:
                st.write(f"🌐 {name}")

        # Conversation stats
        st.subheader("📊 Conversation Stats")
        stats = st.session_state.multilingual_bot.get_conversation_stats()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Messages", stats['total_messages'])
            st.metric("Current Language", supported_langs.get(stats['current_language'], 'Unknown'))

        with col2:
            st.metric("Languages Used", len(stats['languages_used']))
            st.metric("Auto Translate", "Enabled" if stats['auto_translate'] else "Disabled")

    # Main chat interface
    st.subheader("💬 Chat")

    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            # User message
            with st.chat_message("user"):
                detected_lang = message.get('detected_language', {})
                if detected_lang.get('language_name'):
                    st.caption(f"Detected: {detected_lang['language_name']} ({detected_lang.get('confidence', 0):.2f} confidence)")
                st.markdown(message['user_message'])

            # Bot response
            with st.chat_message("assistant"):
                st.markdown(message['response'])

                # Show translation info if applicable
                translation_meta = message.get('translation_metadata', {})
                if translation_meta.get('response_translated'):
                    with st.expander("Translation Details"):
                        st.write(f"**Original (English):** {translation_meta.get('original_response', '')}")
                        st.write(f"**Target Language:** {message.get('target_language', 'Unknown')}")
                        st.caption("Response was translated for better cultural appropriateness")

    # Chat input
    user_input = st.chat_input("Type your message in any language...")

    if user_input:
        # Process the message
        with st.chat_message("user"):
            st.markdown(user_input)

        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Processing message..."):
                response = st.session_state.multilingual_bot.process_message(
                    user_input,
                    selected_lang
                )

            st.markdown(response['response'])

            # Show language detection info
            detected_lang = response.get('detected_language', {})
            if detected_lang.get('confidence', 0) > 0.5:
                st.caption(f"Detected language: {detected_lang.get('language_name', 'Unknown')}")

        # Add to chat history
        st.session_state.chat_history.append({
            'user_message': user_input,
            'response': response['response'],
            'detected_language': response['detected_language'],
            'target_language': response['target_language'],
            'translation_metadata': response.get('translation_metadata', {})
        })

        # Limit chat history
        if len(st.session_state.chat_history) > 20:
            st.session_state.chat_history = st.session_state.chat_history[-20:]

    # Clear chat button
    if st.button("Clear Chat History"):
        st.session_state.chat_history.clear()
        st.rerun()

    # Demo section
    st.subheader("🌟 Try These Examples")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🇪🇸 Spanish"):
            demo_response = st.session_state.multilingual_bot.process_message(
                "¿Cómo estás? ¿Puedes ayudarme con información sobre inteligencia artificial?",
                selected_lang
            )
            st.chat_message("user").write("¿Cómo estás? ¿Puedes ayudarme con información sobre inteligencia artificial?")
            st.chat_message("assistant").write(demo_response['response'])

    with col2:
        if st.button("🇫🇷 French"):
            demo_response = st.session_state.multilingual_bot.process_message(
                "Bonjour! Pouvez-vous m'expliquer le machine learning?",
                selected_lang
            )
            st.chat_message("user").write("Bonjour! Pouvez-vous m'expliquer le machine learning?")
            st.chat_message("assistant").write(demo_response['response'])

    with col3:
        if st.button("🇩🇪 German"):
            demo_response = st.session_state.multilingual_bot.process_message(
                "Guten Tag! Können Sie mir bei Computer Vision helfen?",
                selected_lang
            )
            st.chat_message("user").write("Guten Tag! Können Sie mir bei Computer Vision helfen?")
            st.chat_message("assistant").write(demo_response['response'])


if __name__ == "__main__":
    create_multilingual_chat_ui()