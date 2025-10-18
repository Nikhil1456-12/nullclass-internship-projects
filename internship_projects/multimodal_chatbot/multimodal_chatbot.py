"""
Multi-modal Chatbot with Google Palm/Gemini AI Integration

This module implements a chatbot that can handle both text and image inputs,
generating responses using Google's Palm and Gemini AI models while integrating
with the existing knowledge_updater system for enhanced responses.
"""

import os
import time
import base64
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timezone
import streamlit as st

# Google AI imports
import google.generativeai as genai

# Image processing
from PIL import Image
import io

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
    from knowledge_updater.query.enhancer import QueryEnhancer
    from knowledge_updater.query.response_generator import ResponseGenerator
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure you're running from the project root directory")
    print("Try: python -m streamlit run internship_projects/multimodal_chatbot/multimodal_chatbot.py")
    sys.exit(1)

logger = get_logger(__name__)


class MultimodalChatbot:
    """
    Multi-modal chatbot that handles text and image inputs using Google AI
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the multi-modal chatbot

        Args:
            api_key: Google AI API key (optional, can be set via config)
        """
        self.config = get_config()
        self.logger = get_logger(__name__)

        # Initialize Google AI
        self.api_key = api_key or self.config.get('google_ai', {}).get('api_key')
        if not self.api_key:
            raise ValueError("Google AI API key is required")

        genai.configure(api_key=self.api_key)

        # Initialize available models
        self.models = {
            'gemini': 'gemini-pro',
            'gemini_vision': 'gemini-pro-vision',
            'palm': 'models/text-bison-001'
        }

        # Initialize knowledge base integration
        self.query_enhancer = QueryEnhancer()
        self.response_generator = ResponseGenerator()

        # Chat history
        self.chat_history = []

        # Configuration
        self.max_history_length = self.config.get('multimodal_chatbot', {}).get('max_history_length', 50)
        self.enable_knowledge_base = self.config.get('multimodal_chatbot', {}).get('enable_knowledge_base', True)
        self.vision_confidence_threshold = self.config.get('multimodal_chatbot', {}).get('vision_confidence_threshold', 0.7)

        self.logger.info(
            "Multimodal chatbot initialized",
            models_available=list(self.models.keys()),
            knowledge_base_enabled=self.enable_knowledge_base
        )

    def chat(self, message: str, image: Optional[Image.Image] = None) -> Dict[str, Any]:
        """
        Process a chat message with optional image input

        Args:
            message: Text message from user
            image: Optional PIL Image for vision analysis

        Returns:
            Chatbot response with metadata
        """
        start_time = time.time()

        try:
            self.logger.info(
                "Processing chat message",
                message_length=len(message),
                has_image=image is not None
            )

            # Add user message to history
            self._add_to_history('user', message, image)

            # Process the message
            if image and self._is_vision_query(message):
                response_data = self._process_vision_query(message, image)
            else:
                response_data = self._process_text_query(message)

            # Add assistant response to history
            self._add_to_history('assistant', response_data['response'])

            processing_time = time.time() - start_time

            self.logger.info(
                "Chat message processed",
                processing_time=f"{processing_time:.3f}s",
                response_length=len(response_data.get('response', ''))
            )

            return response_data

        except Exception as e:
            error_msg = f"I apologize, but I encountered an error: {str(e)}"
            self.logger.error("Chat processing failed", error=str(e))

            return {
                'response': error_msg,
                'confidence': 0.0,
                'error': str(e),
                'processing_time': time.time() - start_time
            }

    def _is_vision_query(self, message: str) -> bool:
        """Determine if a query requires vision analysis"""
        vision_keywords = [
            'describe', 'what do you see', 'analyze this image', 'look at',
            'what is this', 'identify', 'explain this picture', 'tell me about'
        ]

        message_lower = message.lower()
        return any(keyword in message_lower for keyword in vision_keywords)

    def _process_vision_query(self, message: str, image: Image.Image) -> Dict[str, Any]:
        """Process a query that involves image analysis"""
        try:
            # Convert image to base64 for Gemini Vision
            image_base64 = self._image_to_base64(image)

            # Initialize Gemini Vision model
            model = genai.GenerativeModel(self.models['gemini_vision'])

            # Create prompt for vision analysis
            vision_prompt = self._create_vision_prompt(message)

            # Generate response
            response = model.generate_content([vision_prompt, image])

            # Extract response text
            response_text = response.text if response else "I can see the image but couldn't generate a detailed description."

            # Enhance response with knowledge base if enabled
            if self.enable_knowledge_base:
                enhanced_response = self._enhance_with_knowledge_base(message, response_text)
                return enhanced_response
            else:
                return {
                    'response': response_text,
                    'confidence': 0.9,
                    'model_used': 'gemini_vision',
                    'vision_analysis': True,
                    'image_processed': True
                }

        except Exception as e:
            self.logger.error("Vision query processing failed", error=str(e))
            return self._generate_fallback_vision_response(message, image, str(e))

    def _process_text_query(self, message: str) -> Dict[str, Any]:
        """Process a text-only query"""
        try:
            # Use Gemini Pro for text generation
            model = genai.GenerativeModel(self.models['gemini'])

            # Enhance query with knowledge base if enabled
            if self.enable_knowledge_base:
                enhanced_query = self.query_enhancer.enhance_query(message)

                # Create context-aware prompt
                context_prompt = self._create_contextual_prompt(message, enhanced_query)

                response = model.generate_content(context_prompt)

                response_text = response.text if response else "I'm having trouble generating a response."

                return {
                    'response': response_text,
                    'confidence': enhanced_query.get('confidence', 0.5),
                    'model_used': 'gemini_pro',
                    'knowledge_base_used': True,
                    'context_results': enhanced_query.get('results_count', 0)
                }
            else:
                # Simple text response without knowledge base
                response = model.generate_content(message)
                response_text = response.text if response else "I'm having trouble generating a response."

                return {
                    'response': response_text,
                    'confidence': 0.8,
                    'model_used': 'gemini_pro',
                    'knowledge_base_used': False
                }

        except Exception as e:
            self.logger.error("Text query processing failed", error=str(e))
            return self._generate_fallback_text_response(message, str(e))

    def _create_vision_prompt(self, message: str) -> str:
        """Create a prompt for vision analysis"""
        return f"""
        Please analyze this image and respond to the user's query: "{message}"

        Provide a detailed, helpful response that:
        1. Describes what you see in the image
        2. Answers the user's specific question
        3. Provides relevant context or additional information if applicable

        Be conversational and helpful in your response.
        """

    def _create_contextual_prompt(self, message: str, enhanced_query: Dict[str, Any]) -> str:
        """Create a prompt that includes knowledge base context"""
        context = enhanced_query.get('context', [])

        if not context:
            return message

        # Include top context results in the prompt
        context_text = "Based on relevant information from my knowledge base:\n\n"
        for i, item in enumerate(context[:3], 1):
            title = item.get('metadata', {}).get('title', 'Relevant Information')
            content = item.get('metadata', {}).get('content', '')[:300]
            source = item.get('metadata', {}).get('source', 'Unknown')

            context_text += f"{i}. {title} (Source: {source})\n{content}...\n\n"

        return f"{context_text}\nUser question: {message}\n\nPlease provide a comprehensive answer based on this context."

    def _enhance_with_knowledge_base(self, original_query: str, vision_response: str) -> Dict[str, Any]:
        """Enhance vision response with knowledge base information"""
        try:
            # Create a combined query for knowledge base enhancement
            combined_query = f"{original_query} {vision_response[:200]}"

            enhanced_query = self.query_enhancer.enhance_query(combined_query)

            # Generate enhanced response
            enhanced_response = self.response_generator.generate_response(
                original_query,
                conversation_context=self._get_recent_context()
            )

            return {
                'response': enhanced_response['response'],
                'confidence': enhanced_response['confidence'],
                'model_used': 'gemini_vision + knowledge_base',
                'vision_analysis': True,
                'knowledge_base_used': True,
                'context_results': enhanced_response['context_used']
            }

        except Exception as e:
            self.logger.warning("Knowledge base enhancement failed", error=str(e))
            return {
                'response': vision_response,
                'confidence': 0.8,
                'model_used': 'gemini_vision',
                'vision_analysis': True,
                'knowledge_base_used': False,
                'enhancement_error': str(e)
            }

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode()

    def _add_to_history(self, role: str, content: str, image: Optional[Image.Image] = None):
        """Add message to chat history"""
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'has_image': image is not None
        }

        self.chat_history.append(message)

        # Maintain history length limit
        if len(self.chat_history) > self.max_history_length:
            self.chat_history = self.chat_history[-self.max_history_length:]

    def _get_recent_context(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent conversation context"""
        recent_messages = []
        for msg in reversed(self.chat_history[-limit*2:]):  # Get more to filter for user messages
            if msg['role'] == 'user':
                recent_messages.insert(0, {
                    'content': msg['content'],
                    'timestamp': msg['timestamp']
                })
                if len(recent_messages) >= limit:
                    break

        return recent_messages

    def _generate_fallback_vision_response(self, message: str, image: Image.Image, error: str) -> Dict[str, Any]:
        """Generate fallback response for vision query failures"""
        fallback_response = f"I can see that you've shared an image, but I'm having trouble analyzing it right now. The image appears to be {image.size} pixels in size and {image.format} format."

        return {
            'response': fallback_response,
            'confidence': 0.3,
            'model_used': 'fallback',
            'vision_analysis': False,
            'error': error,
            'image_info': {
                'size': image.size,
                'format': image.format,
                'mode': image.mode
            }
        }

    def _generate_fallback_text_response(self, message: str, error: str) -> Dict[str, Any]:
        """Generate fallback response for text query failures"""
        return {
            'response': "I'm sorry, I'm having trouble processing your message right now. Please try again in a moment.",
            'confidence': 0.0,
            'model_used': 'fallback',
            'error': error
        }

    def get_chat_history(self) -> List[Dict[str, Any]]:
        """Get current chat history"""
        return self.chat_history.copy()

    def clear_history(self):
        """Clear chat history"""
        self.chat_history.clear()
        self.logger.info("Chat history cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get chatbot statistics"""
        return {
            'messages_in_history': len(self.chat_history),
            'knowledge_base_enabled': self.enable_knowledge_base,
            'models_available': list(self.models.keys()),
            'max_history_length': self.max_history_length,
            'vision_confidence_threshold': self.vision_confidence_threshold
        }


def create_streamlit_ui():
    """Create Streamlit UI for the multi-modal chatbot"""
    st.set_page_config(
        page_title="Multi-modal Chatbot",
        page_icon="🤖",
        layout="wide"
    )

    st.title("🤖 Multi-modal Chatbot with Google Gemini AI")
    st.markdown("Chat with me using text and images! I can analyze images and provide intelligent responses.")

    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        try:
            st.session_state.chatbot = MultimodalChatbot()
            st.session_state.messages = []
        except Exception as e:
            st.error(f"Failed to initialize chatbot: {e}")
            return

    # File uploader for images
    uploaded_file = st.file_uploader("Upload an image (optional)", type=['png', 'jpg', 'jpeg', 'gif'])

    # Display chat messages
    st.subheader("Chat")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("image"):
                st.image(message["image"])

    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message
        user_message = {"role": "user", "content": prompt}
        if uploaded_file:
            user_message["image"] = Image.open(uploaded_file)

        st.session_state.messages.append(user_message)

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
            if uploaded_file:
                st.image(uploaded_file)

        # Get chatbot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chatbot.chat(prompt, user_message.get("image"))

            st.markdown(response['response'])

            # Add assistant response to session
            assistant_message = {
                "role": "assistant",
                "content": response['response'],
                "metadata": {
                    "confidence": response.get('confidence', 0),
                    "model_used": response.get('model_used', 'unknown'),
                    "processing_time": response.get('processing_time', 0)
                }
            }
            st.session_state.messages.append(assistant_message)

    # Sidebar with stats and controls
    with st.sidebar:
        st.subheader("Chatbot Stats")

        if 'chatbot' in st.session_state:
            stats = st.session_state.chatbot.get_stats()
            st.write(f"Messages in history: {stats['messages_in_history']}")
            st.write(f"Knowledge base: {'Enabled' if stats['knowledge_base_enabled'] else 'Disabled'}")

            if st.button("Clear Chat History"):
                st.session_state.chatbot.clear_history()
                st.session_state.messages.clear()
                st.rerun()

        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This multi-modal chatbot uses:
        - **Google Gemini AI** for text and vision processing
        - **Knowledge base integration** for enhanced responses
        - **Streamlit** for the user interface

        Upload an image and ask questions about it, or just chat normally!
        """)


if __name__ == "__main__":
    create_streamlit_ui()