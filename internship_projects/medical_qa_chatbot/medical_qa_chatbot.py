"""
Medical Q&A Chatbot using MedQuAD Dataset

This module implements a specialized medical question-answering chatbot using the MedQuAD dataset.
It includes data preprocessing, medical entity recognition, and a user interface for asking medical questions.
"""

import os
import json
import time
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone
import re
import streamlit as st

# ML and NLP libraries
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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
    from knowledge_updater.vector_db.manager import VectorDBManager
    from knowledge_updater.embeddings.generator import EmbeddingGenerator
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure you're running from the project root directory")
    print("Try: python -m streamlit run internship_projects/medical_qa_chatbot/medical_qa_chatbot.py")
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

        logger.info("NLTK data download completed")
    except Exception as e:
        logger.error(f"Failed to download NLTK data: {e}")
        # Continue anyway - some functionality might still work

# Download NLTK data on import
download_nltk_data()


class MedicalEntityRecognizer:
    """Simple medical entity recognition for symptoms, diseases, and treatments"""

    def __init__(self):
        self.symptom_keywords = {
            'pain', 'ache', 'hurt', 'sore', 'fever', 'cough', 'headache', 'nausea',
            'vomiting', 'diarrhea', 'constipation', 'fatigue', 'tired', 'weak',
            'dizzy', 'lightheaded', 'numb', 'tingling', 'rash', 'swelling',
            'inflammation', 'bleeding', 'bruising', 'itching', 'burning'
        }

        self.disease_keywords = {
            'diabetes', 'cancer', 'heart disease', 'hypertension', 'asthma',
            'pneumonia', 'bronchitis', 'arthritis', 'alzheimer', 'parkinson',
            'depression', 'anxiety', 'migraine', 'stroke', 'obesity'
        }

        self.treatment_keywords = {
            'medication', 'medicine', 'drug', 'pill', 'tablet', 'injection',
            'surgery', 'operation', 'therapy', 'treatment', 'cure', 'remedy',
            'prescription', 'dosage', 'side effect', 'contraindication'
        }

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract medical entities from text"""
        text_lower = text.lower()
        words = word_tokenize(text_lower)

        entities = {
            'symptoms': [],
            'diseases': [],
            'treatments': []
        }

        for word in words:
            if word in self.symptom_keywords:
                entities['symptoms'].append(word)
            elif word in self.disease_keywords:
                entities['diseases'].append(word)
            elif word in self.treatment_keywords:
                entities['treatments'].append(word)

        # Remove duplicates while preserving order
        for category in entities:
            entities[category] = list(dict.fromkeys(entities[category]))

        return entities


class MedQuADDatasetLoader:
    """Load and preprocess MedQuAD dataset"""

    def __init__(self, dataset_path: Optional[str] = None):
        self.dataset_path = dataset_path or "data/medquad_dataset.json"
        self.data = None
        self.processed_data = []

    def load_dataset(self) -> bool:
        """Load MedQuAD dataset from file or download if not exists"""
        try:
            if os.path.exists(self.dataset_path):
                logger.info(f"Loading dataset from {self.dataset_path}")
                with open(self.dataset_path, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
            else:
                logger.info("Dataset not found locally, attempting to download...")
                return self._download_dataset()

            return self._process_dataset()

        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return False

    def _download_dataset(self) -> bool:
        """Download MedQuAD dataset from GitHub"""
        try:
            import requests

            url = "https://raw.githubusercontent.com/abachaa/MedQuAD/master/data/medquad.json"
            logger.info(f"Downloading dataset from {url}")

            response = requests.get(url)
            response.raise_for_status()

            self.data = response.json()
            logger.info("Dataset downloaded successfully")

            # Save to local file
            os.makedirs(os.path.dirname(self.dataset_path), exist_ok=True)
            with open(self.dataset_path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2)

            return self._process_dataset()

        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            return self._create_sample_dataset()

    def _process_dataset(self) -> bool:
        """Process raw dataset into usable format"""
        if not self.data:
            return False

        try:
            logger.info("Processing MedQuAD dataset")

            for item in self.data:
                # Extract Q&A pairs
                question = item.get('question', '').strip()
                answer = item.get('answer', '').strip()

                if question and answer:
                    processed_item = {
                        'question': question,
                        'answer': answer,
                        'focus': item.get('focus', ''),
                        'type': item.get('type', ''),
                        'source': item.get('source', ''),
                        'id': item.get('id', ''),
                        'processed_at': datetime.now(timezone.utc).isoformat()
                    }

                    self.processed_data.append(processed_item)

            logger.info(f"Processed {len(self.processed_data)} Q&A pairs")
            return True

        except Exception as e:
            logger.error(f"Failed to process dataset: {e}")
            return self._create_sample_dataset()

    def _create_sample_dataset(self) -> bool:
        """Create a sample medical Q&A dataset for demonstration"""
        logger.info("Creating sample medical Q&A dataset")

        sample_qa_pairs = [
            {
                'question': 'What are the common symptoms of diabetes?',
                'answer': 'Common symptoms of diabetes include frequent urination, excessive thirst, extreme hunger, unexplained weight loss, fatigue, irritability, blurred vision, slow-healing sores, and frequent infections.',
                'focus': 'diabetes',
                'type': 'symptoms',
                'source': 'Medical Knowledge Base',
                'id': 'sample_001'
            },
            {
                'question': 'How can I manage high blood pressure?',
                'answer': 'Managing high blood pressure typically involves lifestyle changes such as eating a healthy diet low in sodium, regular exercise, maintaining a healthy weight, limiting alcohol consumption, and managing stress. Medications may also be prescribed by your healthcare provider.',
                'focus': 'hypertension',
                'type': 'management',
                'source': 'Medical Knowledge Base',
                'id': 'sample_002'
            },
            {
                'question': 'What are the side effects of aspirin?',
                'answer': 'Common side effects of aspirin include stomach pain, heartburn, nausea, vomiting, and stomach ulcers. More serious side effects can include allergic reactions, bleeding problems, and ringing in the ears. Always consult your doctor before starting aspirin therapy.',
                'focus': 'medication',
                'type': 'side_effects',
                'source': 'Medical Knowledge Base',
                'id': 'sample_003'
            }
        ]

        self.data = sample_qa_pairs
        os.makedirs(os.path.dirname(self.dataset_path), exist_ok=True)

        with open(self.dataset_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2)

        return self._process_dataset()

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        if not self.processed_data:
            return {}

        questions = [item['question'] for item in self.processed_data]
        answers = [item['answer'] for item in self.processed_data]

        return {
            'total_qa_pairs': len(self.processed_data),
            'avg_question_length': sum(len(q.split()) for q in questions) / len(questions),
            'avg_answer_length': sum(len(a.split()) for a in answers) / len(answers),
            'unique_focus_areas': len(set(item['focus'] for item in self.processed_data)),
            'data_sources': list(set(item['source'] for item in self.processed_data))
        }


class MedicalQABot:
    """Medical Question-Answering Chatbot"""

    def __init__(self, dataset_path: Optional[str] = None):
        """Initialize the medical QA chatbot"""
        self.config = get_config()
        self.logger = get_logger(__name__)

        # Initialize components
        self.dataset_loader = MedQuADDatasetLoader(dataset_path)
        self.entity_recognizer = MedicalEntityRecognizer()
        self.vector_manager = VectorDBManager()
        self.embedding_generator = EmbeddingGenerator()

        # Load dataset
        if not self.dataset_loader.load_dataset():
            raise RuntimeError("Failed to load MedQuAD dataset")

        # Initialize TF-IDF for fallback retrieval
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self._initialize_tfidf()

        # Chatbot settings
        self.confidence_threshold = self.config.get('medical_qa', {}).get('confidence_threshold', 0.7)
        self.max_results = self.config.get('medical_qa', {}).get('max_results', 5)
        self.enable_medical_disclaimer = self.config.get('medical_qa', {}).get('enable_medical_disclaimer', True)

        self.logger.info(
            "Medical QA chatbot initialized",
            dataset_size=len(self.dataset_loader.processed_data),
            confidence_threshold=self.confidence_threshold
        )

    def _initialize_tfidf(self):
        """Initialize TF-IDF vectorizer for fallback retrieval"""
        try:
            questions = [item['question'] for item in self.dataset_loader.processed_data]

            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2)
            )

            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(questions)
            self.logger.info("TF-IDF vectorizer initialized")

        except Exception as e:
            self.logger.error(f"Failed to initialize TF-IDF: {e}")

    def ask_question(self, question: str) -> Dict[str, Any]:
        """
        Answer a medical question

        Args:
            question: Medical question from user

        Returns:
            Answer with confidence and metadata
        """
        start_time = time.time()

        try:
            self.logger.info(f"Processing medical question: {question[:50]}...")

            # Extract medical entities
            entities = self.entity_recognizer.extract_entities(question)

            # Search for similar questions in dataset
            similar_questions = self._find_similar_questions(question)

            if similar_questions:
                # Use vector database results
                best_match = similar_questions[0]
                confidence = best_match.get('similarity', 0)

                if confidence >= self.confidence_threshold:
                    answer = self._generate_medical_response(
                        question,
                        best_match,
                        entities,
                        'vector_search'
                    )
                else:
                    answer = self._generate_uncertain_response(question, similar_questions)
            else:
                # Fallback to TF-IDF search
                tfidf_results = self._tfidf_search(question)
                if tfidf_results:
                    best_match = tfidf_results[0]
                    answer = self._generate_medical_response(
                        question,
                        best_match,
                        entities,
                        'tfidf_search'
                    )
                else:
                    answer = self._generate_no_answer_response(question)

            processing_time = time.time() - start_time

            response_data = {
                'question': question,
                'answer': answer,
                'confidence': similar_questions[0].get('similarity', 0) if similar_questions else 0.5,
                'entities_found': entities,
                'search_method': 'vector_search' if similar_questions else 'tfidf_search',
                'results_found': len(similar_questions) if similar_questions else len(tfidf_results) if tfidf_results else 0,
                'processing_time': processing_time,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

            self.logger.info(
                "Medical question processed",
                confidence=response_data['confidence'],
                search_method=response_data['search_method'],
                processing_time=f"{processing_time:.3f}s"
            )

            return response_data

        except Exception as e:
            self.logger.error(f"Failed to process medical question: {e}")
            return self._generate_error_response(question, str(e))

    def _find_similar_questions(self, question: str) -> List[Dict[str, Any]]:
        """Find similar questions using vector database"""
        try:
            # Generate embedding for the question
            question_embedding = self.embedding_generator.generate_embeddings([question])

            if not question_embedding or not question_embedding[0]:
                return []

            # Search vector database
            results = self.vector_manager.query_knowledge_base(
                query=question,
                n_results=self.max_results,
                similarity_threshold=0.5
            )

            # Format results
            formatted_results = []
            for result in results:
                metadata = result.get('metadata', {})

                formatted_result = {
                    'question': metadata.get('question', ''),
                    'answer': metadata.get('answer', ''),
                    'similarity': result.get('similarity', 0),
                    'focus': metadata.get('focus', ''),
                    'source': metadata.get('source', ''),
                    'id': metadata.get('id', '')
                }

                formatted_results.append(formatted_result)

            return formatted_results

        except Exception as e:
            self.logger.error(f"Vector search failed: {e}")
            return []

    def _tfidf_search(self, question: str) -> List[Dict[str, Any]]:
        """Fallback search using TF-IDF"""
        try:
            if not self.tfidf_vectorizer or not self.tfidf_matrix:
                return []

            # Transform question to TF-IDF vector
            question_vector = self.tfidf_vectorizer.transform([question])

            # Calculate cosine similarity
            similarities = cosine_similarity(question_vector, self.tfidf_matrix).flatten()

            # Get top results
            top_indices = similarities.argsort()[-self.max_results:][::-1]

            results = []
            for idx in top_indices:
                if similarities[idx] > 0.1:  # Minimum similarity threshold
                    item = self.dataset_loader.processed_data[idx]
                    results.append({
                        'question': item['question'],
                        'answer': item['answer'],
                        'similarity': similarities[idx],
                        'focus': item['focus'],
                        'source': item['source'],
                        'id': item['id']
                    })

            return results

        except Exception as e:
            self.logger.error(f"TF-IDF search failed: {e}")
            return []

    def _generate_medical_response(
        self,
        question: str,
        best_match: Dict[str, Any],
        entities: Dict[str, List[str]],
        search_method: str
    ) -> str:
        """Generate medical response from best match"""

        # Add medical disclaimer
        disclaimer = ""
        if self.enable_medical_disclaimer:
            disclaimer = (
                "⚠️ **IMPORTANT MEDICAL DISCLAIMER**: This is not professional medical advice. "
                "Please consult with a qualified healthcare provider for personalized medical guidance.\n\n"
            )

        # Generate response based on entities found
        response_parts = [disclaimer]

        if entities['symptoms']:
            response_parts.append(f"I notice you're asking about symptoms including: {', '.join(entities['symptoms'])}.")

        if entities['diseases']:
            response_parts.append(f"This seems to be related to: {', '.join(entities['diseases'])}.")

        # Add the main answer
        main_answer = best_match.get('answer', 'I don\'t have specific information about this medical question.')
        response_parts.append(main_answer)

        # Add source information
        if best_match.get('source'):
            response_parts.append(f"\n\n*Source: {best_match['source']}*")

        return " ".join(response_parts)

    def _generate_uncertain_response(self, question: str, similar_questions: List[Dict[str, Any]]) -> str:
        """Generate response when confidence is low"""
        response = (
            "I'm not entirely confident about the specific answer to your question, "
            "but here are some related topics that might be helpful:\n\n"
        )

        for i, result in enumerate(similar_questions[:3], 1):
            response += f"{i}. **{result.get('question', 'N/A')}**\n"
            answer_preview = result.get('answer', '')[:200]
            if len(result.get('answer', '')) > 200:
                answer_preview += "..."
            response += f"{answer_preview}\n\n"

        response += (
            "⚠️ **Medical Disclaimer**: This is not professional medical advice. "
            "Please consult with a qualified healthcare provider."
        )

        return response

    def _generate_no_answer_response(self, question: str) -> str:
        """Generate response when no relevant information is found"""
        return (
            "I'm sorry, but I don't have specific information about your medical question in my current knowledge base. "
            "For health-related concerns, it's important to consult with a qualified healthcare professional.\n\n"
            "⚠️ **Medical Disclaimer**: This is not professional medical advice. "
            "Please seek guidance from a licensed healthcare provider for any medical concerns."
        )

    def _generate_error_response(self, question: str, error: str) -> Dict[str, Any]:
        """Generate error response"""
        return {
            'question': question,
            'answer': (
                "I apologize, but I encountered an error while processing your medical question. "
                "Please try asking again or consult with a healthcare professional."
            ),
            'confidence': 0.0,
            'entities_found': {},
            'search_method': 'error',
            'results_found': 0,
            'processing_time': 0,
            'error': error,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        return self.dataset_loader.get_statistics()

    def add_to_knowledge_base(self):
        """Add MedQuAD data to vector database for better search"""
        try:
            logger.info("Adding MedQuAD data to vector database")

            # Prepare data for vector database
            qa_data = []
            for item in self.dataset_loader.processed_data:
                # Create a combined text for embedding
                combined_text = f"Question: {item['question']} Answer: {item['answer']}"

                qa_item = {
                    'id': f"medquad_{item['id']}",
                    'content': combined_text,
                    'title': item['question'][:100],
                    'metadata': {
                        'question': item['question'],
                        'answer': item['answer'],
                        'focus': item['focus'],
                        'type': item['type'],
                        'source': item['source'],
                        'dataset': 'MedQuAD'
                    }
                }

                qa_data.append(qa_item)

            # Update vector database
            update_stats = self.vector_manager.update_knowledge_base(qa_data)

            logger.info(
                "MedQuAD data added to knowledge base",
                articles_added=update_stats.get('vectors_added', 0)
            )

            return update_stats

        except Exception as e:
            logger.error(f"Failed to add MedQuAD data to knowledge base: {e}")
            raise


def create_medical_qa_ui():
    """Create Streamlit UI for Medical Q&A Chatbot"""
    st.set_page_config(
        page_title="Medical Q&A Chatbot",
        page_icon="🏥",
        layout="wide"
    )

    st.title("🏥 Medical Q&A Chatbot")
    st.markdown("*Powered by MedQuAD Dataset*")

    # Medical disclaimer at the top
    st.warning(
        "⚠️ **IMPORTANT MEDICAL DISCLAIMER**: This chatbot provides information for educational purposes only. "
        "It is not a substitute for professional medical advice, diagnosis, or treatment. "
        "Always consult with a qualified healthcare provider for medical concerns."
    )

    # Initialize chatbot
    if 'medical_bot' not in st.session_state:
        with st.spinner("Loading medical knowledge base..."):
            try:
                st.session_state.medical_bot = MedicalQABot()
                st.session_state.chat_history = []
                st.success("Medical chatbot ready!")
            except Exception as e:
                st.error(f"Failed to initialize medical chatbot: {e}")
                return

    # Display dataset statistics
    with st.sidebar:
        st.subheader("📊 Knowledge Base Stats")

        if 'medical_bot' in st.session_state:
            stats = st.session_state.medical_bot.get_dataset_stats()
            st.write(f"Q&A Pairs: {stats.get('total_qa_pairs', 0):,}")
            st.write(f"Avg Question Length: {stats.get('avg_question_length', 0):.1f} words")
            st.write(f"Avg Answer Length: {stats.get('avg_answer_length', 0):.1f} words")
            st.write(f"Focus Areas: {stats.get('unique_focus_areas', 0)}")

            if st.button("🔄 Add to Knowledge Base"):
                with st.spinner("Updating knowledge base..."):
                    update_stats = st.session_state.medical_bot.add_to_knowledge_base()
                    st.success(f"Added {update_stats.get('vectors_added', 0)} medical Q&A pairs!")

    # Chat interface
    st.subheader("Ask a Medical Question")

    # Chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Enhanced question input with medical categories
    col1, col2 = st.columns([3, 1])

    with col1:
        user_question = st.chat_input("Enter your medical question here...")

    with col2:
        if st.button("🩺 Categories", help="Browse medical topics"):
            st.session_state.show_categories = not st.session_state.get('show_categories', False)

    # Show medical categories if requested
    if st.session_state.get('show_categories', False):
        st.write("**Medical Categories:**")
        categories = {
            "Symptoms": ["What are common cold symptoms?", "Signs of diabetes?", "COVID-19 symptoms?"],
            "Treatments": ["How to treat hypertension?", "Medications for asthma?", "Therapy options?"],
            "Medications": ["Side effects of aspirin?", "Drug interactions?", "Dosage information?"],
            "Conditions": ["What is diabetes?", "Heart disease explained?", "Mental health conditions?"]
        }

        for category, examples in categories.items():
            st.write(f"**{category}:**")
            cols = st.columns(len(examples))
            for i, example in enumerate(examples):
                if cols[i].button(example[:30] + "...", key=f"med_{category.lower()}_{i}"):
                    user_question = example
                    st.session_state.show_categories = False
                    st.rerun()

    if user_question:
        # Add user question to history with medical context
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_question,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "category": "medical_inquiry"
        })

        # Display user question with enhanced formatting
        with st.chat_message("user"):
            st.markdown(f"**🩺 Medical Question:** {user_question}")
            st.caption(f"⏰ Asked at {datetime.now().strftime('%H:%M:%S')}")

        # Get answer with medical-themed progress
        with st.chat_message("assistant"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Medical-themed progress updates
            progress_steps = [
                ("📋 Reviewing question...", 20),
                ("🔍 Searching medical database...", 40),
                ("🩺 Analyzing medical entities...", 60),
                ("💊 Formulating medical response...", 80),
                ("✅ Finalizing answer...", 100)
            ]

            for status, progress in progress_steps:
                progress_bar.progress(progress)
                status_text.text(status)
                time.sleep(0.4)

            response = st.session_state.medical_bot.ask_question(user_question)
            progress_bar.empty()
            status_text.empty()

            # Display answer with medical disclaimer
            st.markdown("**💊 Medical Response:**")
            st.markdown(response['answer'])

            # Enhanced medical response details
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Analysis", "🩺 Entities", "📚 Sources", "⚠️ Disclaimer"])

            with tab1:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Confidence Level", f"{response['confidence']:.2f}")
                    st.metric("Search Method", response['search_method'].replace('_', ' ').title())
                with col2:
                    st.metric("Results Found", response['results_found'])
                    st.metric("Processing Time", f"{response['processing_time']:.3f}s")
                with col3:
                    st.metric("Query Category", "Medical Q&A")
                    st.metric("Knowledge Base", "MedQuAD Dataset")

            with tab2:
                if response.get('entities_found'):
                    st.write("**Medical Entities Detected:**")
                    for category, entities in response['entities_found'].items():
                        if entities:
                            with st.expander(f"{category.title()}: {', '.join(entities)}"):
                                st.write(f"**Related Terms:** {', '.join(entities)}")
                                if category == 'symptoms':
                                    st.info("💡 These symptoms may indicate various conditions. Always consult a healthcare provider.")
                                elif category == 'diseases':
                                    st.info("💡 This information is for educational purposes. Professional diagnosis required.")
                                elif category == 'treatments':
                                    st.info("💡 Treatment effectiveness varies by individual. Follow medical advice.")
                else:
                    st.info("No specific medical entities detected in this question.")

            with tab3:
                st.write("**Medical Knowledge Sources:**")
                st.info("📖 Responses are based on the MedQuAD dataset, a collection of medical Q&A pairs.")
                st.info("🔬 Information is derived from medical literature and expert knowledge.")
                st.info("⚕️ Always verify with qualified healthcare professionals.")

            with tab4:
                st.warning(
                    "⚠️ **IMPORTANT MEDICAL DISCLAIMER**: This chatbot provides information for educational purposes only. "
                    "It is not a substitute for professional medical advice, diagnosis, or treatment. "
                    "Always consult with a qualified healthcare provider for medical concerns."
                )

        # Add assistant response to history with medical metadata
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response['answer'],
            "metadata": {
                "confidence": response['confidence'],
                "entities_found": response.get('entities_found', {}),
                "search_method": response.get('search_method'),
                "results_found": response['results_found'],
                "processing_time": response['processing_time']
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "disclaimer_shown": True
        })

        # Smart chat history management
        if len(st.session_state.chat_history) > 20:
            # Keep the most recent 20 messages, prioritizing medical Q&A
            st.session_state.chat_history = st.session_state.chat_history[-20:]

    # Clear chat button
    if st.button("Clear Chat History"):
        st.session_state.chat_history.clear()
        st.rerun()


if __name__ == "__main__":
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        st.warning("Downloading required NLTK data...")
        nltk.download('punkt')
        nltk.download('stopwords')

    create_medical_qa_ui()