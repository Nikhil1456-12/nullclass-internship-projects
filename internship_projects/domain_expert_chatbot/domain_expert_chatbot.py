"""
Domain Expert Chatbot using arXiv Dataset

This module implements a specialized chatbot that serves as an expert in computer science,
capable of discussing advanced topics, providing research paper summaries, and explaining complex concepts.
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
    print("Try: python -m streamlit run internship_projects/domain_expert_chatbot/domain_expert_chatbot.py")
    sys.exit(1)

logger = get_logger(__name__)

# Download NLTK data on import
download_nltk_data()


class ArXivDatasetLoader:
    """Load and preprocess arXiv dataset for computer science domain"""

    def __init__(self, dataset_path: Optional[str] = None, subset_size: int = 10000):
        self.dataset_path = dataset_path or "data/arxiv_cs_dataset.json"
        self.subset_size = subset_size  # Limit dataset size for manageable processing
        self.data = None
        self.processed_data = []
        self.categories = [
            'cs.AI', 'cs.CL', 'cs.CV', 'cs.LG', 'cs.NE',  # AI & Machine Learning
            'cs.SE', 'cs.PL', 'cs.OS', 'cs.AR',  # Software & Systems
            'cs.CR', 'cs.CY', 'cs.DB', 'cs.DC',  # Security & Databases
            'cs.DS', 'cs.CC', 'cs.CG', 'cs.GT'   # Algorithms & Theory
        ]

    def load_dataset(self) -> bool:
        """Load arXiv dataset from file or download if not exists"""
        try:
            if os.path.exists(self.dataset_path):
                logger.info(f"Loading dataset from {self.dataset_path}")
                with open(self.dataset_path, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
            else:
                logger.info("Dataset not found locally, attempting to download...")
                return self._download_and_process_arxiv()

            return self._process_dataset()

        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return False

    def _download_and_process_arxiv(self) -> bool:
        """Download and process arXiv dataset using Kaggle API"""
        try:
            import kaggle

            logger.info("Downloading arXiv dataset using Kaggle API")

            # Download dataset
            kaggle.api.dataset_download_files(
                'Cornell-University/arxiv',
                path='data/arxiv_temp',
                unzip=True
            )

            # Process the downloaded data
            return self._process_downloaded_data()

        except Exception as e:
            logger.error(f"Failed to download arXiv dataset: {e}")
            return self._create_sample_dataset()

    def _process_downloaded_data(self) -> bool:
        """Process downloaded arXiv data"""
        try:
            # Look for JSON files in the downloaded data
            data_dir = "data/arxiv_temp"
            json_files = []

            for root, dirs, files in os.walk(data_dir):
                for file in files:
                    if file.endswith('.json'):
                        json_files.append(os.path.join(root, file))

            if not json_files:
                logger.warning("No JSON files found in downloaded data")
                return self._create_sample_dataset()

            # Process the first JSON file (usually contains the main dataset)
            with open(json_files[0], 'r', encoding='utf-8') as f:
                raw_data = json.load(f)

            # Filter for computer science papers and limit size
            cs_papers = []
            for paper in raw_data:
                if any(cat in paper.get('categories', '') for cat in self.categories):
                    cs_papers.append(paper)
                    if len(cs_papers) >= self.subset_size:
                        break

            # Save processed data
            self.data = cs_papers
            os.makedirs(os.path.dirname(self.dataset_path), exist_ok=True)

            with open(self.dataset_path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2)

            return self._process_dataset()

        except Exception as e:
            logger.error(f"Failed to process downloaded data: {e}")
            return self._create_sample_dataset()

    def _create_sample_dataset(self) -> bool:
        """Create a sample dataset for demonstration"""
        logger.info("Creating sample arXiv computer science dataset")

        sample_papers = [
            {
                'id': 'cs-ai-001',
                'title': 'Attention Is All You Need',
                'abstract': 'The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train.',
                'categories': ['cs.CL', 'cs.LG', 'cs.AI'],
                'authors': ['Vaswani et al.'],
                'published': '2017-06-12',
                'updated': '2017-06-12'
            },
            {
                'id': 'cs-ai-002',
                'title': 'Artificial Intelligence: A Modern Approach',
                'abstract': 'Artificial Intelligence (AI) is a broad field of computer science focused on creating systems capable of performing tasks that typically require human intelligence. These tasks include learning, reasoning, problem-solving, perception, and language understanding. Modern AI systems use machine learning algorithms to process data and make predictions or decisions without being explicitly programmed for each specific task.',
                'categories': ['cs.AI'],
                'authors': ['Russell et al.'],
                'published': '2020-01-01',
                'updated': '2020-01-01'
            },
            {
                'id': 'cs-ml-001',
                'title': 'Machine Learning: An Overview',
                'abstract': 'Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from and make predictions on data. These algorithms build mathematical models based on training data to make predictions or decisions without being explicitly programmed to perform the task. Machine learning is widely used in various applications including image recognition, natural language processing, recommendation systems, and autonomous vehicles.',
                'categories': ['cs.LG', 'cs.AI'],
                'authors': ['Mitchell et al.'],
                'published': '2019-03-15',
                'updated': '2019-03-15'
            },
            {
                'id': 'cs-nn-001',
                'title': 'Neural Networks and Deep Learning',
                'abstract': 'Neural networks are computing systems inspired by biological neural networks. They consist of layers of interconnected nodes (neurons) that process and transmit information. Deep learning refers to neural networks with multiple hidden layers that can learn complex patterns and representations from large amounts of data. These models have revolutionized fields like computer vision, natural language processing, and speech recognition.',
                'categories': ['cs.LG', 'cs.NE', 'cs.AI'],
                'authors': ['Goodfellow et al.'],
                'published': '2018-07-20',
                'updated': '2018-07-20'
            },
            {
                'id': 'cs-cv-001',
                'title': 'Computer Vision: Algorithms and Applications',
                'abstract': 'Computer vision is a field of artificial intelligence that trains computers to interpret and understand the visual world. Using digital images and videos, computer vision algorithms can identify objects, faces, scenes, and activities. Applications include autonomous driving, medical image analysis, surveillance systems, and augmented reality.',
                'categories': ['cs.CV', 'cs.AI'],
                'authors': ['Szeliski et al.'],
                'published': '2021-11-01',
                'updated': '2021-11-01'
            },
            {
                'id': 'cs-cv-002',
                'title': 'ImageNet Classification with Deep Convolutional Neural Networks',
                'abstract': 'We trained a large, deep convolutional neural network to classify the 1.2 million high-resolution images in the ImageNet LSVRC-2010 contest into 1000 different classes. Our network achieved a top-5 error rate of 15.3%, more than 10.8 percentage points lower than that of the runner up. The neural network, which has 60 million parameters and 650,000 neurons, consists of five convolutional layers, some of which are followed by max-pooling layers, and three fully-connected layers with a final 1000-way softmax.',
                'categories': ['cs.CV'],
                'authors': ['Krizhevsky et al.'],
                'published': '2012-12-10',
                'updated': '2012-12-10'
            }
        ]

        self.data = sample_papers
        os.makedirs(os.path.dirname(self.dataset_path), exist_ok=True)

        with open(self.dataset_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2)

        return self._process_dataset()

    def _process_dataset(self) -> bool:
        """Process raw dataset into usable format"""
        if not self.data:
            return False

        try:
            logger.info("Processing arXiv computer science dataset")

            for paper in self.data:
                # Extract and clean paper information
                title = paper.get('title', '').strip()
                abstract = paper.get('abstract', '').strip()

                if title and abstract:
                    processed_paper = {
                        'id': paper.get('id', ''),
                        'title': title,
                        'abstract': abstract,
                        'categories': paper.get('categories', []),
                        'authors': paper.get('authors', []),
                        'published_date': paper.get('published', ''),
                        'updated_date': paper.get('updated', ''),
                        'summary': self._generate_summary(title, abstract),
                        'keywords': self._extract_keywords(title + ' ' + abstract),
                        'complexity_level': self._assess_complexity(abstract),
                        'processed_at': datetime.now(timezone.utc).isoformat()
                    }

                    self.processed_data.append(processed_paper)

            logger.info(f"Processed {len(self.processed_data)} research papers")
            return True

        except Exception as e:
            logger.error(f"Failed to process dataset: {e}")
            return False

    def _generate_summary(self, title: str, abstract: str) -> str:
        """Generate a brief summary of the paper"""
        # Simple extractive summarization - take first sentence + key sentences
        try:
            sentences = sent_tokenize(abstract)
            if len(sentences) <= 2:
                return abstract

            summary = sentences[0]  # First sentence
            if len(sentences) > 1:
                summary += ' ' + sentences[1]  # Second sentence

            return summary[:300] + '...' if len(summary) > 300 else summary
        except (LookupError, AttributeError) as e:
            # Fallback if NLTK tokenization fails
            logger.warning(f"NLTK tokenization failed: {e}. Using fallback method.")
            # Simple fallback: split by periods and take first two sentences
            simple_sentences = abstract.split('.')
            if len(simple_sentences) <= 2:
                return abstract

            summary = simple_sentences[0] + '.' + simple_sentences[1] + '.'
            return summary[:300] + '...' if len(summary) > 300 else summary

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract key technical terms from text"""
        # Simple keyword extraction based on technical terms
        technical_terms = {
            'algorithm', 'neural network', 'machine learning', 'deep learning',
            'computer vision', 'natural language processing', 'reinforcement learning',
            'optimization', 'classification', 'regression', 'clustering',
            'artificial intelligence', 'data mining', 'pattern recognition',
            'feature extraction', 'dimensionality reduction', 'ensemble methods'
        }

        text_lower = text.lower()
        found_terms = []

        for term in technical_terms:
            if term in text_lower and term not in found_terms:
                found_terms.append(term)

        return found_terms[:10]  # Return top 10 keywords

    def _assess_complexity(self, abstract: str) -> str:
        """Assess the complexity level of the paper"""
        # Simple complexity assessment based on text features
        try:
            words = word_tokenize(abstract.lower())
            # Count technical terms
            technical_count = sum(1 for word in words if len(word) > 8)
        except (LookupError, AttributeError) as e:
            # Fallback if NLTK tokenization fails
            logger.warning(f"NLTK tokenization failed: {e}. Using fallback method.")
            words = abstract.lower().split()
            # Count technical terms using simple split
            technical_count = sum(1 for word in words if len(word) > 8)

        # Count mathematical symbols (rough approximation)
        math_indicators = ['algorithm', 'theorem', 'lemma', 'proof', 'equation', 'formula']

        if technical_count > 20 or any(indicator in abstract.lower() for indicator in math_indicators):
            return 'advanced'
        elif technical_count > 10:
            return 'intermediate'
        else:
            return 'beginner'

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        if not self.processed_data:
            return {}

        categories = {}
        for paper in self.processed_data:
            for cat in paper['categories']:
                categories[cat] = categories.get(cat, 0) + 1

        return {
            'total_papers': len(self.processed_data),
            'categories': categories,
            'avg_title_length': sum(len(p['title'].split()) for p in self.processed_data) / len(self.processed_data),
            'avg_abstract_length': sum(len(p['abstract'].split()) for p in self.processed_data) / len(self.processed_data),
            'complexity_distribution': {
                'beginner': sum(1 for p in self.processed_data if p['complexity_level'] == 'beginner'),
                'intermediate': sum(1 for p in self.processed_data if p['complexity_level'] == 'intermediate'),
                'advanced': sum(1 for p in self.processed_data if p['complexity_level'] == 'advanced')
            }
        }


class DomainExpertChatbot:
    """Computer Science Domain Expert Chatbot"""

    def __init__(self, dataset_path: Optional[str] = None):
        """Initialize the domain expert chatbot"""
        self.config = get_config()
        self.logger = get_logger(__name__)

        # Initialize components
        self.dataset_loader = ArXivDatasetLoader(dataset_path)
        self.vector_manager = VectorDBManager()
        self.embedding_generator = EmbeddingGenerator()

        # Load dataset
        if not self.dataset_loader.load_dataset():
            raise RuntimeError("Failed to load arXiv dataset")

        # Initialize TF-IDF for fallback retrieval
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self._initialize_tfidf()

        # Chatbot settings
        self.confidence_threshold = self.config.get('domain_expert', {}).get('confidence_threshold', 0.7)
        self.max_results = self.config.get('domain_expert', {}).get('max_results', 5)
        self.domain = "Computer Science"
        self.subdomains = ['AI', 'Machine Learning', 'Computer Vision', 'NLP', 'Systems', 'Theory']

        self.logger.info(
            "Domain expert chatbot initialized",
            domain=self.domain,
            papers_loaded=len(self.dataset_loader.processed_data),
            confidence_threshold=self.confidence_threshold
        )

    def _initialize_tfidf(self):
        """Initialize TF-IDF vectorizer for fallback retrieval"""
        try:
            # Combine title and abstract for better search
            papers_text = [
                f"{paper['title']} {paper['abstract']}"
                for paper in self.dataset_loader.processed_data
            ]

            # Check if we have papers to process
            if not papers_text:
                self.logger.warning("No papers available for TF-IDF initialization")
                return

            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2)
            )

            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(papers_text)
            self.logger.info("TF-IDF vectorizer initialized for domain expert")

        except Exception as e:
            self.logger.error(f"Failed to initialize TF-IDF: {e}")

    def chat(self, query: str, conversation_context: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Process a domain-specific query

        Args:
            query: User's question about computer science
            conversation_context: Previous conversation messages

        Returns:
            Expert response with citations and explanations
        """
        start_time = time.time()

        try:
            self.logger.info(f"Processing domain query: {query[:50]}...")

            # Classify query type
            query_type = self._classify_query_type(query)

            # Find relevant papers
            relevant_papers = self._find_relevant_papers(query)

            if relevant_papers:
                # Generate expert response
                response_data = self._generate_expert_response(
                    query,
                    relevant_papers,
                    query_type,
                    conversation_context
                )
            else:
                response_data = self._generate_no_results_response(query)

            processing_time = time.time() - start_time

            response_data.update({
                'query_type': query_type,
                'processing_time': processing_time,
                'timestamp': datetime.now(timezone.utc).isoformat()
            })

            self.logger.info(
                "Domain query processed",
                query_type=query_type,
                results_found=len(relevant_papers),
                processing_time=f"{processing_time:.3f}s"
            )

            return response_data

        except Exception as e:
            self.logger.error(f"Failed to process domain query: {e}")
            return self._generate_error_response(query, str(e))

    def _classify_query_type(self, query: str) -> str:
        """Classify the type of query"""
        query_lower = query.lower()

        # Paper search queries
        if any(word in query_lower for word in ['paper', 'research', 'study', 'publication', 'arxiv']):
            return 'paper_search'

        # Explanation queries
        if any(word in query_lower for word in ['explain', 'what is', 'how does', 'describe', 'understand']):
            return 'explanation'

        # Comparison queries
        if any(word in query_lower for word in ['compare', 'difference', 'better', 'versus', 'vs']):
            return 'comparison'

        # Latest trends queries
        if any(word in query_lower for word in ['latest', 'recent', 'current', 'trends', 'advances']):
            return 'trends'

        return 'general'

    def _find_relevant_papers(self, query: str) -> List[Dict[str, Any]]:
        """Find relevant papers using vector search"""
        try:
            # Generate embedding for the query
            query_embedding = self.embedding_generator.generate_embeddings([query])

            if not query_embedding or len(query_embedding) == 0 or not query_embedding[0]:
                self.logger.warning("Failed to generate query embedding, returning empty results")
                return []

            # Search vector database
            results = self.vector_manager.query_knowledge_base(
                query=query,
                n_results=self.max_results,
                similarity_threshold=0.5
            )

            # Format results with paper information
            formatted_results = []
            for result in results:
                metadata = result.get('metadata', {})

                formatted_result = {
                    'title': metadata.get('title', ''),
                    'abstract': metadata.get('abstract', ''),
                    'authors': metadata.get('authors', []),
                    'categories': metadata.get('categories', []),
                    'published_date': metadata.get('published_date', ''),
                    'summary': metadata.get('summary', ''),
                    'keywords': metadata.get('keywords', []),
                    'complexity_level': metadata.get('complexity_level', 'intermediate'),
                    'similarity': result.get('similarity', 0),
                    'id': metadata.get('id', '')
                }

                formatted_results.append(formatted_result)

            return formatted_results

        except Exception as e:
            self.logger.error(f"Paper search failed: {e}")
            return []

    def _generate_expert_response(
        self,
        query: str,
        papers: List[Dict[str, Any]],
        query_type: str,
        conversation_context: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Generate expert response based on query type"""

        if query_type == 'paper_search':
            response = self._generate_paper_search_response(query, papers)
        elif query_type == 'explanation':
            response = self._generate_explanation_response(query, papers)
        elif query_type == 'comparison':
            response = self._generate_comparison_response(query, papers)
        elif query_type == 'trends':
            response = self._generate_trends_response(query, papers)
        else:
            response = self._generate_general_response(query, papers)

        # Generate citations
        citations = self._generate_citations(papers)

        return {
            'response': response,
            'confidence': min(papers[0].get('similarity', 0.5) if papers else 0.5, 1.0),
            'papers_found': len(papers),
            'citations': citations,
            'query_type': query_type,
            'domain': self.domain
        }

    def _generate_paper_search_response(self, query: str, papers: List[Dict[str, Any]]) -> str:
        """Generate response for paper search queries"""
        if not papers:
            return "I couldn't find specific research papers matching your query in my current knowledge base."

        response = f"I found {len(papers)} relevant research papers related to your query:\n\n"

        for i, paper in enumerate(papers[:3], 1):  # Top 3 papers
            response += f"{i}. **{paper['title']}**\n"
            response += f"   Authors: {', '.join(paper['authors'][:3])}\n"
            response += f"   Categories: {', '.join(paper['categories'])}\n"
            response += f"   Summary: {paper['summary']}\n\n"

        if len(papers) > 3:
            response += f"... and {len(papers) - 3} more papers. Ask me to dive deeper into any specific paper!"

        return response

    def _generate_explanation_response(self, query: str, papers: List[Dict[str, Any]]) -> str:
        """Generate response for explanation queries"""
        if not papers:
            return f"I don't have specific research context to explain this computer science concept in detail."

        # Use the most relevant paper for explanation
        best_paper = papers[0]

        response = f"Based on research literature, here's an explanation of the concept you're asking about:\n\n"
        response += f"**Key Research**: {best_paper['title']}\n\n"
        response += f"{best_paper['abstract'][:500]}...\n\n"

        # Add complexity-appropriate explanation
        complexity = best_paper.get('complexity_level', 'intermediate')
        if complexity == 'advanced':
            response += "This is an advanced topic often discussed in research papers. "
        elif complexity == 'beginner':
            response += "This concept is accessible even to those new to the field. "

        response += "Would you like me to explain any specific aspect in more detail?"

        return response

    def _generate_comparison_response(self, query: str, papers: List[Dict[str, Any]]) -> str:
        """Generate response for comparison queries"""
        if len(papers) < 2:
            return "I need at least two research papers to make a meaningful comparison."

        response = "Here's a research-based comparison based on relevant papers:\n\n"

        for i, paper in enumerate(papers[:3], 1):
            response += f"{i}. **{paper['title']}**\n"
            response += f"   Approach: {paper['categories']}\n"
            response += f"   Key Insight: {paper['summary'][:200]}...\n\n"

        response += "Each approach has its strengths depending on the specific use case and requirements."

        return response

    def _generate_trends_response(self, query: str, papers: List[Dict[str, Any]]) -> str:
        """Generate response for trends queries"""
        # Sort papers by publication date (most recent first)
        recent_papers = sorted(
            papers,
            key=lambda x: x.get('published_date', ''),
            reverse=True
        )[:3]

        response = "Based on recent research trends in computer science:\n\n"

        for paper in recent_papers:
            response += f"• **{paper['title']}**\n"
            response += f"  Published: {paper['published_date']}\n"
            response += f"  Focus: {', '.join(paper['categories'])}\n\n"

        response += "These recent publications indicate current research directions in the field."

        return response

    def _generate_general_response(self, query: str, papers: List[Dict[str, Any]]) -> str:
        """Generate general domain response"""
        if not papers:
            return f"As a computer science expert, I can help you understand various topics in {self.domain}. Could you be more specific about what you'd like to know?"

        # Check if we have high-quality matches
        best_paper = papers[0]
        best_similarity = best_paper.get('similarity', 0)

        if best_similarity < 0.3:  # Low similarity threshold
            return ("I don't have specific research papers that closely match your query in my current knowledge base. "
                   "However, I'm happy to discuss general computer science concepts or help you understand "
                   "fundamental principles in the field. Could you try asking about a specific topic like "
                   "'machine learning', 'neural networks', 'computer vision', or 'artificial intelligence'?")

        response = f"Based on current research in computer science, here's what I can share about your query:\n\n"
        response += f"The most relevant research I found is: **{best_paper['title']}**\n\n"
        response += f"{best_paper['summary']}\n\n"

        if len(papers) > 1:
            response += f"I also found {len(papers) - 1} other related papers on this topic."

        return response

    def _generate_citations(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate citations for the papers"""
        citations = []

        for paper in papers[:5]:  # Limit to top 5
            citation = {
                'title': paper.get('title', ''),
                'authors': paper.get('authors', []),
                'categories': paper.get('categories', []),
                'published_date': paper.get('published_date', ''),
                'similarity': paper.get('similarity', 0.0)
            }
            citations.append(citation)

        return citations

    def _generate_no_results_response(self, query: str) -> Dict[str, Any]:
        """Generate response when no results are found"""
        return {
            'response': (
                "I don't have specific research papers in my knowledge base that directly address your query. "
                "However, I'm happy to discuss general computer science concepts or help you understand "
                "fundamental principles in the field. Could you rephrase your question or ask about a specific topic?"
            ),
            'confidence': 0.3,
            'papers_found': 0,
            'citations': [],
            'query_type': 'no_results',
            'domain': self.domain
        }

    def _generate_error_response(self, query: str, error: str) -> Dict[str, Any]:
        """Generate error response"""
        return {
            'response': (
                "I apologize, but I encountered an error while processing your computer science question. "
                "Please try asking again or consult academic resources for detailed technical information."
            ),
            'confidence': 0.0,
            'papers_found': 0,
            'citations': [],
            'query_type': 'error',
            'error': error,
            'domain': self.domain
        }

    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        return self.dataset_loader.get_statistics()

    def add_to_knowledge_base(self):
        """Add arXiv papers to vector database"""
        try:
            logger.info("Adding arXiv papers to vector database")

            # Prepare papers for vector database
            papers_data = []
            for paper in self.dataset_loader.processed_data:
                # Create searchable text
                searchable_text = f"{paper['title']} {paper['abstract']} {paper['summary']}"

                paper_item = {
                    'id': f"arxiv_{paper['id']}",
                    'content': searchable_text,
                    'title': paper['title'],
                    'metadata': {
                        'title': paper['title'],
                        'abstract': paper['abstract'],
                        'summary': paper['summary'],
                        'categories': paper['categories'],
                        'authors': paper['authors'],
                        'published_date': paper['published_date'],
                        'keywords': paper['keywords'],
                        'complexity_level': paper['complexity_level'],
                        'dataset': 'arXiv'
                    }
                }

                papers_data.append(paper_item)

            # Update vector database
            update_stats = self.vector_manager.update_knowledge_base(papers_data)

            logger.info(
                "arXiv papers added to knowledge base",
                papers_added=update_stats.get('vectors_added', 0)
            )

            return update_stats

        except Exception as e:
            logger.error(f"Failed to add arXiv papers to knowledge base: {e}")
            raise


def create_domain_expert_ui():
    """Create Streamlit UI for Domain Expert Chatbot"""
    st.set_page_config(
        page_title="Computer Science Expert Chatbot",
        page_icon="🧠",
        layout="wide"
    )

    st.title("🧠 Computer Science Expert Chatbot")
    st.markdown("*Powered by arXiv Research Papers*")

    # Initialize chatbot
    if 'expert_bot' not in st.session_state:
        with st.spinner("Loading computer science knowledge base..."):
            try:
                st.session_state.expert_bot = DomainExpertChatbot()
                st.session_state.chat_history = []
                st.success("Computer Science Expert ready!")
            except Exception as e:
                st.error(f"Failed to initialize expert chatbot: {e}")
                return

    # Display dataset statistics
    with st.sidebar:
        st.subheader("📊 Knowledge Base Stats")

        if 'expert_bot' in st.session_state:
            stats = st.session_state.expert_bot.get_dataset_stats()

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Papers", stats.get('total_papers', 0))
                st.metric("Categories", len(stats.get('categories', {})))

            with col2:
                st.metric("Avg Abstract", f"{stats.get('avg_abstract_length', 0):.0f} words")
                complexity_dist = stats.get('complexity_distribution', {})
                st.metric("Advanced Papers", complexity_dist.get('advanced', 0))

            # Category breakdown
            if stats.get('categories'):
                st.write("**Top Categories:**")
                for cat, count in sorted(stats['categories'].items(), key=lambda x: x[1], reverse=True)[:5]:
                    st.write(f"{cat}: {count}")

            if st.button("🔄 Add to Knowledge Base"):
                with st.spinner("Updating knowledge base..."):
                    update_stats = st.session_state.expert_bot.add_to_knowledge_base()
                    st.success(f"Added {update_stats.get('vectors_added', 0)} research papers!")

    # Chat interface
    st.subheader("Ask me about Computer Science")

    # Chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Enhanced question input with suggestions
    col1, col2 = st.columns([3, 1])

    with col1:
        user_query = st.chat_input("Ask me about AI, machine learning, algorithms, or any CS topic...")

    with col2:
        # Quick suggestion buttons
        if st.button("💡 Suggestions", help="Get topic suggestions"):
            st.session_state.show_suggestions = not st.session_state.get('show_suggestions', False)

    # Show suggestions if requested
    if st.session_state.get('show_suggestions', False):
        st.write("**Try asking about:**")
        suggestions = [
            "What is the latest in computer vision?",
            "Explain neural networks simply",
            "Compare different machine learning algorithms",
            "What are the current trends in AI research?"
        ]

        cols = st.columns(2)
        for i, suggestion in enumerate(suggestions):
            if cols[i % 2].button(suggestion[:40] + "...", key=f"sugg_{i}"):
                user_query = suggestion
                st.session_state.show_suggestions = False
                st.rerun()

    if user_query:
        # Add user query to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_query,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        # Display user query with timestamp
        with st.chat_message("user"):
            st.markdown(f"**{user_query}**")
            st.caption(f"Asked at {datetime.now().strftime('%H:%M:%S')}")

        # Get expert response with progress indication
        with st.chat_message("assistant"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Simulate progress updates
            for i in range(5):
                progress_bar.progress((i + 1) * 20)
                if i == 0:
                    status_text.text("🔍 Searching research papers...")
                elif i == 1:
                    status_text.text("🧠 Analyzing content...")
                elif i == 2:
                    status_text.text("✍️ Formulating response...")
                elif i == 3:
                    status_text.text("🔗 Adding citations...")
                else:
                    status_text.text("✅ Finalizing response...")
                time.sleep(0.2)

            response = st.session_state.expert_bot.chat(user_query)
            progress_bar.empty()
            status_text.empty()

            # Display response with enhanced formatting
            st.markdown(f"**🤖 Expert Response:**")
            st.markdown(response['response'])

            # Enhanced response details in tabs
            tab1, tab2, tab3 = st.tabs(["📊 Details", "📚 Citations", "🔍 Analysis"])

            with tab1:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Confidence Score", f"{response['confidence']:.2f}")
                    st.metric("Query Type", response['query_type'].replace('_', ' ').title())
                with col2:
                    st.metric("Papers Found", response['papers_found'])
                    st.metric("Processing Time", f"{response['processing_time']:.3f}s")
                with col3:
                    st.metric("Domain", response.get('domain', 'Computer Science'))
                    st.metric("Timestamp", datetime.now().strftime('%H:%M:%S'))

            with tab2:
                if response.get('citations'):
                    for i, citation in enumerate(response['citations'][:5], 1):
                        with st.expander(f"📄 Paper {i}: {citation['title'][:60]}..."):
                            st.write(f"**Authors:** {', '.join(citation['authors'][:3])}")
                            st.write(f"**Categories:** {', '.join(citation['categories'])}")
                            st.write(f"**Similarity Score:** {citation['similarity']:.3f}")
                            if citation.get('published_date'):
                                st.write(f"**Published:** {citation['published_date']}")
                else:
                    st.info("No specific citations available for this response.")

            with tab3:
                st.write("**Query Analysis:**")
                st.json({
                    "original_query": user_query,
                    "classified_as": response.get('query_type', 'unknown'),
                    "confidence_level": response.get('confidence', 0),
                    "search_method": "Vector similarity matching",
                    "knowledge_base": "arXiv Computer Science papers"
                })

        # Add assistant response to history with metadata
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response['response'],
            "metadata": {
                "confidence": response['confidence'],
                "papers_found": response['papers_found'],
                "processing_time": response['processing_time'],
                "query_type": response.get('query_type')
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        # Limit chat history with smart truncation
        if len(st.session_state.chat_history) > 20:
            # Keep the most recent 20 messages
            st.session_state.chat_history = st.session_state.chat_history[-20:]

    # Clear chat button
    if st.button("Clear Chat History"):
        st.session_state.chat_history.clear()
        st.rerun()


if __name__ == "__main__":
    create_domain_expert_ui()