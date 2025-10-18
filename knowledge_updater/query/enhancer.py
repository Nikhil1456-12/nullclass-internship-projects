"""
Query enhancer for integrating knowledge base with chatbot responses
"""

import time
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone
import asyncio

from ..core.config import get_config
from ..core.logging import get_logger, LogContext
from ..vector_db.manager import VectorDBManager
from ..embeddings.generator import EmbeddingGenerator

logger = get_logger(__name__)


class QueryEnhancer:
    """Enhances chatbot queries with knowledge base information"""

    def __init__(self):
        """Initialize query enhancer"""
        self.config = get_config()
        self.query_config = self.config.get('query_enhancement', {})

        # Initialize managers
        self.vector_manager = VectorDBManager()
        self.embedding_generator = EmbeddingGenerator()

        # Get configuration
        self.max_context_results = self.query_config.get('max_context_results', 5)
        self.similarity_threshold = self.query_config.get('similarity_threshold', 0.7)
        self.max_response_time_ms = self.query_config.get('max_response_time_ms', 500)
        self.include_metadata = self.query_config.get('include_metadata', True)

        logger.info(
            "Query enhancer initialized",
            max_context_results=self.max_context_results,
            similarity_threshold=self.similarity_threshold
        )

    def enhance_query(
        self,
        query: str,
        conversation_context: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Enhance a query with relevant knowledge base information

        Args:
            query: User's query string
            conversation_context: Previous conversation messages
            **kwargs: Additional parameters

        Returns:
            Enhanced query with context information
        """
        start_time = time.time()

        try:
            with LogContext(logger, query_length=len(query)):
                logger.debug("Enhancing query", query=query[:50])

                # Extract query features
                query_features = self._extract_query_features(query, conversation_context)

                # Search knowledge base
                context_results = self._search_knowledge_base(query, query_features)

                # Filter and rank results
                filtered_results = self._filter_and_rank_results(context_results, query_features)

                # Generate enhanced response data
                enhanced_data = self._generate_enhanced_response(
                    query,
                    filtered_results,
                    query_features
                )

                enhancement_time = time.time() - start_time
                enhancement_time_ms = enhancement_time * 1000

                # Check if we're within response time limits
                if enhancement_time_ms > self.max_response_time_ms:
                    logger.warning(
                        "Query enhancement exceeded time limit",
                        enhancement_time_ms=enhancement_time_ms,
                        limit_ms=self.max_response_time_ms
                    )

                logger.info(
                    "Query enhancement completed",
                    query_length=len(query),
                    results_found=len(filtered_results),
                    enhancement_time_ms=f"{enhancement_time_ms:.1f}",
                    within_time_limit=enhancement_time_ms <= self.max_response_time_ms
                )

                return enhanced_data

        except Exception as e:
            enhancement_time = time.time() - start_time
            logger.error(
                "Query enhancement failed",
                enhancement_time_ms=f"{enhancement_time * 1000:.1f}",
                error=str(e)
            )

            # Return minimal enhancement data on error
            return {
                'original_query': query,
                'enhanced_query': query,
                'context': [],
                'confidence': 0.0,
                'enhancement_time_ms': enhancement_time * 1000,
                'error': str(e)
            }

    def _extract_query_features(
        self,
        query: str,
        conversation_context: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Extract features from the query for better search"""
        features = {
            'keywords': self._extract_keywords(query),
            'entities': self._extract_entities(query),
            'query_type': self._classify_query_type(query),
            'temporal_indicators': self._extract_temporal_indicators(query),
            'complexity': self._assess_query_complexity(query)
        }

        # Add conversation context if available
        if conversation_context:
            features['conversation_context'] = conversation_context
            features['conversation_length'] = len(conversation_context)

        return features

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from query"""
        # Simple keyword extraction - remove stop words
        stop_words = {
            'what', 'when', 'where', 'who', 'why', 'how', 'is', 'are', 'was', 'were',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'tell', 'me', 'about', 'can', 'you', 'please'
        }

        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [
            word for word in words
            if len(word) > 2 and word not in stop_words
        ]

        return list(set(keywords))[:10]  # Return top 10 unique keywords

    def _extract_entities(self, query: str) -> List[str]:
        """Extract named entities from query (simplified)"""
        entities = []

        # Look for capitalized words (potential proper nouns)
        words = query.split()
        for word in words:
            word_clean = re.sub(r'[^\w]', '', word)
            if (word_clean and word_clean[0].isupper() and len(word_clean) > 2):
                entities.append(word_clean)

        # Look for common entity patterns
        patterns = [
            r'\b\d{4}\b',  # Years
            r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # Names
        ]

        for pattern in patterns:
            matches = re.findall(pattern, query)
            entities.extend(matches)

        return list(set(entities))[:5]  # Return top 5 entities

    def _classify_query_type(self, query: str) -> str:
        """Classify the type of query"""
        query_lower = query.lower()

        # Question classification
        if query_lower.startswith(('what', 'which', 'who', 'where', 'when', 'why', 'how')):
            return 'question'

        # Command classification
        if query_lower.startswith(('tell', 'show', 'give', 'list', 'find', 'search')):
            return 'command'

        # Statement classification
        if any(word in query_lower for word in ['think', 'believe', 'feel', 'know']):
            return 'opinion'

        return 'general'

    def _extract_temporal_indicators(self, query: str) -> List[str]:
        """Extract temporal indicators from query"""
        temporal_words = [
            'today', 'yesterday', 'tomorrow', 'recently', 'lately', 'now', 'current',
            'latest', 'newest', 'recent', 'old', 'previous', 'last', 'next', 'week',
            'month', 'year', 'day', 'hour', 'minute', 'second'
        ]

        indicators = []
        query_lower = query.lower()

        for word in temporal_words:
            if word in query_lower:
                indicators.append(word)

        return indicators

    def _assess_query_complexity(self, query: str) -> str:
        """Assess query complexity"""
        word_count = len(query.split())
        has_questions = '?' in query
        has_complex_words = len([w for w in query.split() if len(w) > 6]) > 0

        if word_count <= 3:
            return 'simple'
        elif has_questions and has_complex_words:
            return 'complex'
        else:
            return 'moderate'

    def _search_knowledge_base(
        self,
        query: str,
        query_features: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Search the knowledge base for relevant information"""
        try:
            # Build search query
            search_query = query

            # Enhance search query with keywords if available
            keywords = query_features.get('keywords', [])
            if keywords:
                # Add important keywords to search
                keyword_boost = ' '.join(keywords[:3])
                if keyword_boost:
                    search_query = f"{query} {keyword_boost}"

            # Search vector database
            results = self.vector_manager.query_knowledge_base(
                query=search_query,
                n_results=self.max_context_results * 2,  # Get more for filtering
                similarity_threshold=self.similarity_threshold
            )

            return results

        except Exception as e:
            logger.error("Knowledge base search failed", error=str(e))
            return []

    def _filter_and_rank_results(
        self,
        results: List[Dict[str, Any]],
        query_features: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Filter and rank search results"""
        if not results:
            return []

        filtered_results = []

        for result in results:
            # Apply similarity threshold
            similarity = result.get('similarity', 0)
            if similarity < self.similarity_threshold:
                continue

            # Apply content filters
            if not self._passes_content_filters(result, query_features):
                continue

            # Add ranking score
            ranking_score = self._calculate_ranking_score(result, query_features)
            result['ranking_score'] = ranking_score

            filtered_results.append(result)

        # Sort by ranking score and similarity
        filtered_results.sort(
            key=lambda x: (x.get('ranking_score', 0), x.get('similarity', 0)),
            reverse=True
        )

        # Return top results
        return filtered_results[:self.max_context_results]

    def _passes_content_filters(
        self,
        result: Dict[str, Any],
        query_features: Dict[str, Any]
    ) -> bool:
        """Check if result passes content filters"""
        metadata = result.get('metadata', {})

        # Check if content is too old (if temporal query)
        temporal_indicators = query_features.get('temporal_indicators', [])
        if temporal_indicators:
            # For temporal queries, prefer recent content
            published_at = metadata.get('published_at')
            if published_at:
                try:
                    # Simple recency check - prefer content from last 30 days
                    published_date = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                    days_old = (datetime.now(timezone.utc) - published_date).days

                    if days_old > 30:
                        return False
                except:
                    pass

        # Check content relevance
        content = metadata.get('content', '')
        query_keywords = query_features.get('keywords', [])

        # Simple relevance check - look for keyword matches
        content_lower = content.lower()
        keyword_matches = sum(1 for keyword in query_keywords if keyword in content_lower)

        # Require at least some keyword matches for relevance
        return keyword_matches > 0

    def _calculate_ranking_score(
        self,
        result: Dict[str, Any],
        query_features: Dict[str, Any]
    ) -> float:
        """Calculate ranking score for a result"""
        score = 0.0

        # Base similarity score
        similarity = result.get('similarity', 0)
        score += similarity * 0.4

        # Content freshness bonus
        metadata = result.get('metadata', {})
        published_at = metadata.get('published_at')
        if published_at:
            try:
                published_date = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                days_old = (datetime.now(timezone.utc) - published_date).days

                # Boost score for recent content
                if days_old < 7:
                    score += 0.3
                elif days_old < 30:
                    score += 0.1
            except:
                pass

        # Source quality bonus (could be configured)
        source = metadata.get('source', '')
        if source in ['TechCrunch', 'ArXiv', 'Wikipedia']:
            score += 0.1

        # Content length bonus (prefer substantial content)
        word_count = metadata.get('word_count', 0)
        if word_count > 200:
            score += 0.1

        return min(score, 1.0)  # Cap at 1.0

    def _generate_enhanced_response(
        self,
        original_query: str,
        filtered_results: List[Dict[str, Any]],
        query_features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate enhanced response data"""
        # Calculate overall confidence
        confidence = self._calculate_overall_confidence(filtered_results, query_features)

        # Extract context information
        context = []
        for result in filtered_results:
            context_item = {
                'content': result.get('metadata', {}).get('content', ''),
                'title': result.get('metadata', {}).get('title', ''),
                'source': result.get('metadata', {}).get('source', ''),
                'similarity': result.get('similarity', 0),
                'url': result.get('metadata', {}).get('url', '')
            }

            if self.include_metadata:
                context_item['metadata'] = result.get('metadata', {})

            context.append(context_item)

        # Generate enhanced query (could include context)
        enhanced_query = original_query
        if confidence > 0.8 and context:
            # For high-confidence queries, we could enhance the query
            # but for now, we'll keep the original
            enhanced_query = original_query

        return {
            'original_query': original_query,
            'enhanced_query': enhanced_query,
            'context': context,
            'confidence': confidence,
            'results_count': len(filtered_results),
            'query_features': query_features,
            'enhancement_metadata': {
                'enhanced_at': datetime.now(timezone.utc).isoformat(),
                'enhancer_version': '1.0.0',
                'similarity_threshold': self.similarity_threshold,
                'max_results': self.max_context_results
            }
        }

    def _calculate_overall_confidence(
        self,
        results: List[Dict[str, Any]],
        query_features: Dict[str, Any]
    ) -> float:
        """Calculate overall confidence in the enhancement"""
        if not results:
            return 0.0

        # Average similarity score
        avg_similarity = sum(r.get('similarity', 0) for r in results) / len(results)

        # Number of results factor
        results_factor = min(len(results) / self.max_context_results, 1.0)

        # Query complexity factor
        complexity = query_features.get('complexity', 'moderate')
        complexity_multiplier = {
            'simple': 0.8,
            'moderate': 1.0,
            'complex': 1.2
        }.get(complexity, 1.0)

        confidence = (avg_similarity * 0.6 + results_factor * 0.4) * complexity_multiplier
        return min(confidence, 1.0)

    async def enhance_query_async(
        self,
        query: str,
        conversation_context: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Async version of enhance_query"""
        loop = asyncio.get_event_loop()

        return await loop.run_in_executor(
            None,
            self.enhance_query,
            query,
            conversation_context,
            **kwargs
        )

    def get_enhancer_stats(self) -> Dict[str, Any]:
        """Get statistics about the query enhancer"""
        return {
            'max_context_results': self.max_context_results,
            'similarity_threshold': self.similarity_threshold,
            'max_response_time_ms': self.max_response_time_ms,
            'include_metadata': self.include_metadata,
            'vector_db_provider': self.vector_manager.provider_name,
            'embedding_provider': self.embedding_generator.provider_name
        }