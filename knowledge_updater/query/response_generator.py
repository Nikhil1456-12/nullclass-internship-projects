"""
Response generator that integrates knowledge base with chatbot responses
"""

import time
from typing import Dict, List, Any, Optional
import re

from ..core.config import get_config
from ..core.logging import get_logger
from .enhancer import QueryEnhancer

logger = get_logger(__name__)


class ResponseGenerator:
    """Generates enhanced responses using knowledge base context"""

    def __init__(self):
        """Initialize response generator"""
        self.config = get_config()
        self.query_enhancer = QueryEnhancer()

        # Get response generation settings
        self.response_config = self.config.get('response_generation', {})
        self.max_context_length = self.response_config.get('max_context_length', 2000)
        self.response_template = self.response_config.get('template', 'default')
        self.include_citations = self.response_config.get('include_citations', True)

        logger.info(
            "Response generator initialized",
            max_context_length=self.max_context_length,
            include_citations=self.include_citations
        )

    def generate_response(
        self,
        query: str,
        conversation_context: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate an enhanced response using knowledge base context

        Args:
            query: User's query
            conversation_context: Previous conversation messages
            **kwargs: Additional parameters

        Returns:
            Generated response with context and metadata
        """
        start_time = time.time()

        try:
            logger.debug("Generating enhanced response", query=query[:50])

            # Enhance the query with knowledge base information
            enhanced_query = self.query_enhancer.enhance_query(
                query,
                conversation_context,
                **kwargs
            )

            # Generate response using enhanced context
            response_data = self._generate_response_from_context(enhanced_query)

            generation_time = time.time() - start_time

            logger.info(
                "Response generated successfully",
                query_length=len(query),
                context_results=enhanced_query.get('results_count', 0),
                generation_time=f"{generation_time:.3f}s"
            )

            return response_data

        except Exception as e:
            generation_time = time.time() - start_time
            logger.error(
                "Response generation failed",
                generation_time=f"{generation_time:.3f}s",
                error=str(e)
            )

            # Return fallback response
            return self._generate_fallback_response(query, str(e))

    def _generate_response_from_context(self, enhanced_query: Dict[str, Any]) -> Dict[str, Any]:
        """Generate response using enhanced query context"""
        original_query = enhanced_query.get('original_query', '')
        context = enhanced_query.get('context', [])
        confidence = enhanced_query.get('confidence', 0.0)

        # Generate response based on confidence and available context
        if confidence > 0.8 and context:
            response_text = self._generate_informed_response(original_query, context)
        elif confidence > 0.5 and context:
            response_text = self._generate_contextual_response(original_query, context)
        else:
            response_text = self._generate_basic_response(original_query, context)

        # Add citations if enabled and context is available
        citations = []
        if self.include_citations and context:
            citations = self._generate_citations(context)

        # Truncate response if too long
        if len(response_text) > 4000:
            response_text = response_text[:4000] + "..."

        return {
            'response': response_text,
            'original_query': original_query,
            'confidence': confidence,
            'context_used': len(context),
            'citations': citations,
            'response_metadata': {
                'generated_at': time.time(),
                'generator_version': '1.0.0',
                'response_length': len(response_text),
                'confidence_threshold': 0.5
            }
        }

    def _generate_informed_response(self, query: str, context: List[Dict[str, Any]]) -> str:
        """Generate highly informed response using context"""
        # For high confidence queries, provide detailed information
        context_texts = [item.get('content', '') for item in context[:3]]  # Top 3 results

        # Combine query with context for comprehensive response
        combined_info = f"Query: {query}\n\nRelevant information:\n"
        for i, ctx_text in enumerate(context_texts, 1):
            title = context[i-1].get('title', 'Information')
            source = context[i-1].get('source', 'Unknown')
            combined_info += f"{i}. {title} (Source: {source})\n{ctx_text[:500]}...\n\n"

        # Generate response based on template
        if self.response_template == 'detailed':
            return self._format_detailed_response(query, combined_info, context)
        else:
            return self._format_default_response(query, combined_info, context)

    def _generate_contextual_response(self, query: str, context: List[Dict[str, Any]]) -> str:
        """Generate response with some context awareness"""
        # For medium confidence, provide relevant context
        top_context = context[0] if context else {}
        title = top_context.get('title', 'Relevant Information')
        content = top_context.get('content', '')[:300]
        source = top_context.get('source', 'Source')

        return (
            f"Based on available information, {content}\n\n"
            f"Source: {title} ({source})\n\n"
            f"This information may help answer your query: {query}"
        )

    def _generate_basic_response(self, query: str, context: List[Dict[str, Any]]) -> str:
        """Generate basic response with minimal context"""
        if context:
            # Use first available context
            top_context = context[0]
            content = top_context.get('content', '')[:200]
            return f"Regarding your query '{query}', here's some relevant information: {content}"
        else:
            return f"I understand you're asking about '{query}'. However, I don't have specific information about this topic in my current knowledge base."

    def _format_detailed_response(
        self,
        query: str,
        combined_info: str,
        context: List[Dict[str, Any]]
    ) -> str:
        """Format detailed response with comprehensive information"""
        return (
            f"Here's a comprehensive answer to your query '{query}' based on "
            "the most relevant and recent information available:\n\n"
            f"{combined_info}\n"
            "This information is drawn from reliable sources and represents "
            "the most current understanding of the topic."
        )

    def _format_default_response(
        self,
        query: str,
        combined_info: str,
        context: List[Dict[str, Any]]
    ) -> str:
        """Format default response style"""
        return (
            f"To answer your question '{query}', here's what I found:\n\n"
            f"{combined_info}\n"
            "This information should help address your inquiry."
        )

    def _generate_citations(self, context: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate citations for the response"""
        citations = []

        for i, ctx_item in enumerate(context[:5], 1):  # Limit to top 5
            citation = {
                'index': i,
                'title': ctx_item.get('title', 'Unknown Title'),
                'source': ctx_item.get('source', 'Unknown Source'),
                'url': ctx_item.get('url', ''),
                'similarity': ctx_item.get('similarity', 0.0)
            }

            citations.append(citation)

        return citations

    def _generate_fallback_response(self, query: str, error: str) -> Dict[str, Any]:
        """Generate fallback response when enhancement fails"""
        return {
            'response': (
                f"I apologize, but I encountered an issue while processing your query '{query}'. "
                "Please try rephrasing your question or ask me something else."
            ),
            'original_query': query,
            'confidence': 0.0,
            'context_used': 0,
            'citations': [],
            'response_metadata': {
                'generated_at': time.time(),
                'generator_version': '1.0.0',
                'response_length': 0,
                'error': error,
                'fallback': True
            }
        }

    async def generate_response_async(
        self,
        query: str,
        conversation_context: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Async version of generate_response"""
        # Use the query enhancer's async method
        enhanced_query = await self.query_enhancer.enhance_query_async(
            query,
            conversation_context,
            **kwargs
        )

        return self._generate_response_from_context(enhanced_query)

    def get_generator_stats(self) -> Dict[str, Any]:
        """Get statistics about the response generator"""
        enhancer_stats = self.query_enhancer.get_enhancer_stats()

        return {
            'response_template': self.response_template,
            'max_context_length': self.max_context_length,
            'include_citations': self.include_citations,
            'enhancer_stats': enhancer_stats
        }

    def update_response_settings(
        self,
        template: Optional[str] = None,
        max_context_length: Optional[int] = None,
        include_citations: Optional[bool] = None
    ) -> None:
        """Update response generation settings"""
        if template is not None:
            self.response_template = template

        if max_context_length is not None:
            self.max_context_length = max_context_length

        if include_citations is not None:
            self.include_citations = include_citations

        logger.info(
            "Response settings updated",
            template=self.response_template,
            max_context_length=self.max_context_length,
            include_citations=self.include_citations
        )