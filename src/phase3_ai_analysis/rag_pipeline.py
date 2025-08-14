"""RAG (Retrieval-Augmented Generation) pipeline for Fed Minutes analysis"""

import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

from ..phase2_knowledge_base import FedMinutesSemanticSearch
from .llm_client import BaseLLMClient, SYSTEM_PROMPTS, create_llm_client


@dataclass
class RAGContext:
    """Context retrieved for RAG"""
    chunks: List[Dict[str, Any]]
    query: str
    total_results: int
    search_params: Dict[str, Any]


@dataclass
class RAGResponse:
    """Response from RAG pipeline"""
    answer: str
    context: RAGContext
    citations: List[Dict[str, str]]
    confidence: float
    tokens_used: Dict[str, int]


class FedMinutesRAG:
    """RAG pipeline for Fed Minutes Q&A"""
    
    def __init__(self, 
                 search_interface: FedMinutesSemanticSearch,
                 llm_client: Optional[BaseLLMClient] = None,
                 config: Optional[Dict] = None):
        self.search = search_interface
        self.llm = llm_client or (create_llm_client(config) if config else None)
        self.logger = logging.getLogger(__name__)
        
        if not self.llm:
            raise ValueError("LLM client required for RAG pipeline")
    
    def answer_question(self,
                       question: str,
                       max_context_chunks: int = 5,
                       date_range: Optional[Tuple[str, str]] = None,
                       min_similarity: float = -1.0,
                       include_citations: bool = True) -> RAGResponse:
        """Answer a question using RAG approach"""
        
        # Step 1: Retrieve relevant context
        context = self._retrieve_context(
            question, 
            max_context_chunks, 
            date_range, 
            min_similarity
        )
        
        if not context.chunks:
            return self._create_no_context_response(question, context)
        
        # Step 2: Build prompt with context
        prompt = self._build_rag_prompt(question, context, include_citations)
        
        # Step 3: Generate answer using LLM
        llm_response = self.llm.generate(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPTS["research"]
        )
        
        # Step 4: Extract citations if included
        citations = self._extract_citations(llm_response.content, context) if include_citations else []
        
        # Step 5: Calculate confidence based on context quality
        confidence = self._calculate_confidence(context)
        
        return RAGResponse(
            answer=llm_response.content,
            context=context,
            citations=citations,
            confidence=confidence,
            tokens_used=llm_response.usage
        )
    
    def summarize_period(self,
                        start_date: str,
                        end_date: str,
                        topics: Optional[List[str]] = None,
                        max_chunks: int = 10) -> RAGResponse:
        """Summarize Fed discussions during a specific period"""
        
        # Build query from topics or use general terms
        if topics:
            query = " ".join(topics)
        else:
            query = "monetary policy economic conditions decisions"
        
        # Retrieve context for the period
        context = self._retrieve_context(
            query=query,
            max_chunks=max_chunks,
            date_range=(start_date, end_date),
            min_similarity=-2.0  # Lower threshold for period summaries
        )
        
        if not context.chunks:
            return self._create_no_context_response(f"Fed discussions from {start_date} to {end_date}", context)
        
        # Build summary prompt
        prompt = f"""Based on the Federal Reserve meeting excerpts below, provide a comprehensive summary 
of the Fed's discussions and decisions during the period from {start_date} to {end_date}.

Focus on:
1. Major policy decisions and changes
2. Key economic concerns discussed
3. International monetary issues
4. Significant votes or disagreements
5. Evolution of Fed thinking over this period

Meeting Excerpts:
{self._format_context_for_prompt(context)}

Provide a well-structured summary that captures the most important aspects of Fed deliberations during this period."""
        
        # Generate summary
        llm_response = self.llm.generate(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPTS["summary"]
        )
        
        citations = self._extract_meeting_references(context)
        confidence = self._calculate_confidence(context)
        
        return RAGResponse(
            answer=llm_response.content,
            context=context,
            citations=citations,
            confidence=confidence,
            tokens_used=llm_response.usage
        )
    
    def analyze_topic_evolution(self,
                               topic: str,
                               start_year: int,
                               end_year: int,
                               chunks_per_year: int = 3) -> RAGResponse:
        """Analyze how Fed's discussion of a topic evolved over time"""
        
        # Retrieve context for each year
        yearly_contexts = []
        all_chunks = []
        
        for year in range(start_year, end_year + 1):
            year_context = self._retrieve_context(
                query=topic,
                max_chunks=chunks_per_year,
                date_range=(f"{year}-01-01", f"{year}-12-31"),
                min_similarity=-1.0
            )
            yearly_contexts.append((year, year_context))
            all_chunks.extend(year_context.chunks)
        
        # Create combined context
        combined_context = RAGContext(
            chunks=all_chunks,
            query=topic,
            total_results=sum(ctx.total_results for _, ctx in yearly_contexts),
            search_params={"years": f"{start_year}-{end_year}", "topic": topic}
        )
        
        # Build evolution analysis prompt
        prompt = f"""Analyze how the Federal Reserve's discussion and treatment of "{topic}" 
evolved from {start_year} to {end_year} based on the meeting excerpts below.

Focus on:
1. How the Fed's understanding or concern about this topic changed
2. Key turning points or shifts in policy approach
3. External events that influenced Fed thinking
4. Changes in the frequency or urgency of discussions

Meeting excerpts by year:
"""
        
        for year, year_context in yearly_contexts:
            if year_context.chunks:
                prompt += f"\n\n=== {year} ===\n"
                prompt += self._format_context_for_prompt(year_context, max_chunks=chunks_per_year)
        
        prompt += "\n\nProvide an analytical narrative of how this topic evolved in Fed discussions over the specified period."
        
        # Generate analysis
        llm_response = self.llm.generate(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPTS["timeline"]
        )
        
        citations = self._extract_meeting_references(combined_context)
        confidence = self._calculate_confidence(combined_context)
        
        return RAGResponse(
            answer=llm_response.content,
            context=combined_context,
            citations=citations,
            confidence=confidence,
            tokens_used=llm_response.usage
        )
    
    def compare_periods(self,
                       period1: Tuple[str, str],
                       period2: Tuple[str, str],
                       aspects: List[str]) -> RAGResponse:
        """Compare Fed discussions between two periods"""
        
        # Retrieve context for both periods
        query = " ".join(aspects)
        
        context1 = self._retrieve_context(
            query=query,
            max_chunks=5,
            date_range=period1,
            min_similarity=-1.0
        )
        
        context2 = self._retrieve_context(
            query=query,
            max_chunks=5,
            date_range=period2,
            min_similarity=-1.0
        )
        
        # Combine contexts
        all_chunks = context1.chunks + context2.chunks
        combined_context = RAGContext(
            chunks=all_chunks,
            query=query,
            total_results=context1.total_results + context2.total_results,
            search_params={
                "period1": period1,
                "period2": period2,
                "aspects": aspects
            }
        )
        
        # Build comparison prompt
        prompt = f"""Compare and contrast Federal Reserve discussions between two periods:
Period 1: {period1[0]} to {period1[1]}
Period 2: {period2[0]} to {period2[1]}

Focus on these aspects: {', '.join(aspects)}

Period 1 Excerpts:
{self._format_context_for_prompt(context1)}

Period 2 Excerpts:
{self._format_context_for_prompt(context2)}

Provide a detailed comparison highlighting:
1. Key similarities in Fed approach or concerns
2. Notable differences in policy or priorities
3. How external circumstances influenced changes
4. Evolution of Fed thinking between the periods"""
        
        # Generate comparison
        llm_response = self.llm.generate(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPTS["research"]
        )
        
        citations = self._extract_meeting_references(combined_context)
        confidence = min(self._calculate_confidence(context1), self._calculate_confidence(context2))
        
        return RAGResponse(
            answer=llm_response.content,
            context=combined_context,
            citations=citations,
            confidence=confidence,
            tokens_used=llm_response.usage
        )
    
    def _retrieve_context(self,
                         query: str,
                         max_chunks: int,
                         date_range: Optional[Tuple[str, str]],
                         min_similarity: float) -> RAGContext:
        """Retrieve relevant context using semantic search"""
        
        search_results = self.search.search(
            query=query,
            max_results=max_chunks,
            date_range=date_range,
            min_similarity=min_similarity
        )
        
        return RAGContext(
            chunks=search_results['results'],
            query=query,
            total_results=search_results['total_results'],
            search_params=search_results.get('search_params', {})
        )
    
    def _build_rag_prompt(self, question: str, context: RAGContext, include_citations: bool) -> str:
        """Build RAG prompt with context"""
        
        prompt = f"""Answer the following question based on the Federal Reserve meeting excerpts provided below.

Question: {question}

Meeting Excerpts:
{self._format_context_for_prompt(context)}

Instructions:
- Base your answer on the provided excerpts
- Be specific and cite meetings when relevant
- If the excerpts don't contain enough information, say so
"""
        
        if include_citations:
            prompt += "- Include meeting dates and files in your response where appropriate\n"
        
        prompt += "\nAnswer:"
        
        return prompt
    
    def _format_context_for_prompt(self, context: RAGContext, max_chunks: Optional[int] = None) -> str:
        """Format context chunks for inclusion in prompt"""
        
        formatted_chunks = []
        chunks_to_use = context.chunks[:max_chunks] if max_chunks else context.chunks
        
        for i, chunk in enumerate(chunks_to_use, 1):
            chunk_text = f"""
--- Excerpt {i} ---
Meeting: {chunk['filename']} ({chunk['date'][:10] if chunk['date'] else 'Date unknown'})
Topics: {', '.join(chunk['topics'][:3]) if chunk['topics'] else 'General discussion'}

{chunk['chunk_text']}
"""
            formatted_chunks.append(chunk_text)
        
        return "\n".join(formatted_chunks)
    
    def _extract_citations(self, answer: str, context: RAGContext) -> List[Dict[str, str]]:
        """Extract citations from answer and context"""
        
        citations = []
        seen = set()
        
        # Extract unique meetings from context
        for chunk in context.chunks:
            meeting_id = chunk['filename']
            if meeting_id not in seen:
                seen.add(meeting_id)
                citations.append({
                    "meeting": meeting_id,
                    "date": chunk['date'][:10] if chunk['date'] else "Date unknown",
                    "topics": ", ".join(chunk['topics'][:3]) if chunk['topics'] else "General"
                })
        
        return citations
    
    def _extract_meeting_references(self, context: RAGContext) -> List[Dict[str, str]]:
        """Extract all unique meeting references from context"""
        
        meetings = {}
        for chunk in context.chunks:
            meeting_id = chunk['filename']
            if meeting_id not in meetings:
                meetings[meeting_id] = {
                    "meeting": meeting_id,
                    "date": chunk['date'][:10] if chunk['date'] else "Date unknown",
                    "chunks": 1
                }
            else:
                meetings[meeting_id]["chunks"] += 1
        
        # Sort by date
        citations = list(meetings.values())
        citations.sort(key=lambda x: x["date"])
        
        return citations
    
    def _calculate_confidence(self, context: RAGContext) -> float:
        """Calculate confidence score based on context quality"""
        
        if not context.chunks:
            return 0.0
        
        # Factors for confidence:
        # 1. Number of relevant chunks found
        # 2. Average similarity score
        # 3. Diversity of sources (different meetings)
        
        num_chunks = len(context.chunks)
        chunk_score = min(num_chunks / 3.0, 1.0)  # Max out at 3 chunks
        
        # Average similarity (already inverse distance, so higher is better)
        avg_similarity = sum(c.get('similarity_score', 0) for c in context.chunks) / num_chunks
        similarity_score = max(0, min(avg_similarity * 2, 1.0))  # Scale to 0-1
        
        # Source diversity
        unique_meetings = len(set(c['filename'] for c in context.chunks))
        diversity_score = min(unique_meetings / 2.0, 1.0)  # Max out at 2 meetings
        
        # Weighted average
        confidence = (chunk_score * 0.4 + similarity_score * 0.4 + diversity_score * 0.2)
        
        return round(confidence, 2)
    
    def _create_no_context_response(self, query: str, context: RAGContext) -> RAGResponse:
        """Create response when no relevant context is found"""
        
        return RAGResponse(
            answer=f"I couldn't find relevant Federal Reserve meeting discussions about '{query}' in the available documents. "
                   "This might be because the topic wasn't discussed in the meetings from this period, or the search "
                   "didn't match the way it was discussed in the original documents.",
            context=context,
            citations=[],
            confidence=0.0,
            tokens_used={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        )


def create_rag_pipeline(config: Dict) -> FedMinutesRAG:
    """Factory function to create RAG pipeline"""
    from ..phase2_knowledge_base import create_search_interface
    
    # Create search interface
    search = create_search_interface(config)
    
    # Create LLM client
    llm = create_llm_client(config)
    
    # Create RAG pipeline
    return FedMinutesRAG(search, llm, config)


if __name__ == "__main__":
    # Example usage
    from src.utils.config import load_config
    
    config = load_config()
    rag = create_rag_pipeline(config)
    
    # Example question
    response = rag.answer_question(
        "What were the Fed's main concerns about inflation in 1971?",
        date_range=("1971-01-01", "1971-12-31")
    )
    
    print(f"Question: What were the Fed's main concerns about inflation in 1971?")
    print(f"\nAnswer: {response.answer}")
    print(f"\nConfidence: {response.confidence}")
    print(f"Citations: {len(response.citations)} meetings referenced")
    print(f"Tokens used: {response.tokens_used}")