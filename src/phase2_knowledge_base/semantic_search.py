"""Semantic search interface for Fed Minutes knowledge base"""

import json
import re
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, date
from pathlib import Path
import logging

from .chroma_db import FedMinutesVectorDB
from .vector_embeddings import VectorEmbeddings


class FedMinutesSemanticSearch:
    """High-level semantic search interface for Fed Minutes"""
    
    def __init__(self, vector_db: FedMinutesVectorDB):
        self.vector_db = vector_db
        self.logger = logging.getLogger(__name__)
    
    def search(self, 
              query: str,
              max_results: int = 10,
              date_range: Optional[Tuple[str, str]] = None,
              meeting_types: Optional[List[str]] = None,
              min_similarity: float = -1.0) -> Dict[str, Any]:
        """
        Perform semantic search with optional filters
        
        Args:
            query: Search query text
            max_results: Maximum number of results to return
            date_range: Optional tuple of (start_date, end_date) in ISO format
            meeting_types: Optional list of meeting types to filter by
            min_similarity: Minimum similarity score threshold
            
        Returns:
            Formatted search results with metadata
        """
        
        # Perform hybrid search
        results = self.vector_db.hybrid_search(
            query=query,
            date_range=date_range,
            meeting_types=meeting_types,
            n_results=max_results
        )
        
        # Filter by minimum similarity if specified
        if min_similarity > 0:
            filtered_results = []
            for result in results['results']:
                if result.get('similarity_score', 0) >= min_similarity:
                    filtered_results.append(result)
            results['results'] = filtered_results
            results['total_results'] = len(filtered_results)
        
        # Add search metadata
        results['search_params'] = {
            'query': query,
            'max_results': max_results,
            'date_range': date_range,
            'meeting_types': meeting_types,
            'min_similarity': min_similarity,
            'search_timestamp': datetime.now().isoformat()
        }
        
        return results
    
    def search_by_topic(self, 
                       topic: str,
                       max_results: int = 10,
                       date_range: Optional[Tuple[str, str]] = None) -> Dict[str, Any]:
        """Search for documents related to a specific topic"""
        
        # Enhance query with topic-related terms
        topic_queries = {
            'monetary_policy': 'monetary policy interest rates discount rate federal funds rate',
            'banking_regulation': 'banking regulation bank supervision financial institutions',
            'international_finance': 'international finance foreign exchange balance of payments',
            'economic_conditions': 'economic conditions inflation unemployment economic growth',
            'financial_markets': 'financial markets securities trading market operations',
            'credit_policy': 'credit policy lending rates bank credit'
        }
        
        enhanced_query = topic_queries.get(topic.lower(), topic)
        
        return self.search(
            query=enhanced_query,
            max_results=max_results,
            date_range=date_range
        )
    
    def search_by_participants(self, 
                              participants: List[str],
                              max_results: int = 10,
                              date_range: Optional[Tuple[str, str]] = None) -> Dict[str, Any]:
        """Search for meetings with specific participants"""
        
        query = f"meeting with {' '.join(participants)}"
        
        return self.search(
            query=query,
            max_results=max_results,
            date_range=date_range
        )
    
    def search_decisions(self, 
                        decision_type: str = "approved",
                        max_results: int = 10,
                        date_range: Optional[Tuple[str, str]] = None) -> Dict[str, Any]:
        """Search for specific types of decisions"""
        
        decision_queries = {
            'approved': 'approved decision authorization unanimous',
            'denied': 'denied rejected disapproved',
            'deferred': 'deferred postponed delayed',
            'amended': 'amended modified changed',
            'financial': 'expenditure budget financial amount million dollars'
        }
        
        query = decision_queries.get(decision_type.lower(), f"decision {decision_type}")
        
        return self.search(
            query=query,
            max_results=max_results,
            date_range=date_range
        )
    
    def temporal_analysis(self, 
                         query: str,
                         start_year: int,
                         end_year: int,
                         year_step: int = 1) -> Dict[str, Any]:
        """Analyze how a topic evolved over time"""
        
        results_by_year = {}
        
        for year in range(start_year, end_year + 1, year_step):
            date_range = (f"{year}-01-01", f"{year}-12-31")
            
            year_results = self.search(
                query=query,
                max_results=20,
                date_range=date_range
            )
            
            results_by_year[str(year)] = {
                'total_results': year_results['total_results'],
                'avg_similarity': self._calculate_average_similarity(year_results['results']),
                'top_result': year_results['results'][0] if year_results['results'] else None
            }
        
        return {
            'query': query,
            'time_period': f"{start_year}-{end_year}",
            'temporal_data': results_by_year,
            'analysis_summary': self._summarize_temporal_trends(results_by_year)
        }
    
    def find_related_meetings(self, 
                             reference_meeting_id: str,
                             max_results: int = 10) -> Dict[str, Any]:
        """Find meetings related to a reference meeting"""
        
        # Get the reference meeting's content
        reference_chunks = self.vector_db.collection.get(
            where={"meeting_id": reference_meeting_id},
            include=["documents", "metadatas"]
        )
        
        if not reference_chunks['documents']:
            return {'error': f'Reference meeting {reference_meeting_id} not found'}
        
        # Use the first chunk as the query
        reference_text = reference_chunks['documents'][0][:500]  # First 500 chars
        
        # Search for similar content
        results = self.search(
            query=reference_text,
            max_results=max_results + 5  # Get extra to filter out self-matches
        )
        
        # Filter out chunks from the same meeting
        filtered_results = []
        for result in results['results']:
            if result['meeting_id'] != reference_meeting_id:
                filtered_results.append(result)
                if len(filtered_results) >= max_results:
                    break
        
        results['results'] = filtered_results
        results['total_results'] = len(filtered_results)
        results['reference_meeting'] = reference_meeting_id
        
        return results
    
    def extract_key_themes(self, 
                          date_range: Optional[Tuple[str, str]] = None,
                          min_frequency: int = 3) -> Dict[str, Any]:
        """Extract key themes from the corpus using co-occurrence analysis"""
        
        # Get documents in date range
        if date_range:
            documents = self.vector_db.filter_by_date_range(
                start_date=date_range[0],
                end_date=date_range[1],
                n_results=1000
            )
        else:
            # Sample documents
            sample_results = self.vector_db.collection.get(
                limit=1000,
                include=["documents", "metadatas"]
            )
            documents = self.vector_db._format_get_results(sample_results)
        
        # Extract common terms and themes
        theme_analysis = self._analyze_document_themes(documents, min_frequency)
        
        return {
            'date_range': date_range,
            'documents_analyzed': len(documents),
            'themes': theme_analysis,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _calculate_average_similarity(self, results: List[Dict]) -> float:
        """Calculate average similarity score from results"""
        if not results:
            return 0.0
        
        scores = [r.get('similarity_score', 0) for r in results if r.get('similarity_score') is not None]
        return sum(scores) / len(scores) if scores else 0.0
    
    def _summarize_temporal_trends(self, temporal_data: Dict) -> Dict[str, Any]:
        """Summarize trends in temporal analysis"""
        years = sorted(temporal_data.keys())
        
        if len(years) < 2:
            return {'trend': 'insufficient_data'}
        
        # Calculate trends
        result_counts = [temporal_data[year]['total_results'] for year in years]
        similarities = [temporal_data[year]['avg_similarity'] for year in years]
        
        # Simple trend calculation
        total_trend = 'increasing' if result_counts[-1] > result_counts[0] else 'decreasing'
        similarity_trend = 'increasing' if similarities[-1] > similarities[0] else 'decreasing'
        
        peak_year = years[result_counts.index(max(result_counts))]
        
        return {
            'overall_trend': total_trend,
            'similarity_trend': similarity_trend,
            'peak_year': peak_year,
            'peak_results': max(result_counts),
            'total_span': f"{years[0]}-{years[-1]}"
        }
    
    def _analyze_document_themes(self, documents: List[Dict], min_frequency: int) -> Dict[str, Any]:
        """Analyze themes in a collection of documents"""
        
        # Simple keyword extraction (could be enhanced with NLP)
        theme_keywords = {
            'monetary_policy': ['interest', 'rate', 'monetary', 'policy', 'federal', 'funds'],
            'banking': ['bank', 'banking', 'institution', 'credit', 'loan'],
            'regulation': ['regulation', 'supervision', 'compliance', 'rule'],
            'international': ['international', 'foreign', 'exchange', 'balance'],
            'economic': ['economic', 'economy', 'growth', 'inflation', 'employment'],
            'market': ['market', 'trading', 'securities', 'financial']
        }
        
        theme_counts = {theme: 0 for theme in theme_keywords}
        
        for doc in documents:
            text = doc['chunk_text'].lower()
            for theme, keywords in theme_keywords.items():
                if any(keyword in text for keyword in keywords):
                    theme_counts[theme] += 1
        
        # Filter by minimum frequency
        significant_themes = {
            theme: count for theme, count in theme_counts.items()
            if count >= min_frequency
        }
        
        return {
            'theme_frequencies': significant_themes,
            'most_common_theme': max(significant_themes.keys(), key=significant_themes.get) if significant_themes else None,
            'theme_diversity': len(significant_themes)
        }


class QueryBuilder:
    """Helper class for building complex search queries"""
    
    @staticmethod
    def build_date_query(start_date: str, end_date: str) -> Tuple[str, str]:
        """Build a date range query"""
        return (start_date, end_date)
    
    @staticmethod
    def build_participant_query(names: List[str]) -> str:
        """Build a query for specific participants"""
        return " ".join([f'"{name}"' for name in names])
    
    @staticmethod
    def build_topic_query(topics: List[str]) -> str:
        """Build a query for multiple topics"""
        return " OR ".join(topics)
    
    @staticmethod
    def parse_natural_language_date(date_str: str) -> Optional[str]:
        """Parse natural language dates into ISO format"""
        
        # Simple patterns (could be enhanced)
        patterns = {
            r'(\d{4})': lambda m: f"{m.group(1)}-01-01",
            r'(\d{1,2})/(\d{4})': lambda m: f"{m.group(2)}-{m.group(1):0>2}-01",
            r'(\w+)\s+(\d{4})': lambda m: QueryBuilder._month_to_iso(m.group(1), m.group(2))
        }
        
        for pattern, converter in patterns.items():
            match = re.search(pattern, date_str)
            if match:
                try:
                    return converter(match)
                except:
                    continue
        
        return None
    
    @staticmethod
    def _month_to_iso(month_name: str, year: str) -> str:
        """Convert month name to ISO date"""
        months = {
            'january': '01', 'february': '02', 'march': '03', 'april': '04',
            'may': '05', 'june': '06', 'july': '07', 'august': '08',
            'september': '09', 'october': '10', 'november': '11', 'december': '12'
        }
        
        month_num = months.get(month_name.lower(), '01')
        return f"{year}-{month_num}-01"


def create_search_interface(config_or_path=None) -> FedMinutesSemanticSearch:
    """Factory function to create search interface
    
    Args:
        config_or_path: Either a config dict or path to config file
    """
    from src.utils.config import load_config
    from .chroma_db import create_vector_db
    
    if config_or_path is None or isinstance(config_or_path, str):
        # Load from path
        config = load_config(config_or_path)
    else:
        # Already a config dict
        config = config_or_path
    
    vector_db = create_vector_db(config)
    return FedMinutesSemanticSearch(vector_db)


if __name__ == "__main__":
    # Example usage
    from src.utils.config import load_config
    
    config = load_config()
    search = create_search_interface(config)
    
    # Test searches
    print("Testing semantic search...")
    
    # Basic search
    results = search.search("interest rates and monetary policy", max_results=3)
    print(f"Found {results['total_results']} results for basic search")
    
    # Topic search
    results = search.search_by_topic("monetary_policy", max_results=3)
    print(f"Found {results['total_results']} results for topic search")
    
    # Decision search
    results = search.search_decisions("approved", max_results=3)
    print(f"Found {results['total_results']} results for decision search")
    
    # Temporal analysis
    temporal = search.temporal_analysis("inflation", 1968, 1973)
    print(f"Temporal analysis: {temporal['analysis_summary']}")