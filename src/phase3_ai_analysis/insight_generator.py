"""Automated insight generation for Fed Minutes analysis"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

from .rag_pipeline import FedMinutesRAG, RAGResponse
from ..phase2_knowledge_base import FedMinutesSemanticSearch


@dataclass
class Insight:
    """Generated insight from Fed Minutes"""
    title: str
    content: str
    insight_type: str  # "trend", "anomaly", "pattern", "correlation"
    confidence: float
    time_period: Tuple[str, str]
    supporting_meetings: List[str]
    keywords: List[str]


@dataclass
class InsightReport:
    """Collection of insights for a research topic"""
    topic: str
    insights: List[Insight]
    summary: str
    time_range: Tuple[str, str]
    total_meetings_analyzed: int
    generation_timestamp: str


class FedMinutesInsightGenerator:
    """Automated insight discovery for Fed Minutes"""
    
    def __init__(self, rag_pipeline: FedMinutesRAG, search_interface: FedMinutesSemanticSearch):
        self.rag = rag_pipeline
        self.search = search_interface
        self.logger = logging.getLogger(__name__)
    
    def generate_topic_insights(self,
                               topic: str,
                               start_date: str,
                               end_date: str,
                               max_insights: int = 10) -> InsightReport:
        """Generate comprehensive insights about a topic over time"""
        
        insights = []
        
        # 1. Trend Analysis
        trend_insight = self._analyze_trend(topic, start_date, end_date)
        if trend_insight:
            insights.append(trend_insight)
        
        # 2. Key Decision Points
        decision_insights = self._find_key_decisions(topic, start_date, end_date)
        insights.extend(decision_insights[:3])  # Top 3 decisions
        
        # 3. Anomaly Detection
        anomaly_insights = self._detect_anomalies(topic, start_date, end_date)
        insights.extend(anomaly_insights[:2])  # Top 2 anomalies
        
        # 4. Correlation Analysis
        correlation_insights = self._find_correlations(topic, start_date, end_date)
        insights.extend(correlation_insights[:2])  # Top 2 correlations
        
        # 5. Consensus vs Dissent Analysis
        consensus_insight = self._analyze_consensus(topic, start_date, end_date)
        if consensus_insight:
            insights.append(consensus_insight)
        
        # Sort by confidence and limit
        insights.sort(key=lambda x: x.confidence, reverse=True)
        insights = insights[:max_insights]
        
        # Generate summary
        summary = self._generate_insights_summary(topic, insights, start_date, end_date)
        
        # Count total meetings
        total_meetings = self._count_relevant_meetings(topic, start_date, end_date)
        
        return InsightReport(
            topic=topic,
            insights=insights,
            summary=summary,
            time_range=(start_date, end_date),
            total_meetings_analyzed=total_meetings,
            generation_timestamp=datetime.now().isoformat()
        )
    
    def generate_period_insights(self,
                                start_date: str,
                                end_date: str,
                                focus_areas: Optional[List[str]] = None) -> InsightReport:
        """Generate insights for a specific time period across all topics"""
        
        if not focus_areas:
            focus_areas = [
                "monetary policy",
                "inflation",
                "international monetary",
                "economic conditions",
                "interest rates"
            ]
        
        all_insights = []
        
        for area in focus_areas:
            area_insights = self.generate_topic_insights(
                topic=area,
                start_date=start_date,
                end_date=end_date,
                max_insights=3  # Fewer per topic for period analysis
            )
            all_insights.extend(area_insights.insights)
        
        # Deduplicate and sort
        all_insights = self._deduplicate_insights(all_insights)
        all_insights.sort(key=lambda x: x.confidence, reverse=True)
        
        # Generate period summary
        summary = self._generate_period_summary(start_date, end_date, all_insights)
        
        total_meetings = self._count_relevant_meetings("", start_date, end_date)
        
        return InsightReport(
            topic=f"Period Analysis ({start_date} to {end_date})",
            insights=all_insights[:15],  # Top 15 insights
            summary=summary,
            time_range=(start_date, end_date),
            total_meetings_analyzed=total_meetings,
            generation_timestamp=datetime.now().isoformat()
        )
    
    def _analyze_trend(self, topic: str, start_date: str, end_date: str) -> Optional[Insight]:
        """Analyze trends in topic discussion over time"""
        
        try:
            # Use RAG to analyze topic evolution
            evolution = self.rag.analyze_topic_evolution(
                topic=topic,
                start_year=int(start_date[:4]),
                end_year=int(end_date[:4]),
                chunks_per_year=3
            )
            
            if evolution.confidence < 0.3:
                return None
            
            # Extract trend insight from the analysis
            return Insight(
                title=f"Evolution of {topic.title()} Discussions",
                content=evolution.answer,
                insight_type="trend",
                confidence=evolution.confidence,
                time_period=(start_date, end_date),
                supporting_meetings=[c['meeting'] for c in evolution.citations],
                keywords=topic.split()
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing trend for {topic}: {e}")
            return None
    
    def _find_key_decisions(self, topic: str, start_date: str, end_date: str) -> List[Insight]:
        """Find key decisions related to the topic"""
        
        insights = []
        
        try:
            # Search for decision-related content
            decision_query = f"{topic} decision vote policy change"
            
            search_results = self.search.search(
                query=decision_query,
                max_results=10,
                date_range=(start_date, end_date),
                min_similarity=-1.0
            )
            
            if not search_results['results']:
                return insights
            
            # Group by meetings and find significant decisions
            meeting_groups = {}
            for result in search_results['results']:
                meeting = result['filename']
                if meeting not in meeting_groups:
                    meeting_groups[meeting] = []
                meeting_groups[meeting].append(result)
            
            # Analyze each meeting group for decisions
            for meeting, chunks in meeting_groups.items():
                if len(chunks) < 2:  # Need significant discussion
                    continue
                
                # Combine chunks for analysis
                combined_text = " ".join(chunk['chunk_text'] for chunk in chunks[:3])
                
                # Use RAG to identify key decisions
                decision_query = f"What key decisions were made regarding {topic} in this meeting? Focus on policy changes, votes, and significant announcements."
                
                response = self.rag.llm.generate(
                    prompt=f"Meeting content: {combined_text[:2000]}...\n\nQuestion: {decision_query}",
                    system_prompt="You are analyzing Federal Reserve meeting minutes for key policy decisions. Be specific and factual."
                )
                
                if "decision" in response.content.lower() or "vote" in response.content.lower():
                    insights.append(Insight(
                        title=f"Key Decision in {meeting}",
                        content=response.content,
                        insight_type="pattern",
                        confidence=0.7,  # Medium confidence for decision detection
                        time_period=(chunks[0]['date'][:10], chunks[0]['date'][:10]),
                        supporting_meetings=[meeting],
                        keywords=topic.split() + ["decision", "policy"]
                    ))
                
                if len(insights) >= 3:
                    break
            
        except Exception as e:
            self.logger.error(f"Error finding key decisions for {topic}: {e}")
        
        return insights
    
    def _detect_anomalies(self, topic: str, start_date: str, end_date: str) -> List[Insight]:
        """Detect anomalous patterns in topic discussions"""
        
        insights = []
        
        try:
            # Search for topic across the period
            search_results = self.search.search(
                query=topic,
                max_results=20,
                date_range=(start_date, end_date),
                min_similarity=-1.0
            )
            
            if len(search_results['results']) < 5:
                return insights
            
            # Group by month to find unusual patterns
            monthly_counts = {}
            for result in search_results['results']:
                month_key = result['date'][:7]  # YYYY-MM
                if month_key not in monthly_counts:
                    monthly_counts[month_key] = []
                monthly_counts[month_key].append(result)
            
            # Find months with unusually high discussion
            avg_count = sum(len(chunks) for chunks in monthly_counts.values()) / len(monthly_counts)
            
            for month, chunks in monthly_counts.items():
                if len(chunks) > avg_count * 2:  # More than double average
                    # Analyze why this month had unusual activity
                    combined_text = " ".join(chunk['chunk_text'] for chunk in chunks[:5])
                    
                    anomaly_query = f"Why was there increased discussion about {topic} during this period? What events or concerns drove this focus?"
                    
                    response = self.rag.llm.generate(
                        prompt=f"Meeting excerpts from {month}: {combined_text[:1500]}...\n\nQuestion: {anomaly_query}",
                        system_prompt="Analyze Federal Reserve meeting content for unusual patterns or increased focus on specific topics."
                    )
                    
                    insights.append(Insight(
                        title=f"Increased Focus on {topic.title()} in {month}",
                        content=response.content,
                        insight_type="anomaly",
                        confidence=0.6,
                        time_period=(f"{month}-01", f"{month}-28"),
                        supporting_meetings=list(set(c['filename'] for c in chunks)),
                        keywords=topic.split() + ["unusual", "increased"]
                    ))
        
        except Exception as e:
            self.logger.error(f"Error detecting anomalies for {topic}: {e}")
        
        return insights[:2]  # Return top 2 anomalies
    
    def _find_correlations(self, topic: str, start_date: str, end_date: str) -> List[Insight]:
        """Find correlations between the topic and other themes"""
        
        insights = []
        
        # Common themes to check correlations with
        related_themes = [
            "inflation", "unemployment", "international", "gold", "exchange rates",
            "banking", "credit", "economic growth", "fiscal policy", "wage controls"
        ]
        
        try:
            for theme in related_themes:
                if theme.lower() in topic.lower():
                    continue  # Skip if already the main topic
                
                # Search for co-occurrence
                combined_query = f"{topic} {theme}"
                
                search_results = self.search.search(
                    query=combined_query,
                    max_results=5,
                    date_range=(start_date, end_date),
                    min_similarity=0.0  # Lower threshold for correlation
                )
                
                if len(search_results['results']) >= 3:
                    # Analyze the correlation
                    combined_text = " ".join(
                        chunk['chunk_text'] for chunk in search_results['results'][:3]
                    )
                    
                    correlation_query = f"What is the relationship between {topic} and {theme} in Fed discussions? How do they influence each other?"
                    
                    response = self.rag.llm.generate(
                        prompt=f"Meeting excerpts: {combined_text[:1500]}...\n\nQuestion: {correlation_query}",
                        system_prompt="Analyze correlations and relationships between different topics in Federal Reserve discussions."
                    )
                    
                    if len(response.content) > 100:  # Ensure substantial analysis
                        insights.append(Insight(
                            title=f"Relationship: {topic.title()} and {theme.title()}",
                            content=response.content,
                            insight_type="correlation",
                            confidence=0.5,
                            time_period=(start_date, end_date),
                            supporting_meetings=[c['filename'] for c in search_results['results']],
                            keywords=[topic, theme, "correlation"]
                        ))
                
                if len(insights) >= 2:
                    break
        
        except Exception as e:
            self.logger.error(f"Error finding correlations for {topic}: {e}")
        
        return insights
    
    def _analyze_consensus(self, topic: str, start_date: str, end_date: str) -> Optional[Insight]:
        """Analyze consensus vs dissent patterns in topic discussions"""
        
        try:
            # Search for discussions with disagreement indicators
            dissent_query = f"{topic} dissent disagree oppose minority view"
            
            search_results = self.search.search(
                query=dissent_query,
                max_results=8,
                date_range=(start_date, end_date),
                min_similarity=-1.0
            )
            
            if len(search_results['results']) < 2:
                return None
            
            # Analyze consensus patterns
            combined_text = " ".join(
                chunk['chunk_text'] for chunk in search_results['results'][:5]
            )
            
            consensus_query = f"Analyze the level of consensus vs disagreement about {topic} in Fed meetings. Were there recurring dissenting voices or evolving positions?"
            
            response = self.rag.llm.generate(
                prompt=f"Meeting excerpts: {combined_text[:1800]}...\n\nQuestion: {consensus_query}",
                system_prompt="Analyze consensus and dissent patterns in Federal Reserve discussions."
            )
            
            return Insight(
                title=f"Consensus Analysis: {topic.title()}",
                content=response.content,
                insight_type="pattern",
                confidence=0.6,
                time_period=(start_date, end_date),
                supporting_meetings=[c['filename'] for c in search_results['results']],
                keywords=topic.split() + ["consensus", "dissent"]
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing consensus for {topic}: {e}")
            return None
    
    def _generate_insights_summary(self, topic: str, insights: List[Insight], 
                                 start_date: str, end_date: str) -> str:
        """Generate a summary of all insights for a topic"""
        
        if not insights:
            return f"No significant insights found for {topic} during {start_date} to {end_date}."
        
        # Categorize insights
        trends = [i for i in insights if i.insight_type == "trend"]
        patterns = [i for i in insights if i.insight_type == "pattern"]
        anomalies = [i for i in insights if i.insight_type == "anomaly"]
        correlations = [i for i in insights if i.insight_type == "correlation"]
        
        summary_parts = [
            f"Analysis of {topic} from {start_date} to {end_date} reveals {len(insights)} key insights:"
        ]
        
        if trends:
            summary_parts.append(f"• {len(trends)} trend(s) in topic evolution over time")
        if patterns:
            summary_parts.append(f"• {len(patterns)} significant pattern(s) in Fed discussions")
        if anomalies:
            summary_parts.append(f"• {len(anomalies)} anomal(ies) or unusual focus periods")
        if correlations:
            summary_parts.append(f"• {len(correlations)} correlation(s) with other policy areas")
        
        # Add highest confidence insight summary
        if insights:
            top_insight = max(insights, key=lambda x: x.confidence)
            summary_parts.append(
                f"\\nMost significant finding: {top_insight.title} (confidence: {top_insight.confidence:.2f})"
            )
        
        return "\\n".join(summary_parts)
    
    def _generate_period_summary(self, start_date: str, end_date: str, 
                               insights: List[Insight]) -> str:
        """Generate summary for period analysis"""
        
        summary_parts = [
            f"Period analysis from {start_date} to {end_date} identified {len(insights)} key insights:",
            ""
        ]
        
        # Group insights by type
        insight_types = {}
        for insight in insights:
            if insight.insight_type not in insight_types:
                insight_types[insight.insight_type] = []
            insight_types[insight.insight_type].append(insight)
        
        for insight_type, type_insights in insight_types.items():
            summary_parts.append(f"• {len(type_insights)} {insight_type}(s)")
        
        # Identify key themes
        all_keywords = []
        for insight in insights:
            all_keywords.extend(insight.keywords)
        
        # Count keyword frequency
        keyword_counts = {}
        for keyword in all_keywords:
            keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
        
        # Top themes
        top_themes = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        if top_themes:
            summary_parts.extend([
                "",
                "Most discussed themes:",
                ", ".join([f"{theme} ({count})" for theme, count in top_themes])
            ])
        
        return "\\n".join(summary_parts)
    
    def _deduplicate_insights(self, insights: List[Insight]) -> List[Insight]:
        """Remove duplicate or very similar insights"""
        
        unique_insights = []
        seen_titles = set()
        
        for insight in insights:
            # Simple deduplication by title similarity
            title_key = insight.title.lower().replace(" ", "")
            similar_found = False
            
            for seen_title in seen_titles:
                if abs(len(title_key) - len(seen_title)) < 10:  # Similar length
                    # Check for similar words
                    words1 = set(title_key.split())
                    words2 = set(seen_title.split())
                    overlap = len(words1 & words2) / max(len(words1), len(words2))
                    
                    if overlap > 0.7:  # 70% word overlap
                        similar_found = True
                        break
            
            if not similar_found:
                unique_insights.append(insight)
                seen_titles.add(title_key)
        
        return unique_insights
    
    def _count_relevant_meetings(self, topic: str, start_date: str, end_date: str) -> int:
        """Count meetings that discuss the topic in the date range"""
        
        try:
            search_results = self.search.search(
                query=topic or "federal reserve policy",
                max_results=100,
                date_range=(start_date, end_date),
                min_similarity=-2.0  # Very permissive for counting
            )
            
            unique_meetings = set(result['filename'] for result in search_results['results'])
            return len(unique_meetings)
            
        except Exception as e:
            self.logger.error(f"Error counting meetings: {e}")
            return 0


def create_insight_generator(config: Dict) -> FedMinutesInsightGenerator:
    """Factory function to create insight generator"""
    from . import create_rag_pipeline
    from ..phase2_knowledge_base import create_search_interface
    
    rag = create_rag_pipeline(config)
    search = create_search_interface(config)
    
    return FedMinutesInsightGenerator(rag, search)


if __name__ == "__main__":
    # Example usage
    from src.utils.config import load_config
    
    config = load_config()
    generator = create_insight_generator(config)
    
    # Generate insights about inflation during Nixon Shock period
    insights = generator.generate_topic_insights(
        topic="inflation price stability",
        start_date="1971-01-01",
        end_date="1972-12-31"
    )
    
    print(f"Insights Report: {insights.topic}")
    print(f"Time Range: {insights.time_range[0]} to {insights.time_range[1]}")
    print(f"Insights Found: {len(insights.insights)}")
    print(f"Meetings Analyzed: {insights.total_meetings_analyzed}")
    print(f"\nSummary:\n{insights.summary}")
    
    print(f"\nTop Insights:")
    for i, insight in enumerate(insights.insights[:3], 1):
        print(f"\n{i}. {insight.title}")
        print(f"   Type: {insight.insight_type}, Confidence: {insight.confidence:.2f}")
        print(f"   {insight.content[:200]}...")