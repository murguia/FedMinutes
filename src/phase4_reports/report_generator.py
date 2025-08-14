"""Research report generator for Fed Minutes analysis"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import json

from ..phase3_ai_analysis import FedMinutesRAG, RAGResponse
from ..phase2_knowledge_base import FedMinutesSemanticSearch


@dataclass
class ReportSection:
    """A section of the research report"""
    title: str
    content: str
    section_type: str  # "executive_summary", "findings", "timeline", "conclusion"
    subsections: Optional[List['ReportSection']] = None
    citations: Optional[List[Dict[str, str]]] = None
    confidence: float = 1.0


@dataclass
class Timeline:
    """Timeline of events for a specific topic"""
    topic: str
    events: List[Dict[str, Any]]  # date, description, significance, citations
    start_date: str
    end_date: str
    key_turning_points: List[Dict[str, str]]


@dataclass
class ResearchReport:
    """Complete research report"""
    title: str
    subtitle: Optional[str]
    authors: List[str]
    date_generated: str
    executive_summary: str
    sections: List[ReportSection]
    timeline: Optional[Timeline]
    citations: List[Dict[str, str]]
    appendices: Optional[List[ReportSection]] = None
    metadata: Optional[Dict[str, Any]] = None


class FedMinutesReportGenerator:
    """Generate comprehensive research reports from Fed Minutes analysis"""
    
    def __init__(self, rag_pipeline: FedMinutesRAG, search_interface: FedMinutesSemanticSearch):
        self.rag = rag_pipeline
        self.search = search_interface
        self.logger = logging.getLogger(__name__)
    
    def generate_comprehensive_report(self,
                                    topic: str,
                                    time_period: Tuple[str, str],
                                    report_type: str = "standard",
                                    include_timeline: bool = True,
                                    max_sections: int = 10) -> ResearchReport:
        """Generate a complete research report on a topic"""
        
        start_date, end_date = time_period
        self.logger.info(f"Generating {report_type} report on '{topic}' from {start_date} to {end_date}")
        
        # Generate executive summary
        executive_summary = self._generate_executive_summary(topic, time_period)
        
        # Generate main sections based on report type
        sections = []
        
        if report_type == "standard":
            sections.extend([
                self._generate_background_section(topic, time_period),
                self._generate_main_findings_section(topic, time_period),
                self._generate_policy_evolution_section(topic, time_period),
                self._generate_key_decisions_section(topic, time_period),
                self._generate_implications_section(topic, time_period)
            ])
        
        elif report_type == "crisis_analysis":
            sections.extend([
                self._generate_pre_crisis_section(topic, time_period),
                self._generate_crisis_response_section(topic, time_period),
                self._generate_immediate_impact_section(topic, time_period),
                self._generate_long_term_effects_section(topic, time_period)
            ])
        
        elif report_type == "policy_evolution":
            sections.extend([
                self._generate_initial_policy_section(topic, time_period),
                self._generate_evolution_drivers_section(topic, time_period),
                self._generate_turning_points_section(topic, time_period),
                self._generate_final_state_section(topic, time_period)
            ])
        
        # Generate timeline if requested
        timeline = None
        if include_timeline:
            timeline = self._generate_timeline(topic, time_period)
        
        # Collect all citations
        all_citations = self._consolidate_citations(sections)
        
        # Generate conclusion
        conclusion = self._generate_conclusion(topic, sections, timeline)
        sections.append(conclusion)
        
        # Create the report
        report = ResearchReport(
            title=f"Federal Reserve Policy Analysis: {topic.title()}",
            subtitle=f"A Comprehensive Study of Fed Minutes ({start_date} to {end_date})",
            authors=["Fed Minutes AI Analysis System"],
            date_generated=datetime.now().strftime("%Y-%m-%d"),
            executive_summary=executive_summary.content,
            sections=sections,
            timeline=timeline,
            citations=all_citations,
            metadata={
                "topic": topic,
                "time_period": time_period,
                "report_type": report_type,
                "generation_model": self.rag.llm.model,
                "total_meetings_analyzed": len(set(c['meeting'] for c in all_citations))
            }
        )
        
        return report
    
    def _generate_executive_summary(self, topic: str, time_period: Tuple[str, str]) -> ReportSection:
        """Generate executive summary using AI analysis"""
        
        # Get period summary
        summary_response = self.rag.summarize_period(
            start_date=time_period[0],
            end_date=time_period[1],
            topics=[topic],
            max_chunks=10
        )
        
        # Get key insights
        prompt = f"""Based on the Federal Reserve's discussions of {topic} from {time_period[0]} to {time_period[1]}, 
        provide a concise executive summary (300-400 words) that includes:
        1. The main policy challenges faced
        2. Key decisions and their rationale
        3. Major shifts in Fed thinking
        4. Overall impact and significance
        
        Context: {summary_response.answer}"""
        
        exec_summary = self.rag.llm.generate(
            prompt=prompt,
            system_prompt="You are writing an executive summary for a formal research report on Federal Reserve policy. Be concise, authoritative, and highlight the most important findings."
        )
        
        return ReportSection(
            title="Executive Summary",
            content=exec_summary.content,
            section_type="executive_summary",
            citations=[{"meeting": c["meeting"], "date": c["date"]} for c in summary_response.citations[:5]],
            confidence=summary_response.confidence
        )
    
    def _generate_background_section(self, topic: str, time_period: Tuple[str, str]) -> ReportSection:
        """Generate background and context section"""
        
        # Search for early discussions of the topic
        early_date = time_period[0]
        
        background_query = f"What was the Federal Reserve's initial understanding and approach to {topic} in the early period around {early_date}?"
        
        response = self.rag.answer_question(
            question=background_query,
            date_range=time_period,
            max_context_chunks=5
        )
        
        return ReportSection(
            title="Background and Historical Context",
            content=response.answer,
            section_type="findings",
            citations=[{"meeting": c["meeting"], "date": c["date"]} for c in response.citations],
            confidence=response.confidence
        )
    
    def _generate_main_findings_section(self, topic: str, time_period: Tuple[str, str]) -> ReportSection:
        """Generate main findings section with subsections"""
        
        subsections = []
        
        # Key themes analysis
        themes_query = f"What were the main recurring themes in Fed discussions about {topic}?"
        themes_response = self.rag.answer_question(themes_query, date_range=time_period)
        
        subsections.append(ReportSection(
            title="Key Themes",
            content=themes_response.answer,
            section_type="findings",
            confidence=themes_response.confidence
        ))
        
        # Consensus vs dissent
        consensus_query = f"What areas of consensus and disagreement existed regarding {topic}?"
        consensus_response = self.rag.answer_question(consensus_query, date_range=time_period)
        
        subsections.append(ReportSection(
            title="Consensus and Dissent",
            content=consensus_response.answer,
            section_type="findings",
            confidence=consensus_response.confidence
        ))
        
        # Compile main findings
        main_content = f"The analysis of Federal Reserve minutes from {time_period[0]} to {time_period[1]} reveals several key findings regarding {topic}:"
        
        return ReportSection(
            title="Main Findings",
            content=main_content,
            section_type="findings",
            subsections=subsections,
            confidence=min(s.confidence for s in subsections)
        )
    
    def _generate_timeline(self, topic: str, time_period: Tuple[str, str]) -> Timeline:
        """Generate chronological timeline of key events"""
        
        # Get topic evolution
        start_year = int(time_period[0][:4])
        end_year = int(time_period[1][:4])
        
        evolution = self.rag.analyze_topic_evolution(
            topic=topic,
            start_year=start_year,
            end_year=end_year,
            chunks_per_year=3
        )
        
        # Extract key events from the evolution analysis
        events = []
        
        # Parse the evolution response to identify key dates and events
        # This is a simplified version - could be enhanced with more sophisticated parsing
        for year in range(start_year, end_year + 1):
            year_query = f"What were the most significant Fed decisions or discussions about {topic} in {year}?"
            year_response = self.rag.answer_question(
                year_query, 
                date_range=(f"{year}-01-01", f"{year}-12-31"),
                max_context_chunks=3
            )
            
            if year_response.confidence > 0.5:
                events.append({
                    "date": f"{year}",
                    "description": year_response.answer[:200] + "...",
                    "significance": "High" if year_response.confidence > 0.7 else "Medium",
                    "citations": [c["meeting"] for c in year_response.citations[:2]]
                })
        
        # Identify turning points
        turning_points = []
        if "nixon shock" in topic.lower() or "1971" in str(time_period):
            turning_points.append({
                "date": "1971-08-15",
                "event": "Nixon Shock announcement",
                "impact": "Fundamental shift in monetary policy framework"
            })
        
        return Timeline(
            topic=topic,
            events=events,
            start_date=time_period[0],
            end_date=time_period[1],
            key_turning_points=turning_points
        )
    
    def _generate_conclusion(self, topic: str, sections: List[ReportSection], 
                           timeline: Optional[Timeline]) -> ReportSection:
        """Generate conclusion synthesizing all findings"""
        
        # Summarize key points from all sections
        key_points = []
        for section in sections:
            if section.confidence > 0.6:
                key_points.append(f"- {section.title}: {section.content[:100]}...")
        
        prompt = f"""Based on this comprehensive analysis of {topic} in Federal Reserve minutes, 
        write a conclusion (300-400 words) that:
        1. Synthesizes the main findings
        2. Highlights the historical significance
        3. Discusses implications for understanding Fed policy
        4. Suggests areas for future research
        
        Key findings from the report:
        {chr(10).join(key_points[:5])}"""
        
        conclusion_response = self.rag.llm.generate(
            prompt=prompt,
            system_prompt="Write a scholarly conclusion for a research report on Federal Reserve policy history."
        )
        
        return ReportSection(
            title="Conclusion",
            content=conclusion_response.content,
            section_type="conclusion",
            confidence=0.9
        )
    
    def _consolidate_citations(self, sections: List[ReportSection]) -> List[Dict[str, str]]:
        """Consolidate all unique citations from sections"""
        
        all_citations = {}
        
        for section in sections:
            if section is None:
                continue
            if section.citations:
                for citation in section.citations:
                    meeting_id = citation.get("meeting", "")
                    if meeting_id and meeting_id not in all_citations:
                        all_citations[meeting_id] = citation
            
            # Check subsections
            if section.subsections:
                for subsection in section.subsections:
                    if subsection.citations:
                        for citation in subsection.citations:
                            meeting_id = citation.get("meeting", "")
                            if meeting_id and meeting_id not in all_citations:
                                all_citations[meeting_id] = citation
        
        # Sort by date
        sorted_citations = sorted(all_citations.values(), key=lambda x: x.get("date", ""))
        
        return sorted_citations
    
    # Additional section generators for different report types
    
    def _generate_policy_evolution_section(self, topic: str, time_period: Tuple[str, str]) -> ReportSection:
        """Generate section on policy evolution"""
        evolution = self.rag.analyze_topic_evolution(
            topic=topic,
            start_year=int(time_period[0][:4]),
            end_year=int(time_period[1][:4])
        )
        
        return ReportSection(
            title="Policy Evolution and Development",
            content=evolution.answer,
            section_type="findings",
            confidence=evolution.confidence
        )
    
    def _generate_key_decisions_section(self, topic: str, time_period: Tuple[str, str]) -> ReportSection:
        """Generate section on key decisions"""
        decisions_query = f"What were the most important Federal Reserve decisions regarding {topic} and what was their rationale?"
        response = self.rag.answer_question(decisions_query, date_range=time_period, max_context_chunks=8)
        
        return ReportSection(
            title="Key Decisions and Rationale",
            content=response.answer,
            section_type="findings",
            citations=[{"meeting": c["meeting"], "date": c["date"]} for c in response.citations],
            confidence=response.confidence
        )
    
    def _generate_implications_section(self, topic: str, time_period: Tuple[str, str]) -> ReportSection:
        """Generate section on implications"""
        implications_query = f"What were the broader implications of the Fed's approach to {topic} for monetary policy and the economy?"
        response = self.rag.answer_question(implications_query, date_range=time_period)
        
        return ReportSection(
            title="Policy Implications and Impact",
            content=response.answer,
            section_type="findings",
            confidence=response.confidence
        )
    
    # Crisis analysis sections
    def _generate_pre_crisis_section(self, topic: str, time_period: Tuple[str, str]) -> ReportSection:
        """For crisis analysis: pre-crisis conditions"""
        pre_crisis_query = f"What were the economic conditions and Fed policies leading up to the {topic}? What challenges was the Fed facing?"
        response = self.rag.answer_question(pre_crisis_query, date_range=time_period, max_context_chunks=5)
        
        return ReportSection(
            title="Pre-Crisis Economic Conditions",
            content=response.answer,
            section_type="findings",
            citations=[{"meeting": c["meeting"], "date": c["date"]} for c in response.citations],
            confidence=response.confidence
        )
    
    def _generate_crisis_response_section(self, topic: str, time_period: Tuple[str, str]) -> ReportSection:
        """For crisis analysis: immediate response"""
        crisis_query = f"How did the Federal Reserve immediately respond to the {topic}? What emergency measures or policy changes were implemented?"
        response = self.rag.answer_question(crisis_query, date_range=time_period, max_context_chunks=6)
        
        return ReportSection(
            title="Federal Reserve Crisis Response",
            content=response.answer,
            section_type="findings",
            citations=[{"meeting": c["meeting"], "date": c["date"]} for c in response.citations],
            confidence=response.confidence
        )
    
    def _generate_immediate_impact_section(self, topic: str, time_period: Tuple[str, str]) -> ReportSection:
        """For crisis analysis: immediate impacts"""
        impact_query = f"What were the immediate economic and market impacts of the {topic}? How did the Fed assess these effects?"
        response = self.rag.answer_question(impact_query, date_range=time_period, max_context_chunks=5)
        
        return ReportSection(
            title="Immediate Economic Impact",
            content=response.answer,
            section_type="findings",
            citations=[{"meeting": c["meeting"], "date": c["date"]} for c in response.citations],
            confidence=response.confidence
        )
    
    def _generate_long_term_effects_section(self, topic: str, time_period: Tuple[str, str]) -> ReportSection:
        """For crisis analysis: long-term effects"""
        effects_query = f"What were the longer-term consequences and policy changes that resulted from the {topic}? How did it reshape Fed thinking?"
        response = self.rag.answer_question(effects_query, date_range=time_period, max_context_chunks=5)
        
        return ReportSection(
            title="Long-term Policy Effects",
            content=response.answer,
            section_type="findings",
            citations=[{"meeting": c["meeting"], "date": c["date"]} for c in response.citations],
            confidence=response.confidence
        )
    
    # Policy evolution sections
    def _generate_initial_policy_section(self, topic: str, time_period: Tuple[str, str]) -> ReportSection:
        """For policy evolution: initial state"""
        initial_query = f"What was the Federal Reserve's initial policy stance regarding {topic} at the beginning of this period?"
        response = self.rag.answer_question(initial_query, date_range=time_period, max_context_chunks=4)
        
        return ReportSection(
            title="Initial Policy Framework",
            content=response.answer,
            section_type="findings",
            citations=[{"meeting": c["meeting"], "date": c["date"]} for c in response.citations],
            confidence=response.confidence
        )
    
    def _generate_evolution_drivers_section(self, topic: str, time_period: Tuple[str, str]) -> ReportSection:
        """For policy evolution: what drove changes"""
        drivers_query = f"What factors and events drove changes in Fed policy regarding {topic} over this period?"
        response = self.rag.answer_question(drivers_query, date_range=time_period, max_context_chunks=6)
        
        return ReportSection(
            title="Drivers of Policy Evolution",
            content=response.answer,
            section_type="findings",
            citations=[{"meeting": c["meeting"], "date": c["date"]} for c in response.citations],
            confidence=response.confidence
        )
    
    def _generate_turning_points_section(self, topic: str, time_period: Tuple[str, str]) -> ReportSection:
        """For policy evolution: key turning points"""
        turning_query = f"What were the key turning points or pivotal moments in Fed policy regarding {topic}?"
        response = self.rag.answer_question(turning_query, date_range=time_period, max_context_chunks=5)
        
        return ReportSection(
            title="Policy Turning Points",
            content=response.answer,
            section_type="findings",
            citations=[{"meeting": c["meeting"], "date": c["date"]} for c in response.citations],
            confidence=response.confidence
        )
    
    def _generate_final_state_section(self, topic: str, time_period: Tuple[str, str]) -> ReportSection:
        """For policy evolution: end state"""
        final_query = f"What was the final state of Fed policy regarding {topic} by the end of this period? How had it evolved?"
        response = self.rag.answer_question(final_query, date_range=time_period, max_context_chunks=4)
        
        return ReportSection(
            title="Evolved Policy Framework",
            content=response.answer,
            section_type="findings",
            citations=[{"meeting": c["meeting"], "date": c["date"]} for c in response.citations],
            confidence=response.confidence
        )


def create_report_generator(config: Dict) -> FedMinutesReportGenerator:
    """Factory function to create report generator"""
    from ..phase3_ai_analysis import create_rag_pipeline
    from ..phase2_knowledge_base import create_search_interface
    
    rag = create_rag_pipeline(config)
    search = create_search_interface(config)
    
    return FedMinutesReportGenerator(rag, search)