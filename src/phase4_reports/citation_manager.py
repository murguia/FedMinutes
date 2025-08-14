"""Citation management for academic-style references"""

from typing import List, Dict, Optional
from enum import Enum
from datetime import datetime


class CitationStyle(Enum):
    """Supported citation styles"""
    APA = "apa"
    MLA = "mla"
    CHICAGO = "chicago"
    HARVARD = "harvard"
    CUSTOM = "custom"


class CitationManager:
    """Manage citations and bibliography formatting"""
    
    def __init__(self, style: CitationStyle = CitationStyle.APA):
        self.style = style
    
    def format_citation(self, meeting: str, date: str, 
                       page: Optional[int] = None,
                       quote: Optional[str] = None) -> str:
        """Format a single citation based on style"""
        
        if self.style == CitationStyle.APA:
            return self._format_apa(meeting, date, page, quote)
        elif self.style == CitationStyle.MLA:
            return self._format_mla(meeting, date, page, quote)
        elif self.style == CitationStyle.CHICAGO:
            return self._format_chicago(meeting, date, page, quote)
        elif self.style == CitationStyle.HARVARD:
            return self._format_harvard(meeting, date, page, quote)
        else:
            return self._format_custom(meeting, date, page, quote)
    
    def create_bibliography(self, citations: List[Dict[str, str]], 
                          style: Optional[CitationStyle] = None) -> List[str]:
        """Create formatted bibliography from citations"""
        
        if style:
            self.style = style
        
        formatted_citations = []
        
        # Sort citations by date
        sorted_citations = sorted(citations, key=lambda x: x.get('date', ''))
        
        for citation in sorted_citations:
            formatted = self.format_citation(
                meeting=citation.get('meeting', 'Unknown meeting'),
                date=citation.get('date', 'n.d.'),
                page=citation.get('page'),
                quote=citation.get('quote')
            )
            
            # Avoid duplicates
            if formatted not in formatted_citations:
                formatted_citations.append(formatted)
        
        return formatted_citations
    
    def _format_apa(self, meeting: str, date: str, 
                   page: Optional[int] = None, quote: Optional[str] = None) -> str:
        """Format citation in APA style"""
        
        # Extract year from date
        try:
            year = datetime.fromisoformat(date).year
        except:
            year = "n.d."
        
        # Basic format: Federal Reserve. (Year). Meeting title. Federal Reserve Archives.
        citation = f"Federal Reserve. ({year}). {meeting}. Federal Reserve Archives."
        
        if page:
            citation = citation[:-1] + f", p. {page}."
        
        return citation
    
    def _format_mla(self, meeting: str, date: str,
                   page: Optional[int] = None, quote: Optional[str] = None) -> str:
        """Format citation in MLA style"""
        
        # Format date in MLA style
        try:
            dt = datetime.fromisoformat(date)
            formatted_date = dt.strftime("%d %b. %Y")
        except:
            formatted_date = "n.d."
        
        # Basic format: "Meeting Title." Federal Reserve Archives, Date.
        citation = f'"{meeting}." Federal Reserve Archives, {formatted_date}.'
        
        if page:
            citation = citation[:-1] + f", p. {page}."
        
        return citation
    
    def _format_chicago(self, meeting: str, date: str,
                       page: Optional[int] = None, quote: Optional[str] = None) -> str:
        """Format citation in Chicago style"""
        
        # Format date
        try:
            dt = datetime.fromisoformat(date)
            formatted_date = dt.strftime("%B %d, %Y")
        except:
            formatted_date = "n.d."
        
        # Chicago format
        citation = f"Federal Reserve. {meeting}. {formatted_date}. Federal Reserve Archives."
        
        if page:
            citation = citation[:-1] + f", {page}."
        
        return citation
    
    def _format_harvard(self, meeting: str, date: str,
                       page: Optional[int] = None, quote: Optional[str] = None) -> str:
        """Format citation in Harvard style"""
        
        # Extract year
        try:
            year = datetime.fromisoformat(date).year
        except:
            year = "n.d."
        
        # Harvard format
        citation = f"Federal Reserve {year}, '{meeting}', Federal Reserve Archives."
        
        if page:
            citation = citation[:-1] + f", p. {page}."
        
        return citation
    
    def _format_custom(self, meeting: str, date: str,
                      page: Optional[int] = None, quote: Optional[str] = None) -> str:
        """Custom citation format for Fed Minutes project"""
        
        # Simple, clear format
        citation = f"{meeting} ({date})"
        
        if page:
            citation += f", page {page}"
        
        if quote:
            citation += f': "{quote[:50]}..."' if len(quote) > 50 else f': "{quote}"'
        
        return citation
    
    def format_inline_citation(self, meeting: str, date: str, style: Optional[CitationStyle] = None) -> str:
        """Format inline citation (parenthetical reference)"""
        
        if style:
            self.style = style
        
        # Extract year
        try:
            year = datetime.fromisoformat(date).year
        except:
            year = "n.d."
        
        if self.style == CitationStyle.APA:
            return f"(Federal Reserve, {year})"
        elif self.style == CitationStyle.MLA:
            return f"(Federal Reserve)"
        elif self.style == CitationStyle.CHICAGO:
            return f"(Federal Reserve {year})"
        elif self.style == CitationStyle.HARVARD:
            return f"(Federal Reserve {year})"
        else:
            return f"({meeting}, {year})"
    
    def format_footnote(self, meeting: str, date: str, page: Optional[int] = None,
                       note_number: int = 1) -> str:
        """Format footnote citation"""
        
        base_citation = self.format_citation(meeting, date, page)
        return f"{note_number}. {base_citation}"
    
    def create_citation_index(self, citations: List[Dict[str, str]]) -> Dict[str, List[str]]:
        """Create an index of citations by topic or year"""
        
        index = {
            "by_year": {},
            "by_meeting_type": {},
            "by_topic": {}
        }
        
        for citation in citations:
            # Index by year
            try:
                year = str(datetime.fromisoformat(citation.get('date', '')).year)
                if year not in index["by_year"]:
                    index["by_year"][year] = []
                index["by_year"][year].append(citation.get('meeting', ''))
            except:
                pass
            
            # Index by meeting type
            meeting = citation.get('meeting', '')
            if 'FOMC' in meeting:
                meeting_type = 'FOMC'
            elif 'Board' in meeting:
                meeting_type = 'Board'
            else:
                meeting_type = 'Other'
            
            if meeting_type not in index["by_meeting_type"]:
                index["by_meeting_type"][meeting_type] = []
            index["by_meeting_type"][meeting_type].append(meeting)
            
            # Index by topic (if provided)
            topics = citation.get('topics', [])
            for topic in topics:
                if topic not in index["by_topic"]:
                    index["by_topic"][topic] = []
                index["by_topic"][topic].append(meeting)
        
        return index


def format_fed_citation(meeting: str, date: str, style: CitationStyle = CitationStyle.APA) -> str:
    """Convenience function to format a Fed Minutes citation"""
    manager = CitationManager(style)
    return manager.format_citation(meeting, date)


def create_bibliography(citations: List[Dict[str, str]], 
                       style: CitationStyle = CitationStyle.APA) -> List[str]:
    """Convenience function to create a bibliography"""
    manager = CitationManager(style)
    return manager.create_bibliography(citations, style)