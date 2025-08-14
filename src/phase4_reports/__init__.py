"""Phase 4: Research Report Generation Module

This module provides automated research report generation capabilities for Fed Minutes:
- Comprehensive research reports with multiple analysis types
- Executive summary generation from AI insights
- Timeline analysis and policy evolution tracking
- Multiple export formats (HTML, PDF, Word)
- Academic citation management
- Statistical analysis and visualizations

"""

from .report_generator import (
    FedMinutesReportGenerator,
    ResearchReport,
    ReportSection,
    Timeline,
    create_report_generator
)

from .export_manager import (
    ExportManager,
    ExportFormat,
    export_to_html,
    export_to_pdf,
    export_to_word
)

from .citation_manager import (
    CitationManager,
    CitationStyle,
    format_fed_citation,
    create_bibliography
)

__all__ = [
    # Report Generator
    'FedMinutesReportGenerator',
    'ResearchReport', 
    'ReportSection',
    'Timeline',
    'create_report_generator',
    
    # Export Manager
    'ExportManager',
    'ExportFormat',
    'export_to_html',
    'export_to_pdf',
    'export_to_word',
    
    # Citation Manager
    'CitationManager',
    'CitationStyle',
    'format_fed_citation',
    'create_bibliography'
]