"""Export manager for generating reports in multiple formats"""

import os
from typing import Dict, Any, Optional, List
from enum import Enum
from datetime import datetime
import json

from .report_generator import ResearchReport, ReportSection, Timeline


class ExportFormat(Enum):
    """Supported export formats"""
    HTML = "html"
    PDF = "pdf"
    MARKDOWN = "markdown"
    WORD = "docx"
    LATEX = "latex"
    JSON = "json"


class ExportManager:
    """Manage export of research reports to various formats"""
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def export_report(self, report: ResearchReport, format: ExportFormat, 
                     filename: Optional[str] = None) -> str:
        """Export report to specified format"""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{report.title.replace(' ', '_')}_{timestamp}"
        
        if format == ExportFormat.HTML:
            return self._export_to_html(report, filename)
        elif format == ExportFormat.MARKDOWN:
            return self._export_to_markdown(report, filename)
        elif format == ExportFormat.JSON:
            return self._export_to_json(report, filename)
        elif format == ExportFormat.PDF:
            return self._export_to_pdf(report, filename)
        elif format == ExportFormat.WORD:
            return self._export_to_word(report, filename)
        elif format == ExportFormat.LATEX:
            return self._export_to_latex(report, filename)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _export_to_html(self, report: ResearchReport, filename: str) -> str:
        """Export report to HTML format"""
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{report.title}</title>
    <style>
        body {{
            font-family: 'Times New Roman', serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
        }}
        h1, h2, h3 {{
            color: #1a1a1a;
        }}
        h1 {{
            text-align: center;
            margin-bottom: 10px;
        }}
        .subtitle {{
            text-align: center;
            font-style: italic;
            margin-bottom: 20px;
        }}
        .metadata {{
            text-align: center;
            color: #666;
            margin-bottom: 40px;
        }}
        .executive-summary {{
            background-color: #f5f5f5;
            padding: 20px;
            margin: 20px 0;
            border-left: 4px solid #333;
        }}
        .section {{
            margin: 30px 0;
        }}
        .subsection {{
            margin-left: 20px;
        }}
        .timeline {{
            background-color: #f9f9f9;
            padding: 20px;
            margin: 20px 0;
        }}
        .timeline-event {{
            margin: 10px 0;
            padding: 10px;
            border-left: 3px solid #666;
        }}
        .citations {{
            font-size: 0.9em;
            margin-top: 40px;
            border-top: 1px solid #ccc;
            padding-top: 20px;
        }}
        .citation {{
            margin: 5px 0;
        }}
        .confidence {{
            color: #666;
            font-size: 0.8em;
        }}
    </style>
</head>
<body>
    <h1>{report.title}</h1>
    <p class="subtitle">{report.subtitle}</p>
    <p class="metadata">
        {', '.join(report.authors)}<br>
        Generated: {report.date_generated}
    </p>
    
    <div class="executive-summary">
        <h2>Executive Summary</h2>
        <p>{report.executive_summary}</p>
    </div>
"""
        
        # Add main sections
        for section in report.sections:
            html_content += self._section_to_html(section)
        
        # Add timeline if present
        if report.timeline:
            html_content += self._timeline_to_html(report.timeline)
        
        # Add citations
        html_content += self._citations_to_html(report.citations)
        
        html_content += """
</body>
</html>"""
        
        filepath = os.path.join(self.output_dir, f"{filename}.html")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return filepath
    
    def _section_to_html(self, section: ReportSection, level: int = 2) -> str:
        """Convert a section to HTML"""
        
        html = f'<div class="section">\n'
        html += f'<h{level}>{section.title}</h{level}>\n'
        
        if section.confidence < 1.0:
            html += f'<span class="confidence">(Confidence: {section.confidence:.2f})</span>\n'
        
        # Convert content paragraphs
        paragraphs = section.content.split('\n\n')
        for para in paragraphs:
            if para.strip():
                html += f'<p>{para.strip()}</p>\n'
        
        # Add subsections
        if section.subsections:
            for subsection in section.subsections:
                html += f'<div class="subsection">\n'
                html += self._section_to_html(subsection, level + 1)
                html += '</div>\n'
        
        html += '</div>\n'
        return html
    
    def _timeline_to_html(self, timeline: Timeline) -> str:
        """Convert timeline to HTML"""
        
        html = '<div class="timeline">\n'
        html += f'<h2>Timeline: {timeline.topic}</h2>\n'
        html += f'<p><em>{timeline.start_date} to {timeline.end_date}</em></p>\n'
        
        for event in timeline.events:
            html += '<div class="timeline-event">\n'
            html += f'<strong>{event["date"]}</strong>: {event["description"]}<br>\n'
            html += f'<em>Significance: {event["significance"]}</em>\n'
            html += '</div>\n'
        
        if timeline.key_turning_points:
            html += '<h3>Key Turning Points</h3>\n'
            for tp in timeline.key_turning_points:
                html += f'<p><strong>{tp["date"]}</strong>: {tp["event"]} - {tp["impact"]}</p>\n'
        
        html += '</div>\n'
        return html
    
    def _citations_to_html(self, citations: List[Dict[str, str]]) -> str:
        """Convert citations to HTML"""
        
        html = '<div class="citations">\n'
        html += '<h2>References</h2>\n'
        
        for i, citation in enumerate(citations, 1):
            html += f'<div class="citation">'
            html += f'{i}. Federal Reserve. ({citation.get("date", "n.d.")}). '
            html += f'<em>{citation.get("meeting", "Meeting transcript")}</em>. '
            html += f'Federal Reserve Archives.'
            html += '</div>\n'
        
        html += '</div>\n'
        return html
    
    def _export_to_markdown(self, report: ResearchReport, filename: str) -> str:
        """Export report to Markdown format"""
        
        md_content = f"""# {report.title}

*{report.subtitle}*

**Authors:** {', '.join(report.authors)}  
**Date:** {report.date_generated}

## Executive Summary

{report.executive_summary}

---

"""
        
        # Add sections
        for section in report.sections:
            md_content += self._section_to_markdown(section)
        
        # Add timeline
        if report.timeline:
            md_content += self._timeline_to_markdown(report.timeline)
        
        # Add citations
        md_content += self._citations_to_markdown(report.citations)
        
        filepath = os.path.join(self.output_dir, f"{filename}.md")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        return filepath
    
    def _section_to_markdown(self, section: ReportSection, level: int = 2) -> str:
        """Convert section to Markdown"""
        
        md = f"\n{'#' * level} {section.title}\n\n"
        
        if section.confidence < 1.0:
            md += f"*Confidence: {section.confidence:.2f}*\n\n"
        
        md += section.content + "\n"
        
        if section.subsections:
            for subsection in section.subsections:
                md += self._section_to_markdown(subsection, level + 1)
        
        return md
    
    def _timeline_to_markdown(self, timeline: Timeline) -> str:
        """Convert timeline to Markdown"""
        
        md = f"\n## Timeline: {timeline.topic}\n\n"
        md += f"*Period: {timeline.start_date} to {timeline.end_date}*\n\n"
        
        for event in timeline.events:
            md += f"- **{event['date']}**: {event['description']} "
            md += f"*(Significance: {event['significance']})*\n"
        
        if timeline.key_turning_points:
            md += "\n### Key Turning Points\n\n"
            for tp in timeline.key_turning_points:
                md += f"- **{tp['date']}**: {tp['event']} - {tp['impact']}\n"
        
        return md + "\n"
    
    def _citations_to_markdown(self, citations: List[Dict[str, str]]) -> str:
        """Convert citations to Markdown"""
        
        md = "\n## References\n\n"
        
        for i, citation in enumerate(citations, 1):
            md += f"{i}. Federal Reserve. ({citation.get('date', 'n.d.')}). "
            md += f"*{citation.get('meeting', 'Meeting transcript')}*. "
            md += "Federal Reserve Archives.\n"
        
        return md
    
    def _export_to_json(self, report: ResearchReport, filename: str) -> str:
        """Export report to JSON format"""
        
        # Convert report to dictionary
        report_dict = {
            "title": report.title,
            "subtitle": report.subtitle,
            "authors": report.authors,
            "date_generated": report.date_generated,
            "executive_summary": report.executive_summary,
            "sections": [self._section_to_dict(s) for s in report.sections],
            "timeline": self._timeline_to_dict(report.timeline) if report.timeline else None,
            "citations": report.citations,
            "metadata": report.metadata
        }
        
        filepath = os.path.join(self.output_dir, f"{filename}.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def _section_to_dict(self, section: ReportSection) -> Dict[str, Any]:
        """Convert section to dictionary"""
        return {
            "title": section.title,
            "content": section.content,
            "section_type": section.section_type,
            "confidence": section.confidence,
            "subsections": [self._section_to_dict(s) for s in section.subsections] if section.subsections else None,
            "citations": section.citations
        }
    
    def _timeline_to_dict(self, timeline: Timeline) -> Dict[str, Any]:
        """Convert timeline to dictionary"""
        return {
            "topic": timeline.topic,
            "events": timeline.events,
            "start_date": timeline.start_date,
            "end_date": timeline.end_date,
            "key_turning_points": timeline.key_turning_points
        }
    
    def _export_to_pdf(self, report: ResearchReport, filename: str) -> str:
        """Export report to PDF format (requires additional dependencies)"""
        # First export to HTML, then convert to PDF
        # This would require wkhtmltopdf or similar tool
        
        # For now, export to HTML with print-friendly CSS
        html_path = self._export_to_html(report, f"{filename}_print")
        
        # TODO: Implement PDF conversion
        # Example with pdfkit:
        # import pdfkit
        # pdf_path = os.path.join(self.output_dir, f"{filename}.pdf")
        # pdfkit.from_file(html_path, pdf_path)
        
        return html_path
    
    def _export_to_word(self, report: ResearchReport, filename: str) -> str:
        """Export report to Word format (requires python-docx)"""
        # TODO: Implement Word export
        # This would require python-docx library
        
        # For now, export to markdown which can be imported to Word
        return self._export_to_markdown(report, filename)
    
    def _export_to_latex(self, report: ResearchReport, filename: str) -> str:
        """Export report to LaTeX format"""
        
        latex_content = r"""\documentclass[12pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[margin=1in]{geometry}
\usepackage{times}
\usepackage{hyperref}

\title{""" + report.title + r"""}
\author{""" + r" \and ".join(report.authors) + r"""}
\date{""" + report.date_generated + r"""}

\begin{document}

\maketitle

\begin{abstract}
""" + report.executive_summary + r"""
\end{abstract}

"""
        
        # Add sections
        for section in report.sections:
            latex_content += self._section_to_latex(section)
        
        # Add bibliography
        latex_content += r"""
\begin{thebibliography}{99}
"""
        
        for i, citation in enumerate(report.citations, 1):
            latex_content += r"\bibitem{fed" + str(i) + "} "
            latex_content += f"Federal Reserve. ({citation.get('date', 'n.d.')}). "
            latex_content += f"\\emph{{{citation.get('meeting', 'Meeting transcript')}}}. "
            latex_content += "Federal Reserve Archives.\n"
        
        latex_content += r"""
\end{thebibliography}

\end{document}"""
        
        filepath = os.path.join(self.output_dir, f"{filename}.tex")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(latex_content)
        
        return filepath
    
    def _section_to_latex(self, section: ReportSection, level: int = 0) -> str:
        """Convert section to LaTeX"""
        
        section_commands = ["\\section", "\\subsection", "\\subsubsection", "\\paragraph"]
        command = section_commands[min(level, 3)]
        
        latex = f"\n{command}{{{section.title}}}\n\n"
        latex += section.content.replace("&", r"\&").replace("%", r"\%").replace("$", r"\$")
        latex += "\n"
        
        if section.subsections:
            for subsection in section.subsections:
                latex += self._section_to_latex(subsection, level + 1)
        
        return latex


# Convenience functions
def export_to_html(report: ResearchReport, filename: Optional[str] = None, 
                   output_dir: str = "reports") -> str:
    """Export report to HTML"""
    manager = ExportManager(output_dir)
    return manager.export_report(report, ExportFormat.HTML, filename)


def export_to_pdf(report: ResearchReport, filename: Optional[str] = None,
                  output_dir: str = "reports") -> str:
    """Export report to PDF"""
    manager = ExportManager(output_dir)
    return manager.export_report(report, ExportFormat.PDF, filename)


def export_to_word(report: ResearchReport, filename: Optional[str] = None,
                   output_dir: str = "reports") -> str:
    """Export report to Word"""
    manager = ExportManager(output_dir)
    return manager.export_report(report, ExportFormat.WORD, filename)