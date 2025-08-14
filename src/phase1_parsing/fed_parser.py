import os
import re
import fitz
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import json
from collections import defaultdict

# ========================================
# Data Classes for Structured Information
# ========================================

@dataclass
class Attendee:
    """Structured representation of meeting attendee."""
    name: str
    title: Optional[str] = None
    organization: Optional[str] = None
    role: Optional[str] = None  # e.g., "Chairman", "Vice Chairman"
    
@dataclass
class Decision:
    """Structured representation of a decision."""
    action: str  # approved, denied, etc.
    subject: str  # what was decided
    vote: Optional[str] = None  # unanimous, split, etc.
    conditions: List[str] = None  # any conditions attached
    references: List[str] = None  # Item numbers, letters, etc.
    financial_amounts: List[str] = None  # monetary amounts mentioned
    institutions: List[str] = None  # banks/companies involved
    full_text: str = ""

@dataclass
class Topic:
    """Topic discussed in meeting."""
    category: str
    title: str
    discussion_length: int  # approximate characters
    key_points: List[str]
    page_references: List[int]

@dataclass
class Meeting:
    """Complete meeting record."""
    filename: str
    date: Optional[datetime]
    meeting_type: str  # regular, special, emergency
    attendees: List[Attendee]
    decisions: List[Decision]
    topics: List[Topic]
    total_pages: int
    raw_text: str

# ========================================
# Enhanced Pattern Definitions
# ========================================

class FedMinutesPatterns:
    """Centralized patterns for Federal Reserve minutes parsing."""
    
    # Date patterns (more comprehensive)
    DATE_PATTERNS = [
        (r'Minutes\s+(?:for|of).*?([A-Za-z]+\s+\d{1,2},\s+\d{4})', 1),
        (r'(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})', 1),
        (r'([A-Za-z]+\s+\d{1,2},\s+\d{4})', 1),
        (r'(\d{1,2}/\d{1,2}/\d{2,4})', 1)
    ]
    
    # Meeting types
    MEETING_TYPES = {
        'regular': r'regular\s+meeting|scheduled\s+meeting',
        'special': r'special\s+meeting|special\s+session',
        'emergency': r'emergency\s+meeting|urgent\s+session',
        'executive': r'executive\s+session',
        'joint': r'joint\s+meeting|joint\s+session'
    }
    
    # Enhanced attendee patterns
    ATTENDEE_PATTERNS = {
        'present_section': r'PRESENT:\s*(.*?)(?=\n\s*\n|\Z|Approved|Discussion|Item\s+No)',
        'board_members': r'(?:Mr\.|Mrs\.|Miss|Ms\.|Dr\.|M\s*r\.|M\s*rs\.)\s+([A-Z][a-z]+(?:\s+[A-Z]\.)?\s+[A-Z][a-z]+)',
        'with_title': r'((?:Mr\.|Mrs\.|Miss|Ms\.|Dr\.|M\s*r\.|M\s*rs\.)\s+[^,]+),\s*([^,\n]+?)(?:,|\n|$)'
    }
    
    # Decision action verbs (expanded)
    DECISION_ACTIONS = [
        'approved', 'denied', 'rejected', 'authorized', 'ratified',
        'adopted', 'extended', 'deferred', 'tabled', 'postponed',
        'recommended', 'endorsed', 'accepted', 'declined', 'modified',
        'amended', 'no objection', 'agreed', 'consented', 'granted'
    ]
    
    # Voting patterns
    VOTING_PATTERNS = {
        'unanimous': r'unanimously|unanimous\s+vote|without\s+objection',
        'split': r'(\d+)\s+to\s+(\d+)|split\s+vote',
        'dissent': r'dissenting|with\s+dissent|voting\s+against',
        'abstention': r'abstaining|abstention'
    }
    
    # Financial patterns
    FINANCIAL_PATTERNS = {
        'amount': r'\$[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion|thousand))?',
        'percentage': r'\d+(?:\.\d+)?%|percent',
        'interest_rate': r'\d+(?:\.\d+)?(?:\s*(?:percent|basis\s+points|bps))'
    }
    
    # Regulatory references
    REGULATORY_PATTERNS = {
        'section': r'Section\s+\d+(?:\([a-zA-Z]\))?(?:\s+of\s+[^,\n]+)?',
        'regulation': r'Regulation\s+[A-Z]+',
        'act': r'(?:Federal\s+Reserve|Banking|Bank\s+Holding\s+Company)\s+Act',
        'form': r'Form\s+[A-Z\.\-]+\s*[\d\-]*'
    }

# ========================================
# Core Parser Class
# ========================================

class FedMinutesParser:
    """Comprehensive parser for Federal Reserve meeting minutes."""
    
    def __init__(self):
        self.patterns = FedMinutesPatterns()
        
    def parse_file(self, filepath: str) -> Meeting:
        """Parse a single Fed minutes file."""
        
        # Extract text
        if filepath.endswith('.txt'):
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
        else:  # PDF
            with fitz.open(filepath) as doc:
                text = "\n".join([page.get_text() for page in doc])
                total_pages = len(doc)
        
        # Clean text
        text = self._clean_text(text)
        
        # Parse components
        date = self._extract_date(text)
        meeting_type = self._determine_meeting_type(text)
        attendees = self._extract_attendees(text)
        decisions = self._extract_decisions(text)
        topics = self._extract_topics(text)
        
        return Meeting(
            filename=os.path.basename(filepath),
            date=date,
            meeting_type=meeting_type,
            attendees=attendees,
            decisions=decisions,
            topics=topics,
            total_pages=total_pages if 'total_pages' in locals() else 0,
            raw_text=text
        )
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text while preserving structure."""
        # Remove FR numbers
        text = re.sub(r'\bFR\s*\d+\b', '', text)
        
        # Fix OCR issues
        text = text.replace('01', "O'")  # O'Connell fix
        
        # Fix spaced honorifics BEFORE other processing
        text = text.replace('M r .', 'Mr.')
        text = text.replace('M r.', 'Mr.')
        text = text.replace('M rs .', 'Mrs.')
        text = text.replace('M rs.', 'Mrs.')
        
        # IMPORTANT: Don't remove ALL whitespace - preserve newlines
        # Replace multiple spaces with single space, but keep newlines
        text = re.sub(r'[ \t]+', ' ', text)  # Only horizontal whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)  # Reduce multiple newlines to double
        
        return text.strip()
    
    def _extract_date(self, text: str) -> Optional[datetime]:
        """Extract and parse meeting date."""
        for pattern, group in self.patterns.DATE_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                date_str = match.group(group)
                try:
                    # Try multiple date formats
                    for fmt in ['%B %d, %Y', '%b %d, %Y', '%m/%d/%Y', '%m/%d/%y']:
                        try:
                            return datetime.strptime(date_str, fmt)
                        except ValueError:
                            continue
                except:
                    pass
        return None
    
    def _determine_meeting_type(self, text: str) -> str:
        """Determine the type of meeting."""
        text_lower = text.lower()
        for meeting_type, pattern in self.patterns.MEETING_TYPES.items():
            if re.search(pattern, text_lower):
                return meeting_type
        return "regular"  # default
    
    def _extract_attendees(self, text: str) -> List[Attendee]:
        """Extract attendees with titles and organizations."""
        attendees = []
        
        # Find PRESENT section - be more careful about what we capture
        present_match = re.search(
            r'PRESENT:\s*(.*?)(?=Approved|Application|Letter|Item\s+No\.|Discussion|Minutes|The meeting|$)',
            text, 
            re.IGNORECASE | re.DOTALL
        )
        
        if present_match:
            attendee_text = present_match.group(1)
            
            # IMPORTANT: Split on common attendee patterns
            # Each attendee typically starts with Mr./Mrs./Dr./Miss or is on a new line
            attendee_pattern = r'((?:Mr\.|Mrs\.|Miss|Ms\.|Dr\.|Messrs\.)\s+[^,\n]+(?:,\s*[^,\n]+)?)'
            
            # Find all individual attendee entries
            matches = re.findall(attendee_pattern, attendee_text)
            
            for match in matches:
                match = match.strip()
                
                # Skip if this contains multiple "Mr." (indicates we grabbed too much)
                if match.count('Mr.') > 1 or match.count('Mrs.') > 1:
                    # Split it further
                    sub_matches = re.findall(r'((?:Mr\.|Mrs\.|Miss|Ms\.|Dr\.)\s+[A-Z][^,]*?)(?=\s+Mr\.|\s+Mrs\.|\s+Dr\.|$)', match)
                    for sub_match in sub_matches:
                        self._parse_single_attendee(sub_match.strip(), attendees)
                else:
                    self._parse_single_attendee(match, attendees)
        
        return attendees

    def _parse_single_attendee(self, text: str, attendees: list):
        """Parse a single attendee entry."""
        if not text:
            return
        
        text = text.strip()
        
        # Handle "and" cases (Mr. X and Mrs. Y)
        if ' and ' in text and ('Mr.' in text or 'Mrs.' in text or 'Messrs.' in text):
            # Handle "Mr. Forrestal and Mrs. Heller, Senior Attorneys"
            if re.match(r'Mr\.\s+\w+\s+and\s+Mrs\.\s+\w+', text):
                parts = text.split(', ', 1)
                names_part = parts[0]
                title_part = parts[1] if len(parts) > 1 else None
                
                # Split the names
                name_matches = re.findall(r'((?:Mr\.|Mrs\.)\s+\w+)', names_part)
                for name in name_matches:
                    attendees.append(Attendee(
                        name=self._clean_name(name),
                        title=title_part,
                        organization=self._extract_organization(title_part),
                        role=None
                    ))
                return
            
            # Handle "Messrs. X and Y"
            elif text.startswith('Messrs.'):
                parts = text.split(', ', 1)
                names_part = parts[0].replace('Messrs.', '').strip()
                title_part = parts[1] if len(parts) > 1 else None
                
                # Split on "and"
                names = names_part.split(' and ')
                for name in names:
                    attendees.append(Attendee(
                        name='Mr. ' + name.strip(),
                        title=title_part,
                        organization=self._extract_organization(title_part),
                        role=None
                    ))
                return
        
        # Parse single person: "Mr. Name, Title"
        parts = text.split(',', 1)
        name = parts[0].strip()
        title = parts[1].strip() if len(parts) > 1 else None
        
        # Clean up the name
        name = self._clean_name(name)
        
        # For titles that continue across lines, try to complete them
        if title and title.endswith(' and'):
            # Title is incomplete, might continue on next line
            title = title.rstrip(' and')
            # Add indicator that this might be incomplete
            title = title + '...'  # or handle differently
        
        # Determine role from title
        role = None
        organization = None
        
        if title:
            if 'Vice Chairman' in title:
                role = 'Vice Chairman'
            elif 'Chairman' in title:
                role = 'Chairman'
            elif title == 'Secretary':
                role = 'Secretary'
            
            organization = self._extract_organization(title)
        
        attendees.append(Attendee(
            name=name,
            title=title,
            organization=organization,
            role=role
        ))

    def _extract_organization(self, title_text: str) -> Optional[str]:
        """Extract organization from title text."""
        if not title_text:
            return None
        
        # Look for Division of X
        div_match = re.search(r'Division of ([^,]+)', title_text)
        if div_match:
            return f"Division of {div_match.group(1).strip()}"
        
        # Look for other organizational units
        if 'Federal Reserve Bank' in title_text:
            return 'Federal Reserve Bank'
        if 'Legal Division' in title_text:
            return 'Legal Division'
        
        return None
        
    def _clean_name(self, name: str) -> str:
        """Clean and standardize names."""
        # Remove extra whitespace
        name = re.sub(r'\s+', ' ', name).strip()
        
        # Standardize honorifics
        name = re.sub(r'\bMr\s*\.\s*', 'Mr. ', name)
        name = re.sub(r'\bMrs\s*\.\s*', 'Mrs. ', name)
        name = re.sub(r'\bDr\s*\.\s*', 'Dr. ', name)
        name = re.sub(r'\bMs\s*\.\s*', 'Ms. ', name)
        
        return name
    
    def _extract_decisions(self, text: str) -> List[Decision]:
        """Extract detailed decision information."""
        decisions = []
        
        # Build pattern for decision sentences
        actions_pattern = '|'.join(self.patterns.DECISION_ACTIONS)
        decision_pattern = rf'\b({actions_pattern})\b(.*?)(?:\.|;|\n\n)'
        
        for match in re.finditer(decision_pattern, text, re.IGNORECASE | re.DOTALL):
            action = match.group(1).lower()
            content = match.group(2)[:500]  # Limit length
            
            # Extract vote information
            vote = None
            for vote_type, vote_pattern in self.patterns.VOTING_PATTERNS.items():
                if re.search(vote_pattern, content, re.IGNORECASE):
                    vote = vote_type
                    break
            
            # Extract financial amounts
            amounts = re.findall(self.patterns.FINANCIAL_PATTERNS['amount'], content)
            
            # Extract institution names
            institutions = self._extract_institutions(content)
            
            # Extract references
            references = self._extract_references(content)
            
            # Extract conditions
            conditions = self._extract_conditions(content)
            
            # Determine subject
            subject = self._determine_subject(content)
            
            decisions.append(Decision(
                action=action,
                subject=subject,
                vote=vote,
                conditions=conditions if conditions else None,
                references=references if references else None,
                financial_amounts=amounts if amounts else None,
                institutions=institutions if institutions else None,
                full_text=match.group(0).strip()
            ))
        
        return decisions
    
    def _extract_institutions(self, text: str) -> List[str]:
        """Extract institution names from text."""
        institutions = []
        
        # Bank patterns
        bank_patterns = [
            r'([A-Z][\w\s]+?(?:Bank|Trust Company|Banking Company|National Bank))',
            r'([A-Z][\w\s]+?(?:Federal Reserve Bank|Reserve Bank))',
            r'([A-Z][\w\s]+?(?:Corporation|Company|Inc\.))'
        ]
        
        for pattern in bank_patterns:
            matches = re.findall(pattern, text)
            institutions.extend([m.strip() for m in matches])
        
        return list(set(institutions))  # Remove duplicates
    
    def _extract_references(self, text: str) -> List[str]:
        """Extract document references."""
        references = []
        
        # Item numbers
        items = re.findall(r'Item\s+No\.?\s*\d+', text, re.IGNORECASE)
        references.extend(items)
        
        # Letters
        letters = re.findall(r'Letter\s+(?:to|from)\s+[^,\n]+', text, re.IGNORECASE)
        references.extend([l[:50] for l in letters])  # Truncate long references
        
        # Memorandums
        memos = re.findall(r'Memorandum\s+(?:from|to)\s+[^,\n]+', text, re.IGNORECASE)
        references.extend([m[:50] for m in memos])
        
        return references
    
    def _extract_conditions(self, text: str) -> List[str]:
        """Extract conditions attached to decisions."""
        conditions = []
        
        condition_patterns = [
            r'provided\s+(?:that\s+)?([^,\n]+)',
            r'subject\s+to\s+([^,\n]+)',
            r'on\s+condition\s+that\s+([^,\n]+)',
            r'contingent\s+upon\s+([^,\n]+)'
        ]
        
        for pattern in condition_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            conditions.extend([m.strip() for m in matches])
        
        return conditions
    
    def _determine_subject(self, text: str) -> str:
        """Determine the subject of a decision."""
        # Look for key phrases
        if re.search(r'branch|establishment', text, re.IGNORECASE):
            return "Branch establishment"
        elif re.search(r'merger|acquisition', text, re.IGNORECASE):
            return "Merger/Acquisition"
        elif re.search(r'interest\s+rate|discount\s+rate', text, re.IGNORECASE):
            return "Interest rate policy"
        elif re.search(r'regulation|rule', text, re.IGNORECASE):
            return "Regulatory matter"
        elif re.search(r'appointment|consultant', text, re.IGNORECASE):
            return "Personnel matter"
        elif re.search(r'report|minutes', text, re.IGNORECASE):
            return "Administrative matter"
        else:
            # Extract first meaningful phrase
            clean_text = re.sub(r'\s+', ' ', text).strip()
            return clean_text[:50] + "..." if len(clean_text) > 50 else clean_text
    
    def _extract_topics(self, text: str) -> List[Topic]:
        """Extract discussion topics from the meeting."""
        topics = []
        
        # Define topic patterns and keywords
        topic_patterns = {
            'Monetary Policy': {
                'keywords': ['interest rate', 'discount rate', 'open market', 'reserve requirement', 
                           'federal funds', 'monetary policy'],
                'sections': []
            },
            'Banking Regulation': {
                'keywords': ['bank holding', 'merger', 'acquisition', 'branch', 'subsidiary',
                           'capital requirement', 'examination'],
                'sections': []
            },
            'International Finance': {
                'keywords': ['foreign', 'exchange', 'international', 'balance of payments', 
                           'gold', 'bretton woods', 'eurodollar'],
                'sections': []
            },
            'Economic Conditions': {
                'keywords': ['inflation', 'unemployment', 'gdp', 'growth', 'recession',
                           'economic outlook', 'forecast'],
                'sections': []
            },
            'Administrative Matters': {
                'keywords': ['appointment', 'consultant', 'personnel', 'minutes', 'report',
                           'correspondence', 'memorandum'],
                'sections': []
            }
        }
        
        # Split text into sections
        sections = self._split_into_sections(text)
        
        for section in sections:
            section_text = section['text'].lower()
            section_topics = []
            
            # Score each topic category
            for category, config in topic_patterns.items():
                score = sum(1 for keyword in config['keywords'] if keyword in section_text)
                if score > 0:
                    section_topics.append((category, score))
            
            # Assign to highest scoring category
            if section_topics:
                best_category = max(section_topics, key=lambda x: x[1])[0]
                
                # Extract key points
                key_points = self._extract_key_points(section['text'])
                
                topics.append(Topic(
                    category=best_category,
                    title=section.get('title', 'Untitled Section'),
                    discussion_length=len(section['text']),
                    key_points=key_points,
                    page_references=section.get('pages', [])
                ))
        
        return topics
    
    def _split_into_sections(self, text: str) -> List[Dict]:
        """Split text into logical sections."""
        sections = []
        
        # Look for section headers
        header_patterns = [
            r'\n([A-Z][^.\n]{10,50})\.\s*\n',  # Title followed by period
            r'\n([A-Z][^.\n]{10,50})\n',  # Title on its own line
            r'(?:Discussion of|Consideration of|Review of)\s+([^.\n]+)',
        ]
        
        current_pos = 0
        for pattern in header_patterns:
            for match in re.finditer(pattern, text):
                if match.start() > current_pos:
                    sections.append({
                        'title': match.group(1).strip(),
                        'text': text[match.start():match.start()+2000],  # Take next 2000 chars
                        'pages': []  # Would need page tracking for this
                    })
        
        # If no sections found, create one big section
        if not sections:
            sections.append({
                'title': 'Main Content',
                'text': text[:5000],  # First 5000 chars
                'pages': []
            })
        
        return sections
    
    def _extract_key_points(self, text: str, max_points: int = 5) -> List[str]:
        """Extract key points from a text section."""
        key_points = []
        
        # Look for enumerated points
        enum_patterns = [
            r'(?:\d+\.|[a-z]\))\s+([^.\n]+)',
            r'(?:First|Second|Third|Fourth|Fifth),\s+([^.\n]+)',
        ]
        
        for pattern in enum_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            key_points.extend([m.strip() for m in matches[:max_points]])
        
        # If no enumerated points, extract sentences with decision verbs
        if not key_points:
            sentences = text.split('.')
            for sentence in sentences[:max_points]:
                if any(action in sentence.lower() for action in self.patterns.DECISION_ACTIONS):
                    key_points.append(sentence.strip())
        
        return key_points[:max_points]

# ========================================
# Batch Processing Functions
# ========================================

class FedMinutesBatchProcessor:
    """Process multiple Fed minutes files."""
    
    def __init__(self):
        self.parser = FedMinutesParser()
    
    def convert_pdfs_to_txt(self, pdf_dir: str, txt_dir: str = "TXTs") -> None:
        """Convert all PDFs to text files for faster processing."""
        os.makedirs(txt_dir, exist_ok=True)
        
        import glob
        pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
        
        for i, pdf_path in enumerate(sorted(pdf_files)):
            filename = os.path.basename(pdf_path)
            txt_path = os.path.join(txt_dir, filename.replace(".pdf", ".txt"))
            
            # Skip if already converted
            if os.path.exists(txt_path):
                print(f"Skipping {filename} (already converted)")
                continue
            
            try:
                # Extract text from PDF
                with fitz.open(pdf_path) as doc:
                    text_pages = [page.get_text("text") for page in doc]
                    text = "\n".join(text_pages)
                
                # Save to TXT file
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(text)
                
                print(f"Converted {i+1}/{len(pdf_files)}: {filename}")
            except Exception as e:
                print(f"Error converting {filename}: {e}")
    
    def process_directory(self, 
                         pdf_dir: str,
                         txt_dir: str = "TXTs",
                         use_txt: bool = True,
                         max_files: Optional[int] = None) -> pd.DataFrame:
        """Process all files in a directory."""
        
        # Convert PDFs to TXT first if requested
        if use_txt:
            print("=== Converting PDFs to TXT files ===")
            self.convert_pdfs_to_txt(pdf_dir, txt_dir)
            print("\n=== Processing TXT files ===")
        
        # Get list of files to process
        import glob
        if use_txt and os.path.exists(txt_dir):
            files = glob.glob(os.path.join(txt_dir, "*.txt"))
            source_dir = txt_dir
            print(f"Found {len(files)} TXT files in {txt_dir}")
        else:
            files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
            source_dir = pdf_dir
            print(f"Found {len(files)} PDF files in {pdf_dir}")
        
        files = sorted(files)[:max_files] if max_files else sorted(files)
        print(f"Processing {len(files)} files (max_files={max_files})")
        
        meetings = []
        errors = []
        
        for i, filepath in enumerate(files):
            try:
                print(f"Processing {i+1}/{len(files)}: {os.path.basename(filepath)}")
                meeting = self.parser.parse_file(filepath)
                meetings.append(meeting)
            except Exception as e:
                print(f"Error processing {filepath}: {e}")
                errors.append({'file': filepath, 'error': str(e)})
        
        # Convert to DataFrame
        df = self._meetings_to_dataframe(meetings)
        
        # Save errors if any
        if errors:
            pd.DataFrame(errors).to_csv('parsing_errors.csv', index=False)
            print(f"Saved {len(errors)} errors to parsing_errors.csv")
        
        return df
    
    def _meetings_to_dataframe(self, meetings: List[Meeting]) -> pd.DataFrame:
        """Convert Meeting objects to DataFrame."""
        
        records = []
        for meeting in meetings:
            record = {
                'filename': meeting.filename,
                'date': meeting.date,
                'meeting_type': meeting.meeting_type,
                'total_pages': meeting.total_pages,
                'num_attendees': len(meeting.attendees),
                'num_decisions': len(meeting.decisions),
                'num_topics': len(meeting.topics),
                # Serialize lists of dicts as valid JSON
                'attendees': json.dumps([asdict(a) for a in meeting.attendees]),
                'decisions': json.dumps([asdict(d) for d in meeting.decisions]),
                'topics': json.dumps([asdict(t) for t in meeting.topics]),
                'text_length': len(meeting.raw_text),
                'raw_text': meeting.raw_text  # Include the actual text for embeddings
            }
            
            # Add summary fields
            record['board_members'] = json.dumps([a.name for a in meeting.attendees if a.role == 'Governor'])
            record['unanimous_decisions'] = sum(1 for d in meeting.decisions if d.vote == 'unanimous')
            record['total_amount_approved'] = self._sum_amounts(meeting.decisions)
            record['main_topics'] = json.dumps(list(set(t.category for t in meeting.topics)))
            
            records.append(record)
        
        return pd.DataFrame(records)
    
    def _sum_amounts(self, decisions: List[Decision]) -> float:
        """Sum all financial amounts in decisions."""
        total = 0
        for decision in decisions:
            if decision.financial_amounts:
                for amount in decision.financial_amounts:
                    # Parse amount string
                    clean = re.sub(r'[,$]', '', amount)
                    try:
                        value = float(clean)
                        if 'million' in amount.lower():
                            value *= 1_000_000
                        elif 'billion' in amount.lower():
                            value *= 1_000_000_000
                        total += value
                    except:
                        pass
        return total

# ========================================
# Analysis and Validation Functions
# ========================================

def analyze_parsing_results(df: pd.DataFrame) -> Dict:
    """Analyze parsed results for quality and completeness."""
    
    analysis = {
        'total_files': len(df),
        'date_coverage': {
            'start': df['date'].min(),
            'end': df['date'].max(),
            'missing_dates': df['date'].isna().sum()
        },
        'attendance': {
            'avg_attendees': df['num_attendees'].mean(),
            'min_attendees': df['num_attendees'].min(),
            'max_attendees': df['num_attendees'].max()
        },
        'decisions': {
            'total': df['num_decisions'].sum(),
            'avg_per_meeting': df['num_decisions'].mean(),
            'unanimous_rate': df['unanimous_decisions'].sum() / df['num_decisions'].sum()
        },
        'topics': {
            'unique_categories': set().union(*df['main_topics'].tolist()),
            'avg_topics_per_meeting': df['num_topics'].mean()
        },
        'financial': {
            'total_approved': df['total_amount_approved'].sum(),
            'avg_per_meeting': df['total_amount_approved'].mean()
        }
    }
    
    return analysis

def export_to_formats(df: pd.DataFrame, output_dir: str = "fed_minutes_output"):
    """Export parsed data to multiple formats."""

    os.makedirs(output_dir, exist_ok=True)

    # Main dataframe (already JSON strings)
    df.to_csv(f"{output_dir}/meetings_summary.csv", index=False)
    df.to_json(f"{output_dir}/meetings_full.json", orient='records', date_format='iso')

    # Decisions table
    decisions_records = []
    for _, row in df.iterrows():
        decisions = json.loads(row['decisions'])  # parse JSON string into list of dicts
        for decision in decisions:
            decisions_records.append({
                'filename': row['filename'],
                'date': row['date'],
                **{
                    k: json.dumps(v) if isinstance(v, list) else v
                    for k, v in decision.items()
                }
            })
    pd.DataFrame(decisions_records).to_csv(f"{output_dir}/all_decisions.csv", index=False)

    # Attendees table
    attendee_records = []
    for _, row in df.iterrows():
        attendees = json.loads(row['attendees'])
        for attendee in attendees:
            attendee_records.append({
                'filename': row['filename'],
                'date': row['date'],
                **attendee
            })
    pd.DataFrame(attendee_records).to_csv(f"{output_dir}/all_attendees.csv", index=False)

    # Topics table
    topic_records = []
    for _, row in df.iterrows():
        topics = json.loads(row['topics'])
        for topic in topics:
            topic_records.append({
                'filename': row['filename'],
                'date': row['date'],
                **{
                    k: json.dumps(v) if isinstance(v, list) else v
                    for k, v in topic.items()
                }
            })
    pd.DataFrame(topic_records).to_csv(f"{output_dir}/all_topics.csv", index=False)

    print(f"Exported data to {output_dir}/")

# ========================================
# Main Execution
# ========================================

if __name__ == "__main__":
    # Import config loader
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from src.utils.config import load_config
    
    # Load configuration
    config = load_config()
    
    # Initialize processor
    processor = FedMinutesBatchProcessor()
    
    # Process files (will convert PDFs to TXT first, then process TXT files)
    df = processor.process_directory(
        pdf_dir=config['paths']['pdf_dir'],
        txt_dir=config['paths']['txt_dir'],
        use_txt=True,        # Convert to TXT first for efficiency
        max_files=None       # Process all files
    )
    
    # Analyze results
    analysis = analyze_parsing_results(df)
    print("\n=== Analysis Results ===")
    print(json.dumps(analysis, indent=2, default=str))
    
    # Export to multiple formats
    export_to_formats(df, config['paths']['processed_dir'])
    
    print("\n✅ Processing complete!")
