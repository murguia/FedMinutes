import os
import pytest
from src.phase1_parsing.fed_parser import FedMinutesParser

# Test with TXT files since PDFs are not in git
TXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw", "TXTs"))

def txt_path(name):
    return os.path.join(TXT_DIR, name)

@pytest.mark.skipif(not os.path.exists(TXT_DIR), reason="TXT files not available")
def test_parse_real_txt_file():
    """Test parsing a real TXT file (converted from PDF)"""
    parser = FedMinutesParser()
    txt_file = txt_path("NT50005.txt")
    
    if not os.path.exists(txt_file):
        pytest.skip("Test file NT50005.txt not found")
    
    # Parse the file
    meeting = parser.parse_file(txt_file)
    
    # Basic assertions
    assert meeting is not None
    assert meeting.filename == "NT50005.txt"
    assert meeting.date is not None
    assert len(meeting.attendees) > 5
    assert len(meeting.decisions) > 3

@pytest.mark.skipif(not os.path.exists(TXT_DIR), reason="TXT files not available") 
def test_attendees_real_txt():
    """Test attendee extraction from real TXT file"""
    parser = FedMinutesParser()
    txt_file = txt_path("NT50005.txt")
    
    if not os.path.exists(txt_file):
        pytest.skip("Test file NT50005.txt not found")
        
    meeting = parser.parse_file(txt_file)
    attendees = meeting.attendees
    
    # Ensure multiple attendees parsed
    assert len(attendees) > 5
    # Check roles are preserved - look for Chairman/Vice Chairman
    roles = [a.role for a in attendees if a.role]
    titles = [a.title for a in attendees if a.title]
    assert any("Chairman" in str(role) for role in roles) or any("Chairman" in str(title) for title in titles)

@pytest.mark.skipif(not os.path.exists(TXT_DIR), reason="TXT files not available")
def test_decisions_real_txt():
    """Test decision extraction from real TXT file"""
    parser = FedMinutesParser()
    txt_file = txt_path("NT50005.txt")
    
    if not os.path.exists(txt_file):
        pytest.skip("Test file NT50005.txt not found")
        
    meeting = parser.parse_file(txt_file)
    decisions = meeting.decisions
    
    # Ensure multiple decisions are found
    assert len(decisions) > 3
    # Check that decisions have actions
    assert all(d.action for d in decisions)
    # Check for references if available
    all_references = []
    for d in decisions:
        if d.references:
            all_references.extend(d.references)
    
    # Should have some kind of references (Item No., letters, etc.)
    assert len(all_references) > 0
