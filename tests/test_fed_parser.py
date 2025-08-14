import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.phase1_parsing.fed_parser import FedMinutesParser
from datetime import datetime

def test_date_extraction():
    parser = FedMinutesParser()
    text = "Minutes of January 3, 1967 PRESENT: Mr. Robertson"
    date = parser._extract_date(text)
    assert date is not None
    assert date.year == 1967
    assert date.month == 1
    assert date.day == 3

def test_attendees_extraction():
    parser = FedMinutesParser()
    text = "PRESENT: Mr. Robertson, Vice Chairman, Mr. Shepardson"
    attendees = parser._extract_attendees(text)
    assert len(attendees) >= 1
    # Check that at least one attendee was found
    attendee_names = [a.name for a in attendees]
    assert len(attendee_names) > 0
    # Check that we found some recognizable names
    all_text = ' '.join(attendee_names)
    assert "Robertson" in all_text or "Shepardson" in all_text

def test_decision_extraction():
    parser = FedMinutesParser()
    text = "Approved unanimously after consideration: Item No. 8."
    decisions = parser._extract_decisions(text)
    assert len(decisions) >= 1
    assert decisions[0].action == "approved"
    # Check that it found some reference or content
    assert len(decisions[0].full_text) > 0

def test_text_cleaning():
    parser = FedMinutesParser()
    text = "M r . Robertson    and   M rs . Smith"
    cleaned = parser._clean_text(text)
    assert "Mr. Robertson" in cleaned
    assert "Mrs. Smith" in cleaned
    # Check that multiple spaces are reduced
    assert "   " not in cleaned
