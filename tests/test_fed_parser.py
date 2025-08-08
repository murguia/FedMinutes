import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.phase1_parsing import fed_parser

def test_date_extraction():
    text = "Minutes of January 3, 1967 PRESENT: Mr. Robertson"
    assert fed_parser.extract_date(text) == "January 3, 1967"

def test_attendees_extraction():
    text = "PRESENT: Mr. Robertson, Vice Chairman Mr. Shepardson"
    attendees = fed_parser.extract_attendees(text)
    assert attendees[0] == "Mr. Robertson, Vice Chairman"
    assert "Mr. Shepardson" in attendees

def test_decision_extraction():
    text = "Approved unanimously after consideration: Item No. 8."
    decisions = fed_parser.extract_decisions(text)
    print(decisions)
    assert decisions[0]["action"] == "approved"
    assert "unanimously" in decisions[0]["qualifiers"]
    assert "Item No. 8" in decisions[0]["references"]

def test_full_metadata():
    text = """
    Minutes of January 3, 1967
    PRESENT: Mr. Robertson, Vice Chairman
    Approved unanimously after consideration: Item No. 8.
    """
    date, attendees, decisions = fed_parser.extract_metadata(text)
    assert date == "January 3, 1967"
    assert len(attendees) == 1
    assert decisions[0]["action"] == "approved"
