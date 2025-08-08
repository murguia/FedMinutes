import os
from src.phase1_parsing.fed_parser import extract_text, extract_metadata, extract_attendees, extract_decisions

PDF_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "PDFs"))

def pdf_path(name):
    return os.path.join(PDF_DIR, name)

def test_attendees_real_pdf():
    text = extract_text(pdf_path("NT50005.pdf"))
    attendees = extract_attendees(text)
    # Ensure multiple attendees parsed
    assert len(attendees) > 5
    # Check roles are preserved
    assert any("Chairman" in a or "Vice Chairman" in a for a in attendees)

def test_decisions_real_pdf_items_and_letters():
    text = extract_text(pdf_path("NT50005.pdf"))
    decisions = extract_decisions(text)
    # Ensure multiple decisions are found
    assert len(decisions) > 3
    # Ensure at least one decision references "Item No."
    assert any(any("Item No." in ref for ref in d["references"]) for d in decisions)
    # Ensure at least one decision references a "Letter to"
    assert any(any("Letter to" in ref or "Letters to" in ref for ref in d["references"]) for d in decisions)

def test_metadata_real_pdf():
    text = extract_text(pdf_path("NT50005.pdf"))
    date, attendees, decisions = extract_metadata(text)
    # Date is parsed
    assert "1967" in date
    # Attendees parsed
    assert len(attendees) > 0
    # Decisions include at least one qualifier or reference
    assert any(d["qualifiers"] or d["references"] for d in decisions)
