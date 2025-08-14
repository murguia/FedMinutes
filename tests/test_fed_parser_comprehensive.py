"""Comprehensive unit tests for the Fed Minutes Parser"""

import pytest
from datetime import datetime
from src.phase1_parsing.fed_parser import (
    FedMinutesParser, FedMinutesPatterns, 
    Attendee, Decision, Topic, Meeting
)


class TestFedMinutesPatterns:
    """Test the pattern definitions"""
    
    def test_date_patterns(self):
        patterns = FedMinutesPatterns()
        assert len(patterns.DATE_PATTERNS) > 0
        # Test that patterns are tuples with regex and group number
        for pattern, group in patterns.DATE_PATTERNS:
            assert isinstance(pattern, str)
            assert isinstance(group, int)
    
    def test_decision_actions(self):
        patterns = FedMinutesPatterns()
        assert "approved" in patterns.DECISION_ACTIONS
        assert "denied" in patterns.DECISION_ACTIONS
        assert "authorized" in patterns.DECISION_ACTIONS
        assert len(patterns.DECISION_ACTIONS) >= 15  # Should have many actions
    
    def test_voting_patterns(self):
        patterns = FedMinutesPatterns()
        assert "unanimous" in patterns.VOTING_PATTERNS
        assert "split" in patterns.VOTING_PATTERNS
        assert all(isinstance(p, str) for p in patterns.VOTING_PATTERNS.values())


class TestDataClasses:
    """Test the data class structures"""
    
    def test_attendee_creation(self):
        attendee = Attendee(
            name="Mr. Robertson",
            title="Vice Chairman",
            organization="Federal Reserve",
            role="Chairman"
        )
        assert attendee.name == "Mr. Robertson"
        assert attendee.title == "Vice Chairman"
        assert attendee.organization == "Federal Reserve"
        assert attendee.role == "Chairman"
    
    def test_decision_creation(self):
        decision = Decision(
            action="approved",
            subject="Interest rate change",
            vote="unanimous",
            conditions=["subject to review"],
            references=["Item No. 1"],
            financial_amounts=["$1,000,000"],
            institutions=["Federal Reserve Bank"],
            full_text="Approved unanimously the interest rate change"
        )
        assert decision.action == "approved"
        assert decision.vote == "unanimous"
        assert len(decision.conditions) == 1
        assert len(decision.references) == 1
    
    def test_topic_creation(self):
        topic = Topic(
            category="Monetary Policy",
            title="Interest Rate Discussion",
            discussion_length=1500,
            key_points=["Inflation concerns", "Economic growth"],
            page_references=[1, 2, 3]
        )
        assert topic.category == "Monetary Policy"
        assert topic.discussion_length == 1500
        assert len(topic.key_points) == 2


class TestTextCleaning:
    """Test text cleaning and normalization"""
    
    def test_clean_fr_numbers(self):
        parser = FedMinutesParser()
        text = "This is FR 1234 a test FR1234 document"
        cleaned = parser._clean_text(text)
        assert "FR 1234" not in cleaned
        assert "FR1234" not in cleaned
    
    def test_fix_ocr_issues(self):
        parser = FedMinutesParser()
        text = "01Connell met with 01Brien"
        cleaned = parser._clean_text(text)
        assert "O'Connell" in cleaned
        assert "O'Brien" in cleaned
    
    def test_fix_spaced_honorifics(self):
        parser = FedMinutesParser()
        text = "M r . Smith and M rs . Jones attended"
        cleaned = parser._clean_text(text)
        assert "Mr. Smith" in cleaned
        assert "Mrs. Jones" in cleaned
    
    def test_whitespace_normalization(self):
        parser = FedMinutesParser()
        text = "This   has     multiple    spaces\n\n\n\nand newlines"
        cleaned = parser._clean_text(text)
        assert "   " not in cleaned  # No triple spaces
        assert "\n\n\n" not in cleaned  # No triple newlines


class TestDateExtraction:
    """Test date extraction with various formats"""
    
    def test_standard_date_format(self):
        parser = FedMinutesParser()
        text = "Minutes of the meeting held on January 15, 1967"
        date = parser._extract_date(text)
        assert date is not None
        assert date.year == 1967
        assert date.month == 1
        assert date.day == 15
    
    def test_minutes_for_date_format(self):
        parser = FedMinutesParser()
        text = "Minutes for December 30, 1968"
        date = parser._extract_date(text)
        assert date is not None
        assert date.year == 1968
        assert date.month == 12
        assert date.day == 30
    
    def test_weekday_date_format(self):
        parser = FedMinutesParser()
        text = "Meeting held Monday, March 5, 1970"
        date = parser._extract_date(text)
        assert date is not None
        assert date.year == 1970
        assert date.month == 3
        assert date.day == 5
    
    def test_slash_date_format(self):
        parser = FedMinutesParser()
        text = "Date: 06/15/1969"
        date = parser._extract_date(text)
        assert date is not None
        assert date.year == 1969
        assert date.month == 6
        assert date.day == 15
    
    def test_no_date_found(self):
        parser = FedMinutesParser()
        text = "This text has no date information"
        date = parser._extract_date(text)
        assert date is None


class TestMeetingTypeDetection:
    """Test meeting type determination"""
    
    def test_regular_meeting(self):
        parser = FedMinutesParser()
        text = "Minutes of the regular meeting of the Board"
        meeting_type = parser._determine_meeting_type(text)
        assert meeting_type == "regular"
    
    def test_special_meeting(self):
        parser = FedMinutesParser()
        text = "This special meeting was called to discuss"
        meeting_type = parser._determine_meeting_type(text)
        assert meeting_type == "special"
    
    def test_emergency_meeting(self):
        parser = FedMinutesParser()
        text = "An emergency meeting was convened"
        meeting_type = parser._determine_meeting_type(text)
        assert meeting_type == "emergency"
    
    def test_default_meeting_type(self):
        parser = FedMinutesParser()
        text = "Minutes of the meeting"
        meeting_type = parser._determine_meeting_type(text)
        assert meeting_type == "regular"  # Default


class TestAttendeeExtraction:
    """Test attendee extraction scenarios"""
    
    def test_simple_attendee_list(self):
        parser = FedMinutesParser()
        text = "PRESENT: Mr. Smith, Mr. Jones, Mrs. Williams"
        attendees = parser._extract_attendees(text)
        assert len(attendees) == 3
        names = [a.name for a in attendees]
        assert "Mr. Smith" in names
        assert "Mr. Jones" in names
        assert "Mrs. Williams" in names
    
    def test_attendees_with_titles(self):
        parser = FedMinutesParser()
        text = "PRESENT: Mr. Robertson, Vice Chairman, Mr. Mitchell, Governor"
        attendees = parser._extract_attendees(text)
        assert len(attendees) >= 2
        # Find Robertson
        robertson = next((a for a in attendees if "Robertson" in a.name), None)
        assert robertson is not None
        assert robertson.title == "Vice Chairman" or robertson.role == "Vice Chairman"
    
    def test_messrs_format(self):
        parser = FedMinutesParser()
        text = "PRESENT: Messrs. Smith and Jones, Attorneys"
        attendees = parser._extract_attendees(text)
        assert len(attendees) == 2
        names = [a.name for a in attendees]
        assert any("Smith" in name for name in names)
        assert any("Jones" in name for name in names)
    
    def test_complex_titles(self):
        parser = FedMinutesParser()
        text = "PRESENT: Mr. Holland, Manager, Division of Data Processing"
        attendees = parser._extract_attendees(text)
        assert len(attendees) >= 1
        holland = attendees[0]
        assert "Holland" in holland.name
        assert holland.organization == "Division of Data Processing" or "Division" in str(holland.title)


class TestDecisionExtraction:
    """Test decision extraction scenarios"""
    
    def test_simple_approval(self):
        parser = FedMinutesParser()
        text = "The Board approved the application."
        decisions = parser._extract_decisions(text)
        assert len(decisions) >= 1
        assert decisions[0].action == "approved"
    
    def test_unanimous_decision(self):
        parser = FedMinutesParser()
        text = "The proposal was approved unanimously by all members present."
        decisions = parser._extract_decisions(text)
        assert len(decisions) >= 1
        assert decisions[0].action == "approved"
        assert decisions[0].vote == "unanimous"
    
    def test_conditional_approval(self):
        parser = FedMinutesParser()
        text = "Approved the merger subject to certain conditions outlined in the letter."
        decisions = parser._extract_decisions(text)
        assert len(decisions) >= 1
        assert decisions[0].action == "approved"
        assert len(decisions[0].conditions) > 0
    
    def test_financial_decision(self):
        parser = FedMinutesParser()
        text = "The Board authorized an expenditure of $50,000 for the project."
        decisions = parser._extract_decisions(text)
        assert len(decisions) >= 1
        assert decisions[0].action == "authorized"
        assert decisions[0].financial_amounts is not None
        assert "$50,000" in decisions[0].financial_amounts
    
    def test_decision_with_references(self):
        parser = FedMinutesParser()
        text = "Approved Item No. 5 as recommended in the memorandum."
        decisions = parser._extract_decisions(text)
        assert len(decisions) >= 1
        assert decisions[0].action == "approved"
        assert decisions[0].references is not None
        assert any("Item No. 5" in ref for ref in decisions[0].references)


class TestInstitutionExtraction:
    """Test extraction of institution names"""
    
    def test_bank_name_extraction(self):
        parser = FedMinutesParser()
        text = "Application from First National Bank was approved"
        institutions = parser._extract_institutions(text)
        assert len(institutions) > 0
        assert "First National Bank" in institutions
    
    def test_federal_reserve_bank(self):
        parser = FedMinutesParser()
        text = "The Federal Reserve Bank of New York submitted a report"
        institutions = parser._extract_institutions(text)
        assert len(institutions) > 0
        assert any("Federal Reserve Bank" in inst for inst in institutions)
    
    def test_multiple_institutions(self):
        parser = FedMinutesParser()
        text = "Merger between Chase Bank and Morgan Trust Company"
        institutions = parser._extract_institutions(text)
        assert len(institutions) >= 2
        assert any("Chase Bank" in inst for inst in institutions)
        assert any("Morgan Trust Company" in inst for inst in institutions)


class TestTopicExtraction:
    """Test topic extraction and categorization"""
    
    def test_monetary_policy_topic(self):
        parser = FedMinutesParser()
        text = "Discussion of interest rates and discount rate policy"
        topics = parser._extract_topics(text)
        assert len(topics) > 0
        assert any(t.category == "Monetary Policy" for t in topics)
    
    def test_banking_regulation_topic(self):
        parser = FedMinutesParser()
        text = "Review of bank holding company applications and merger requests"
        topics = parser._extract_topics(text)
        assert len(topics) > 0
        assert any(t.category == "Banking Regulation" for t in topics)
    
    def test_international_finance_topic(self):
        parser = FedMinutesParser()
        text = "Foreign exchange operations and international balance of payments"
        topics = parser._extract_topics(text)
        assert len(topics) > 0
        assert any(t.category == "International Finance" for t in topics)


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_text(self):
        parser = FedMinutesParser()
        text = ""
        date = parser._extract_date(text)
        attendees = parser._extract_attendees(text)
        decisions = parser._extract_decisions(text)
        assert date is None
        assert len(attendees) == 0
        assert len(decisions) == 0
    
    def test_malformed_text(self):
        parser = FedMinutesParser()
        text = "This is just random text with no structure"
        # Should not raise exceptions
        date = parser._extract_date(text)
        attendees = parser._extract_attendees(text)
        decisions = parser._extract_decisions(text)
        # May or may not find anything, but shouldn't crash
        assert isinstance(attendees, list)
        assert isinstance(decisions, list)
    
    def test_unicode_handling(self):
        parser = FedMinutesParser()
        text = "PRESENT: Mr. O'Brien, Mrs. Müller, M. François"
        attendees = parser._extract_attendees(text)
        # Should handle unicode characters gracefully
        assert len(attendees) >= 1
        names = [a.name for a in attendees]
        assert any("O'Brien" in name or "Brien" in name for name in names)


class TestIntegration:
    """Test full parsing integration"""
    
    def test_parse_complete_document(self):
        parser = FedMinutesParser()
        text = """
        Minutes of January 15, 1967
        
        PRESENT: Mr. Robertson, Vice Chairman
                 Mr. Shepardson
                 Mr. Mitchell, Governor
                 
        The Board approved unanimously the application of First National Bank
        for a merger with City Trust Company, subject to conditions outlined
        in Item No. 3.
        
        The Board also authorized an expenditure of $25,000 for renovations.
        """
        
        # Clean the text first
        text = parser._clean_text(text)
        
        # Extract components
        date = parser._extract_date(text)
        attendees = parser._extract_attendees(text)
        decisions = parser._extract_decisions(text)
        meeting_type = parser._determine_meeting_type(text)
        
        # Verify extractions
        assert date is not None
        assert date.year == 1967
        assert date.month == 1
        assert date.day == 15
        
        assert len(attendees) >= 2
        assert any("Robertson" in a.name for a in attendees)
        
        assert len(decisions) >= 2
        assert any(d.action == "approved" for d in decisions)
        assert any(d.financial_amounts and "$25,000" in d.financial_amounts for d in decisions)
        
        assert meeting_type == "regular"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])