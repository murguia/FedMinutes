"""Unit tests for the Fed Minutes Batch Processor"""

import pytest
import pandas as pd
import os
import tempfile
import shutil
from pathlib import Path
from src.phase1_parsing.fed_parser import (
    FedMinutesBatchProcessor, Meeting, Attendee, Decision, Topic
)


class TestBatchProcessor:
    """Test the batch processing functionality"""
    
    def test_processor_initialization(self):
        processor = FedMinutesBatchProcessor()
        assert processor.parser is not None
        assert hasattr(processor, 'convert_pdfs_to_txt')
        assert hasattr(processor, 'process_directory')
    
    def test_meetings_to_dataframe_empty(self):
        processor = FedMinutesBatchProcessor()
        meetings = []
        df = processor._meetings_to_dataframe(meetings)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
    
    def test_meetings_to_dataframe_single(self):
        processor = FedMinutesBatchProcessor()
        
        # Create a sample meeting
        meeting = Meeting(
            filename="test.txt",
            date=pd.Timestamp("1967-01-01"),
            meeting_type="regular",
            attendees=[
                Attendee(name="Mr. Smith", title="Chairman"),
                Attendee(name="Mr. Jones", title="Governor")
            ],
            decisions=[
                Decision(
                    action="approved",
                    subject="Test decision",
                    vote="unanimous",
                    full_text="Approved unanimously"
                )
            ],
            topics=[
                Topic(
                    category="Monetary Policy",
                    title="Interest Rates",
                    discussion_length=500,
                    key_points=["Point 1"],
                    page_references=[1]
                )
            ],
            total_pages=10,
            raw_text="Sample text"
        )
        
        meetings = [meeting]
        df = processor._meetings_to_dataframe(meetings)
        
        assert len(df) == 1
        assert df.iloc[0]['filename'] == "test.txt"
        assert df.iloc[0]['num_attendees'] == 2
        assert df.iloc[0]['num_decisions'] == 1
        assert df.iloc[0]['num_topics'] == 1
        assert df.iloc[0]['meeting_type'] == "regular"
    
    def test_sum_amounts_empty(self):
        processor = FedMinutesBatchProcessor()
        decisions = []
        total = processor._sum_amounts(decisions)
        assert total == 0
    
    def test_sum_amounts_single(self):
        processor = FedMinutesBatchProcessor()
        decisions = [
            Decision(
                action="approved",
                subject="Budget",
                financial_amounts=["$1,000"],
                full_text=""
            )
        ]
        total = processor._sum_amounts(decisions)
        assert total == 1000
    
    def test_sum_amounts_multiple(self):
        processor = FedMinutesBatchProcessor()
        decisions = [
            Decision(
                action="approved",
                subject="Budget 1",
                financial_amounts=["$1,000", "$500"],
                full_text=""
            ),
            Decision(
                action="approved",
                subject="Budget 2",
                financial_amounts=["$2,500"],
                full_text=""
            )
        ]
        total = processor._sum_amounts(decisions)
        assert total == 4000
    
    def test_sum_amounts_with_millions(self):
        processor = FedMinutesBatchProcessor()
        decisions = [
            Decision(
                action="approved",
                subject="Large budget",
                financial_amounts=["$5.5 million"],
                full_text=""
            )
        ]
        total = processor._sum_amounts(decisions)
        assert total == 5_500_000
    
    def test_sum_amounts_with_billions(self):
        processor = FedMinutesBatchProcessor()
        decisions = [
            Decision(
                action="approved",
                subject="Huge budget",
                financial_amounts=["$2.3 billion"],
                full_text=""
            )
        ]
        total = processor._sum_amounts(decisions)
        assert total == 2_300_000_000


class TestDataExport:
    """Test export functionality"""
    
    def test_export_to_formats(self):
        """Test export_to_formats function"""
        from src.phase1_parsing.fed_parser import export_to_formats
        
        # Create test data
        test_data = pd.DataFrame({
            'filename': ['test1.txt', 'test2.txt'],
            'date': pd.to_datetime(['2023-01-01', '2023-01-02']),
            'meeting_type': ['regular', 'special'],
            'total_pages': [10, 20],
            'num_attendees': [5, 8],
            'num_decisions': [3, 5],
            'num_topics': [2, 3],
            'attendees': ['[{"name": "Mr. Smith"}]', '[{"name": "Mr. Jones"}]'],
            'decisions': ['[{"action": "approved", "subject": "Test"}]', '[]'],
            'topics': ['[{"category": "Policy", "title": "Test"}]', '[]'],
            'text_length': [1000, 2000],
            'board_members': ['["Mr. Smith"]', '[]'],
            'unanimous_decisions': [1, 0],
            'total_amount_approved': [1000.0, 0.0],
            'main_topics': ['["Policy"]', '[]']
        })
        
        # Create temporary directory for output
        with tempfile.TemporaryDirectory() as tmpdir:
            export_to_formats(test_data, tmpdir)
            
            # Check that files were created
            assert os.path.exists(os.path.join(tmpdir, 'meetings_summary.csv'))
            assert os.path.exists(os.path.join(tmpdir, 'meetings_full.json'))
            assert os.path.exists(os.path.join(tmpdir, 'all_decisions.csv'))
            assert os.path.exists(os.path.join(tmpdir, 'all_attendees.csv'))
            assert os.path.exists(os.path.join(tmpdir, 'all_topics.csv'))
            
            # Verify CSV content
            summary_df = pd.read_csv(os.path.join(tmpdir, 'meetings_summary.csv'))
            assert len(summary_df) == 2
            assert 'filename' in summary_df.columns
            
            # Verify decisions CSV
            decisions_df = pd.read_csv(os.path.join(tmpdir, 'all_decisions.csv'))
            assert len(decisions_df) >= 1  # At least one decision
            assert 'action' in decisions_df.columns


class TestAnalysisFunctions:
    """Test analysis and validation functions"""
    
    def test_analyze_parsing_results(self):
        """Test analyze_parsing_results function"""
        from src.phase1_parsing.fed_parser import analyze_parsing_results
        
        # Create test dataframe
        test_df = pd.DataFrame({
            'date': pd.to_datetime(['2023-01-01', '2023-01-02', None]),
            'num_attendees': [10, 15, 5],
            'num_decisions': [5, 8, 0],
            'num_topics': [3, 4, 1],
            'unanimous_decisions': [2, 3, 0],
            'total_amount_approved': [1000, 5000, 0],
            'main_topics': [['Policy'], ['Banking', 'Policy'], []]
        })
        
        analysis = analyze_parsing_results(test_df)
        
        # Check structure
        assert 'total_files' in analysis
        assert 'date_coverage' in analysis
        assert 'attendance' in analysis
        assert 'decisions' in analysis
        assert 'topics' in analysis
        assert 'financial' in analysis
        
        # Check values
        assert analysis['total_files'] == 3
        assert analysis['date_coverage']['missing_dates'] == 1
        assert analysis['attendance']['avg_attendees'] == 10.0
        assert analysis['decisions']['total'] == 13
        assert analysis['financial']['total_approved'] == 6000


class TestFileProcessing:
    """Test file processing with mock data"""
    
    @pytest.fixture
    def temp_test_dir(self):
        """Create a temporary directory with test files"""
        temp_dir = tempfile.mkdtemp()
        
        # Create test TXT files
        txt_dir = os.path.join(temp_dir, "TXTs")
        os.makedirs(txt_dir)
        
        # Sample content
        test_content = """
        Minutes of January 1, 1967
        
        PRESENT: Mr. Test, Chairman
                 Mr. Sample, Governor
        
        The Board approved the test application.
        """
        
        # Create a few test files
        for i in range(3):
            with open(os.path.join(txt_dir, f"test{i}.txt"), 'w') as f:
                f.write(test_content)
        
        yield temp_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_process_directory_txt_files(self, temp_test_dir):
        """Test processing a directory of TXT files"""
        processor = FedMinutesBatchProcessor()
        
        df = processor.process_directory(
            pdf_dir=os.path.join(temp_test_dir, "PDFs"),  # Doesn't exist
            txt_dir=os.path.join(temp_test_dir, "TXTs"),
            use_txt=True,
            max_files=2
        )
        
        assert len(df) == 2  # Limited to 2 files
        assert all(df['num_attendees'] >= 1)
        assert all(df['num_decisions'] >= 1)
    
    def test_convert_pdfs_to_txt_directory_creation(self, temp_test_dir):
        """Test that convert_pdfs_to_txt creates the output directory"""
        processor = FedMinutesBatchProcessor()
        
        pdf_dir = os.path.join(temp_test_dir, "PDFs")
        txt_dir = os.path.join(temp_test_dir, "NewTXTs")
        
        # Create empty PDF dir
        os.makedirs(pdf_dir)
        
        # Run conversion (no PDFs, but should create directory)
        processor.convert_pdfs_to_txt(pdf_dir, txt_dir)
        
        assert os.path.exists(txt_dir)


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_process_directory_nonexistent_path(self):
        """Test handling of non-existent directories"""
        processor = FedMinutesBatchProcessor()
        
        df = processor.process_directory(
            pdf_dir="/nonexistent/path",
            txt_dir="/nonexistent/path",
            use_txt=True
        )
        
        assert len(df) == 0  # Should return empty dataframe
    
    def test_invalid_json_in_dataframe(self):
        """Test handling of invalid JSON in dataframe conversion"""
        processor = FedMinutesBatchProcessor()
        
        # Create meeting with various data types
        meeting = Meeting(
            filename="test.txt",
            date=None,  # Missing date
            meeting_type="regular",
            attendees=[],  # Empty attendees
            decisions=[],  # Empty decisions
            topics=[],  # Empty topics
            total_pages=0,
            raw_text=""
        )
        
        df = processor._meetings_to_dataframe([meeting])
        assert len(df) == 1
        assert df.iloc[0]['num_attendees'] == 0
        assert df.iloc[0]['num_decisions'] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])