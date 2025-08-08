#!/usr/bin/env python3
"""
Federal Reserve Minutes Parser - Validation and Testing Script

This script provides comprehensive validation and testing tools for the 
Fed Minutes Parser to ensure quality before processing the full corpus.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import Counter
import json
import os
import shutil
from datetime import datetime

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import the main parser
try:
    from src.phase1_parsing.fed_parser import FedMinutesParser, FedMinutesBatchProcessor
except ImportError:
    print("Error: Could not import Fed Minutes Parser. Please ensure the project structure is intact.")
    exit(1)

# ========================================
# Validation Functions
# ========================================

class FedMinutesValidator:
    """Comprehensive validation for Fed Minutes parsing results."""
    
    def __init__(self, df: pd.DataFrame):
        """Initialize validator with parsed dataframe."""
        self.df = df
        self.validation_results = {}
        
    def run_full_validation(self) -> Dict:
        """Run all validation checks and return comprehensive report."""
        print("="*60)
        print("FEDERAL RESERVE MINUTES - PARSING VALIDATION REPORT")
        print("="*60)
        print(f"Total files analyzed: {len(self.df)}")
        
        if len(self.df) == 0:
            print("⚠️ WARNING: No files were processed!")
            print("Please check:")
            print("  - PDF directory path is correct")
            print("  - PDF files exist in the specified directory")
            print("  - File permissions allow reading")
            print("="*60)
            return {"error": "No files processed", "total_files": 0}
        
        print(f"Date range: {self.df['date'].min()} to {self.df['date'].max()}")
        print("="*60 + "\n")
        
        # Run all checks
        self.check_dates()
        self.check_attendees()
        self.check_decisions()
        self.check_topics()
        self.check_financial_data()
        self.check_text_quality()
        self.calculate_quality_score()
        
        return self.validation_results
    
    def check_dates(self) -> None:
        """Validate date extraction."""
        print("📅 DATE VALIDATION")
        print("-" * 40)
        
        # Missing dates
        missing_dates = self.df['date'].isna().sum()
        missing_pct = (missing_dates / len(self.df)) * 100
        
        print(f"✓ Successfully extracted: {len(self.df) - missing_dates}/{len(self.df)} ({100-missing_pct:.1f}%)")
        
        if missing_dates > 0:
            print(f"⚠️ Missing dates: {missing_dates} files")
            print(f"   Files: {self.df[self.df['date'].isna()]['filename'].tolist()[:5]}")
            if missing_dates > 5:
                print(f"   ... and {missing_dates - 5} more")
        
        # Check for date anomalies
        if not self.df['date'].isna().all():
            date_df = self.df[self.df['date'].notna()].copy()
            date_df['year'] = pd.to_datetime(date_df['date']).dt.year
            
            # Check year distribution
            year_counts = date_df['year'].value_counts().sort_index()
            print(f"\n📊 Year distribution:")
            for year, count in year_counts.items():
                bar = "█" * int(count * 2)  # Scale for display
                print(f"   {year}: {bar} ({count})")
        
        self.validation_results['date_extraction_rate'] = 100 - missing_pct
        self.validation_results['missing_dates'] = missing_dates
        print()
    
    def check_attendees(self) -> None:
        """Validate attendee extraction."""
        print("👥 ATTENDEE VALIDATION")
        print("-" * 40)
        
        # Basic statistics
        avg_attendees = self.df['num_attendees'].mean()
        min_attendees = self.df['num_attendees'].min()
        max_attendees = self.df['num_attendees'].max()
        
        print(f"Average attendees per meeting: {avg_attendees:.1f}")
        print(f"Range: {min_attendees} - {max_attendees}")
        
        # Flag suspicious values
        low_attendance = self.df[self.df['num_attendees'] < 3]
        high_attendance = self.df[self.df['num_attendees'] > 50]
        
        if len(low_attendance) > 0:
            print(f"\n⚠️ Suspiciously low attendance (<3): {len(low_attendance)} files")
            for _, row in low_attendance.head(3).iterrows():
                print(f"   - {row['filename']}: {row['num_attendees']} attendees")
        
        if len(high_attendance) > 0:
            print(f"\n⚠️ Unusually high attendance (>50): {len(high_attendance)} files")
            for _, row in high_attendance.head(3).iterrows():
                print(f"   - {row['filename']}: {row['num_attendees']} attendees")
        
        # Check for common board members
        all_attendees = []
        for attendees_json in self.df['attendees']:
            if attendees_json:
                try:
                    if isinstance(attendees_json, str):
                        attendees = json.loads(attendees_json)
                    else:
                        attendees = attendees_json
                    all_attendees.extend([a.get('name', '') for a in attendees])
                except (json.JSONDecodeError, TypeError):
                    # Skip if we can't parse
                    continue
        
        attendee_counts = Counter(all_attendees)
        print(f"\n🏆 Most frequent attendees:")
        for name, count in attendee_counts.most_common(5):
            if name:
                print(f"   - {name}: {count} meetings")
        
        self.validation_results['avg_attendees'] = avg_attendees
        self.validation_results['low_attendance_files'] = len(low_attendance)
        print()
    
    def check_decisions(self) -> None:
        """Validate decision extraction."""
        print("📋 DECISION VALIDATION")
        print("-" * 40)
        
        # Basic statistics
        total_decisions = self.df['num_decisions'].sum()
        avg_decisions = self.df['num_decisions'].mean()
        
        print(f"Total decisions found: {total_decisions}")
        print(f"Average per meeting: {avg_decisions:.1f}")
        
        # Files with no decisions
        no_decisions = self.df[self.df['num_decisions'] == 0]
        if len(no_decisions) > 0:
            print(f"\n⚠️ Files with no decisions: {len(no_decisions)}")
            for _, row in no_decisions.head(5).iterrows():
                print(f"   - {row['filename']}")
        
        # Decision type distribution
        decision_types = Counter()
        for decisions_json in self.df['decisions']:
            if decisions_json:
                try:
                    if isinstance(decisions_json, str):
                        decisions = json.loads(decisions_json)
                    else:
                        decisions = decisions_json
                    for decision in decisions:
                        decision_types[decision.get('action', 'unknown')] += 1
                except (json.JSONDecodeError, TypeError):
                    continue
        
        print(f"\n📊 Decision types:")
        for action, count in decision_types.most_common(10):
            print(f"   - {action}: {count}")
        
        # Voting patterns
        unanimous_count = self.df['unanimous_decisions'].sum()
        if total_decisions > 0:
            unanimous_rate = (unanimous_count / total_decisions) * 100
            print(f"\n🗳️ Voting patterns:")
            print(f"   - Unanimous decisions: {unanimous_count}/{total_decisions} ({unanimous_rate:.1f}%)")
        
        self.validation_results['avg_decisions'] = avg_decisions
        self.validation_results['no_decision_files'] = len(no_decisions)
        self.validation_results['decision_types'] = dict(decision_types)
        print()


        print("\n🔍 DETAILED ANALYSIS OF PROBLEMATIC FILES")
        print("-" * 40)
        
        problem_files = self.df[(self.df['num_attendees'] < 3) | (self.df['num_decisions'] == 0)]
        
        if len(problem_files) > 0:
            print(f"Files with low attendance OR no decisions: {len(problem_files)}")
            
            # Group by issue type
            only_attendance = problem_files[problem_files['num_attendees'] < 3]['filename'].tolist()
            only_decisions = problem_files[problem_files['num_decisions'] == 0]['filename'].tolist()
            both_issues = list(set(only_attendance) & set(only_decisions))
            
            print(f"  - Low attendance only: {len(set(only_attendance) - set(both_issues))}")
            print(f"  - No decisions only: {len(set(only_decisions) - set(both_issues))}")
            print(f"  - Both issues: {len(both_issues)}")
            
            # Show sample files
            print("\nSample problematic files:")
            for _, row in problem_files.head(10).iterrows():
                print(f"  {row['filename']}:")
                print(f"    Date: {row['date']}")
                print(f"    Attendees: {row['num_attendees']}")
                print(f"    Decisions: {row['num_decisions']}")
                print(f"    Text length: {row.get('text_length', 'N/A')}")
            
            # Save to validation directory
            validation_dir = 'data/validation'
            os.makedirs(validation_dir, exist_ok=True)
            
            problem_path = os.path.join(validation_dir, 'problematic_files.csv')
            problem_files[['filename', 'date', 'num_attendees', 'num_decisions', 'text_length']].to_csv(
                problem_path, index=False
            )
            print(f"\n💾 Full list saved to {problem_path}")
        
    def check_topics(self) -> None:
        """Validate topic extraction."""
        print("📚 TOPIC VALIDATION")
        print("-" * 40)
        
        # Topic coverage
        files_with_topics = (self.df['num_topics'] > 0).sum()
        topic_coverage = (files_with_topics / len(self.df)) * 100
        
        print(f"Files with topics identified: {files_with_topics}/{len(self.df)} ({topic_coverage:.1f}%)")
        
        # Topic distribution
        all_topics = []
        for topics_json in self.df['main_topics']:
            if topics_json:
                try:
                    if isinstance(topics_json, str):
                        topics = json.loads(topics_json)
                    else:
                        topics = topics_json
                    all_topics.extend(topics)
                except (json.JSONDecodeError, TypeError):
                    continue
        
        topic_counts = Counter(all_topics)
        print(f"\n📊 Topic distribution:")
        for topic, count in topic_counts.most_common():
            bar = "█" * int(count / 2)  # Scale for display
            print(f"   {topic}: {bar} ({count})")
        
        self.validation_results['topic_coverage'] = topic_coverage
        self.validation_results['topic_distribution'] = dict(topic_counts)
        print()
    
    def check_financial_data(self) -> None:
        """Validate financial data extraction."""
        print("💰 FINANCIAL DATA VALIDATION")
        print("-" * 40)
        
        # Files with financial amounts
        files_with_amounts = (self.df['total_amount_approved'] > 0).sum()
        
        if files_with_amounts > 0:
            total_approved = self.df['total_amount_approved'].sum()
            avg_amount = self.df[self.df['total_amount_approved'] > 0]['total_amount_approved'].mean()
            
            print(f"Files with financial amounts: {files_with_amounts}")
            print(f"Total amount approved: ${total_approved:,.2f}")
            print(f"Average per meeting (when present): ${avg_amount:,.2f}")
            
            # Top approvals
            top_approvals = self.df.nlargest(5, 'total_amount_approved')[['filename', 'date', 'total_amount_approved']]
            if not top_approvals.empty:
                print(f"\n💵 Largest approvals:")
                for _, row in top_approvals.iterrows():
                    print(f"   - {row['filename']} ({row['date']}): ${row['total_amount_approved']:,.2f}")
        else:
            print("No financial amounts found in any documents")
        
        self.validation_results['files_with_amounts'] = files_with_amounts
        print()
    
    def check_text_quality(self) -> None:
        """Check text extraction quality."""
        print("📝 TEXT QUALITY VALIDATION")
        print("-" * 40)
        
        # Text length statistics
        avg_length = self.df['text_length'].mean()
        min_length = self.df['text_length'].min()
        max_length = self.df['text_length'].max()
        
        print(f"Average document length: {avg_length:,.0f} characters")
        print(f"Range: {min_length:,} - {max_length:,} characters")
        
        # Flag suspiciously short documents
        short_docs = self.df[self.df['text_length'] < 1000]
        if len(short_docs) > 0:
            print(f"\n⚠️ Suspiciously short documents (<1000 chars): {len(short_docs)}")
            for _, row in short_docs.head(3).iterrows():
                print(f"   - {row['filename']}: {row['text_length']} characters")
        
        self.validation_results['avg_text_length'] = avg_length
        self.validation_results['short_documents'] = len(short_docs)
        print()
    
    def calculate_quality_score(self) -> float:
        """Calculate overall quality score."""
        print("🎯 OVERALL QUALITY SCORE")
        print("-" * 40)
        
        # Weight different factors
        scores = {
            'date_extraction': self.validation_results['date_extraction_rate'] / 100,
            'attendee_quality': min(1.0, self.validation_results['avg_attendees'] / 10),
            'decision_quality': min(1.0, self.validation_results['avg_decisions'] / 5),
            'topic_coverage': self.validation_results['topic_coverage'] / 100,
            'text_quality': 1.0 - (self.validation_results['short_documents'] / len(self.df))
        }
        
        # Calculate weighted average
        weights = {
            'date_extraction': 0.25,
            'attendee_quality': 0.20,
            'decision_quality': 0.25,
            'topic_coverage': 0.15,
            'text_quality': 0.15
        }
        
        overall_score = sum(scores[k] * weights[k] for k in scores) * 100
        
        print(f"Component scores:")
        for component, score in scores.items():
            print(f"   - {component}: {score*100:.1f}%")
        
        print(f"\n📈 OVERALL QUALITY SCORE: {overall_score:.1f}%")
        
        if overall_score >= 80:
            print("✅ Excellent quality - Ready for full processing!")
        elif overall_score >= 60:
            print("⚠️ Good quality - Minor adjustments may help")
        else:
            print("❌ Quality issues detected - Review parser configuration")
        
        self.validation_results['overall_score'] = overall_score
        self.validation_results['component_scores'] = scores
        
        return overall_score

# ========================================
# Testing Functions
# ========================================

def test_single_file(filepath: str) -> None:
    """Test parser on a single file with detailed output."""
    print(f"\n{'='*60}")
    print(f"TESTING SINGLE FILE: {filepath}")
    print('='*60)
    
    parser = FedMinutesParser()
    
    try:
        meeting = parser.parse_file(filepath)
        
        print(f"\n✅ Successfully parsed!")
        print(f"\n📅 Date: {meeting.date}")
        print(f"📄 Type: {meeting.meeting_type}")
        print(f"📄 Pages: {meeting.total_pages}")
        
        print(f"\n👥 Attendees ({len(meeting.attendees)}):")
        for i, attendee in enumerate(meeting.attendees[:5], 1):
            print(f"   {i}. {attendee.name}")
            if attendee.title:
                print(f"      Title: {attendee.title}")
            if attendee.role:
                print(f"      Role: {attendee.role}")
        if len(meeting.attendees) > 5:
            print(f"   ... and {len(meeting.attendees) - 5} more")
        
        print(f"\n📋 Decisions ({len(meeting.decisions)}):")
        for i, decision in enumerate(meeting.decisions[:3], 1):
            print(f"   {i}. {decision.action.upper()}: {decision.subject[:60]}...")
            if decision.vote:
                print(f"      Vote: {decision.vote}")
            if decision.financial_amounts:
                print(f"      Amounts: {', '.join(decision.financial_amounts)}")
            if decision.institutions:
                print(f"      Institutions: {', '.join(decision.institutions[:2])}")
        if len(meeting.decisions) > 3:
            print(f"   ... and {len(meeting.decisions) - 3} more")
        
        print(f"\n📚 Topics ({len(meeting.topics)}):")
        for topic in meeting.topics:
            print(f"   - {topic.category}: {topic.title}")
            print(f"     Length: {topic.discussion_length:,} characters")
        
    except Exception as e:
        print(f"\n❌ Error parsing file: {e}")
        import traceback
        traceback.print_exc()

def test_subset(pdf_dir: str, txt_dir: str, num_files: int = 5) -> pd.DataFrame:
    """Test parser on a subset of files."""
    print(f"\n{'='*60}")
    print(f"TESTING SUBSET: {num_files} files")
    print('='*60)
    
    # Debug: print paths
    print(f"PDF directory: {pdf_dir}")
    print(f"TXT directory: {txt_dir}")
    print(f"Current working directory: {os.getcwd()}")
    
    processor = FedMinutesBatchProcessor()
    
    # Process subset
    df = processor.process_directory(
        pdf_dir=pdf_dir,
        txt_dir=txt_dir,
        use_txt=True,
        max_files=num_files
    )
    
    print(f"\n✅ Processed {len(df)} files")
    
    # Run validation
    validator = FedMinutesValidator(df)
    validator.run_full_validation()
    
    return df

def create_test_set(source_dir: str, test_dir: str, num_files: int = 10) -> None:
    """Create a test set with specific number of files."""
    os.makedirs(test_dir, exist_ok=True)
    
    files = sorted([f for f in os.listdir(source_dir) if f.endswith('.pdf')])[:num_files]
    
    print(f"Creating test set with {len(files)} files in {test_dir}/")
    
    for file in files:
        shutil.copy(
            os.path.join(source_dir, file),
            os.path.join(test_dir, file)
        )
        print(f"   Copied: {file}")
    
    print(f"✅ Test set created with {len(files)} files")

def compare_with_original(df: pd.DataFrame, index: int = 0) -> None:
    """Compare parsed output with original text."""
    if index >= len(df):
        print(f"Index {index} out of range. DataFrame has {len(df)} rows.")
        return
    
    row = df.iloc[index]
    
    print(f"\n{'='*60}")
    print(f"COMPARISON: {row['filename']}")
    print('='*60)
    
    # Try to load original text
    txt_file = f"TXTs/{row['filename'].replace('.pdf', '.txt')}"
    if os.path.exists(txt_file):
        with open(txt_file, 'r', encoding='utf-8') as f:
            original = f.read()
        
        print("\n--- ORIGINAL TEXT (first 500 chars) ---")
        print(original[:500])
        print("\n--- PARSED RESULTS ---")
    else:
        print("\n--- PARSED RESULTS ---")
    
    print(f"Date: {row['date']}")
    print(f"Meeting type: {row.get('meeting_type', 'N/A')}")
    print(f"Attendees: {row['num_attendees']}")
    print(f"Decisions: {row['num_decisions']}")
    print(f"Topics: {row['num_topics']}")
    
    if row['decisions']:
        print(f"\nFirst decision:")
        decision = row['decisions'][0]
        print(f"  Action: {decision.get('action', 'N/A')}")
        print(f"  Subject: {decision.get('subject', 'N/A')}")
        print(f"  Full text: {decision.get('full_text', 'N/A')[:200]}...")

# ========================================
# Main Execution
# ========================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate Fed Minutes Parser")
    parser.add_argument('--mode', choices=['single', 'subset', 'validate', 'create-test'],
                       default='subset', help='Testing mode')
    parser.add_argument('--file', type=str, help='Single file to test')
    parser.add_argument('--num-files', type=int, default=10,
                       help='Number of files to test in subset mode')
    parser.add_argument('--pdf-dir', type=str, default='PDFs',
                       help='Directory containing PDF files')
    parser.add_argument('--txt-dir', type=str, default='TXTs',
                       help='Directory for TXT files')
    parser.add_argument('--csv-file', type=str,
                       help='CSV file to validate (for validate mode)')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        if not args.file:
            print("Please specify a file with --file")
        else:
            test_single_file(args.file)
    
    elif args.mode == 'subset':
        df = test_subset(args.pdf_dir, args.txt_dir, args.num_files)
        
        # Save test results to validation directory
        import os
        validation_dir = 'data/validation'
        os.makedirs(validation_dir, exist_ok=True)
        
        results_path = os.path.join(validation_dir, 'test_results.csv')
        df.to_csv(results_path, index=False)
        print(f"\n💾 Test results saved to {results_path}")
    
    elif args.mode == 'validate':
        if args.csv_file:
            df = pd.read_csv(args.csv_file)
            # Convert JSON strings back to lists/dicts if needed
            for col in ['attendees', 'decisions', 'topics', 'main_topics']:
                if col in df.columns:
                    df[col] = df[col].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
        else:
            print("Please specify a CSV file with --csv-file")
            exit(1)
        
        validator = FedMinutesValidator(df)
        results = validator.run_full_validation()
        
        # Save validation report to validation directory
        validation_dir = 'data/validation'
        os.makedirs(validation_dir, exist_ok=True)
        
        report_path = os.path.join(validation_dir, 'validation_report.json')
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Validation report saved to {report_path}")
    
    elif args.mode == 'create-test':
        create_test_set(args.pdf_dir, 'test_PDFs', args.num_files)
