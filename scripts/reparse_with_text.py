#!/usr/bin/env python3
"""Re-run parsing to include raw text for embeddings"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.phase1_parsing.fed_parser import FedMinutesBatchProcessor
from src.utils.config import load_config

def main():
    print("Re-parsing Fed Minutes to include raw text for embeddings...")
    
    # Load config
    config = load_config()
    
    # Create processor
    processor = FedMinutesBatchProcessor()
    
    # Process all TXT files (includes raw text now)
    df = processor.process_directory(
        pdf_dir=config['paths']['pdf_dir'],
        txt_dir=config['paths']['txt_dir'],
        use_txt=True  # Use existing TXT files
    )
    
    print(f"\nProcessed {len(df)} meetings")
    
    # Verify raw_text is included
    if 'raw_text' in df.columns:
        print("✓ raw_text column included")
        print(f"Average text length: {df['text_length'].mean():.0f} characters")
    else:
        print("✗ raw_text column missing!")
        return 1
    
    # Export results
    print("\nExporting results...")
    from src.phase1_parsing.fed_parser import export_to_formats
    export_to_formats(df, config['paths']['processed_dir'])
    
    print(f"\nDone! Files saved to {config['paths']['processed_dir']}")
    print("\nYou can now re-run the knowledge base notebook to build embeddings with the text content.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())