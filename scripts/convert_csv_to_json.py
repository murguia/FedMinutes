#!/usr/bin/env python3
"""Convert meetings_summary.csv to meetings_full.json with raw_text included"""

import sys
import os
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

def main():
    processed_dir = project_root / 'data' / 'processed'
    csv_file = processed_dir / 'meetings_summary.csv'
    json_file = processed_dir / 'meetings_full.json'
    
    if not csv_file.exists():
        print(f"CSV file not found: {csv_file}")
        return 1
    
    print(f"Loading CSV file: {csv_file}")
    df = pd.read_csv(csv_file)
    
    print(f"Loaded {len(df)} meetings")
    print(f"Columns: {list(df.columns)}")
    
    if 'raw_text' in df.columns:
        print("✓ raw_text column found in CSV")
        avg_text_length = df['raw_text'].str.len().mean()
        print(f"Average text length: {avg_text_length:.0f} characters")
        
        # Save as JSON
        print(f"Saving to JSON: {json_file}")
        df.to_json(json_file, orient='records', indent=2)
        print("✓ Conversion complete")
        
        # Verify JSON file
        df_check = pd.read_json(json_file)
        if 'raw_text' in df_check.columns:
            print("✓ raw_text column confirmed in JSON file")
        else:
            print("✗ raw_text column missing in JSON file")
            
    else:
        print("✗ raw_text column missing from CSV")
        return 1
    
    return 0

if __name__ == "__main__":
    try:
        import pandas as pd
        sys.exit(main())
    except ImportError:
        print("pandas not available in this environment")
        print("Please run: pip install pandas")
        sys.exit(1)