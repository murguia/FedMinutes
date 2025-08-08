#!/usr/bin/env python3
"""Run the complete analysis pipeline."""

import subprocess
import sys
from pathlib import Path

def run_phase(phase_name: str, script_path: str):
    """Run a pipeline phase."""
    print(f"\n{'='*50}")
    print(f"Running {phase_name}")
    print('='*50)
    
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"Error in {phase_name}:")
        print(result.stderr)
        sys.exit(1)
    else:
        print(f"✅ {phase_name} completed successfully")

def main():
    base_path = Path(__file__).parent.parent
    
    # Phase 1: Parsing
    run_phase(
        "Phase 1: Document Parsing",
        base_path / "src/phase1_parsing/run_parsing.py"
    )
    
    # Phase 2: Embeddings
    run_phase(
        "Phase 2: Creating Embeddings",
        base_path / "src/phase2_embedding/run_embedding.py"
    )
    
    # Phase 3: Analysis
    run_phase(
        "Phase 3: Statistical Analysis",
        base_path / "src/phase3_analysis/run_analysis.py"
    )
    
    print("\n🎉 Full pipeline completed successfully!")

if __name__ == "__main__":
    main()

