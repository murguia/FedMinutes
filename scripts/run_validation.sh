#!/bin/bash

# Run the Fed Minutes validation script using the virtual environment

# Change to project root directory
cd "$(dirname "$0")/.."

# Activate virtual environment
source .venv/bin/activate

# Run validation with specified mode
MODE=${1:-subset}
NUM_FILES=${2:-10}

echo "Running Fed Minutes Validation..."
echo "Mode: $MODE"
if [ "$MODE" = "subset" ]; then
    echo "Number of files: $NUM_FILES"
fi

python scripts/validate_fed_parser.py --mode $MODE --num-files $NUM_FILES \
    --pdf-dir data/raw/PDFs --txt-dir data/raw/TXTs

# Deactivate virtual environment
deactivate