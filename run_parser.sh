#!/bin/bash

# Run the Fed Minutes parser using the virtual environment

# Activate virtual environment
source .venv/bin/activate

# Run the parser
echo "Running Fed Minutes Parser..."
python -m src.phase1_parsing.fed_parser

# Deactivate virtual environment
deactivate