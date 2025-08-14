#!/bin/bash
# Setup script for Fed Minutes Analysis Project

echo "Fed Minutes Analysis - Setup Script"
echo "==================================="
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Check if processed data exists
if [ ! -f "data/processed/meetings_full.json" ]; then
    echo ""
    echo "Processed data not found. Generating from TXT files..."
    echo "This will take approximately 5 minutes..."
    python -m src.phase1_parsing.fed_parser
else
    echo "Processed data already exists."
fi

# Check if embeddings exist
if [ ! -d "data/processed/embeddings" ]; then
    echo ""
    echo "Building knowledge base (embeddings and vector DB)..."
    echo "This will take approximately 10 minutes..."
    ./scripts/build_knowledge_base.py
else
    echo "Knowledge base already exists."
fi

echo ""
echo "Setup complete! You can now:"
echo "1. Run the demo notebook: jupyter lab notebooks/03_knowledge_base_demo.ipynb"
echo "2. Explore the data: jupyter lab notebooks/01_exploration.ipynb"
echo "3. Check validation: jupyter lab notebooks/02_validation.ipynb"
echo ""
echo "Happy researching!"