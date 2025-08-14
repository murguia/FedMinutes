# Data Directory Structure

This directory contains the Fed Minutes data in various stages of processing.

## Directory Structure

```
data/
├── raw/
│   ├── PDFs/        # Original PDF files (not in git, too large)
│   └── TXTs/        # Extracted text files (in git, ~42MB)
├── processed/       # Parsed and structured data (generated)
├── validation/      # Validation reports (generated)
└── vector_db/       # ChromaDB database (generated)
```

## Important Note on Large Files

Several large files are **excluded from git** to keep the repository size manageable:

- `processed/meetings_full.json` (77MB)
- `processed/meetings_summary.csv` (75MB)
- `processed/all_*.csv` (various sizes)
- `processed/embeddings/` (877MB total)
- `vector_db/` (1.7GB ChromaDB)

## How to Regenerate Missing Files

If you've cloned this repository and need the processed data:

### 1. Generate Parsed Data (Phase 1)
```bash
python -m src.phase1_parsing.fed_parser
```
This creates:
- `processed/meetings_full.json`
- `processed/meetings_summary.csv`
- `processed/all_*.csv` files

### 2. Build Knowledge Base (Phase 2)
```bash
./scripts/build_knowledge_base.py
```
This creates:
- `processed/embeddings/` directory with vector embeddings
- `vector_db/` directory with ChromaDB database

### 3. Verify Everything Works
```bash
jupyter lab notebooks/03_knowledge_base_demo.ipynb
```

## File Sizes Reference

- **Raw TXTs**: ~42MB (included in git)
- **Parsed Data**: ~230MB (generated)
- **Embeddings**: ~877MB (generated)
- **Vector DB**: ~1.7GB (generated)
- **Total**: ~2.8GB (only 42MB in git)

## Notes

- The TXT files are the source data and are included in git
- All other large files can be regenerated from the TXT files
- Generation takes approximately 10-15 minutes on a modern machine