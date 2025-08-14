"""Main script to run parsing pipeline."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.phase1_parsing.fed_parser import FedMinutesParser, FedMinutesBatchProcessor
from src.utils.config import load_config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Load configuration
    config = load_config()
    
    # Initialize processor
    processor = FedMinutesBatchProcessor()
    
    # Convert PDFs to TXT if needed
    logger.info("Converting PDFs to TXT...")
    processor.convert_pdfs_to_txt(
        pdf_dir=config['paths']['pdf_dir'],
        txt_dir=config['paths']['txt_dir']
    )
    
    # Process all files
    logger.info("Parsing documents...")
    df = processor.process_directory(
        pdf_dir=config['paths']['pdf_dir'],
        txt_dir=config['paths']['txt_dir'],
        use_txt=True
    )
    
    # Export results
    logger.info("Exporting results...")
    from src.phase1_parsing.fed_parser import export_to_formats
    export_to_formats(df, config['paths']['processed_dir'])
    
    logger.info(f"✅ Parsed {len(df)} documents successfully!")
    return df

if __name__ == "__main__":
    main()


