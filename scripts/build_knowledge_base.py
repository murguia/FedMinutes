#!/usr/bin/env python3
"""Script to build the Fed Minutes knowledge base"""

import sys
import os
import argparse
from pathlib import Path
import logging
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.utils.config import load_config
from src.phase2_knowledge_base import (
    create_embeddings_pipeline,
    create_vector_db,
    create_search_interface
)


def setup_logging(log_level: str = "INFO"):
    """Set up logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def build_embeddings(config: dict, force_rebuild: bool = False):
    """Build vector embeddings from parsed meetings data"""
    logger = logging.getLogger(__name__)
    
    # Check for existing processed data
    processed_dir = Path(config['paths']['processed_dir'])
    meetings_file = processed_dir / 'meetings_full.json'
    
    if not meetings_file.exists():
        logger.error(f"No meetings data found at {meetings_file}")
        logger.error("Please run Phase 1 parsing first using: python -m src.phase1_parsing.fed_parser")
        return False
    
    # Check if embeddings already exist
    embeddings_dir = processed_dir / 'embeddings'
    embeddings_file = embeddings_dir / 'embeddings.npy'
    
    if embeddings_file.exists() and not force_rebuild:
        logger.info(f"Embeddings already exist at {embeddings_dir}")
        logger.info("Use --force-rebuild to regenerate")
        return True
    
    logger.info("Building vector embeddings...")
    
    # Load meetings data
    df = pd.read_json(meetings_file)
    logger.info(f"Loaded {len(df)} meetings")
    
    # Create embedding pipeline
    pipeline = create_embeddings_pipeline(config)
    
    # Process meetings into chunks
    chunks = pipeline.process_meetings_dataframe(df)
    logger.info(f"Created {len(chunks)} document chunks")
    
    # Generate embeddings
    chunks, embeddings = pipeline.generate_embeddings_for_chunks(chunks)
    
    # Save processed data
    pipeline.save_processed_data(chunks, embeddings, str(embeddings_dir))
    logger.info(f"Saved embeddings to {embeddings_dir}")
    
    return True


def build_vector_database(config: dict, force_rebuild: bool = False):
    """Build ChromaDB vector database"""
    logger = logging.getLogger(__name__)
    
    # Check if embeddings exist
    processed_dir = Path(config['paths']['processed_dir'])
    embeddings_dir = processed_dir / 'embeddings'
    chunks_file = embeddings_dir / 'document_chunks.json'
    
    if not chunks_file.exists():
        logger.error(f"No document chunks found at {chunks_file}")
        logger.error("Please build embeddings first")
        return False
    
    logger.info("Building vector database...")
    
    # Create vector database
    vector_db = create_vector_db(config, reset=force_rebuild)
    
    # Load chunks
    import json
    from datetime import datetime
    from src.phase2_knowledge_base.vector_embeddings import DocumentChunk
    
    with open(chunks_file, 'r') as f:
        chunks_data = json.load(f)
    
    # Convert to DocumentChunk objects
    chunks = []
    for chunk_data in chunks_data:
        chunk = DocumentChunk(
            chunk_id=chunk_data['chunk_id'],
            meeting_id=chunk_data['meeting_id'],
            filename=chunk_data['filename'],
            date=datetime.fromisoformat(chunk_data['date']) if chunk_data['date'] else None,
            chunk_text=chunk_data['chunk_text'],
            chunk_index=chunk_data['chunk_index'],
            total_chunks=chunk_data['total_chunks'],
            meeting_type=chunk_data['meeting_type'],
            attendees=chunk_data['attendees'],
            topics=chunk_data['topics'],
            decisions_summary=chunk_data['decisions_summary'],
            page_references=chunk_data['page_references']
        )
        chunks.append(chunk)
    
    logger.info(f"Loaded {len(chunks)} chunks")
    
    # Add to database
    vector_db.add_document_chunks(chunks)
    
    # Get stats
    stats = vector_db.get_collection_stats()
    logger.info(f"Database stats: {stats}")
    
    logger.info("Vector database built successfully!")
    return True


def test_knowledge_base(config: dict):
    """Test the knowledge base with sample queries"""
    logger = logging.getLogger(__name__)
    
    logger.info("Testing knowledge base...")
    
    try:
        # Create search interface
        search = create_search_interface(config)
        
        # Test queries
        test_queries = [
            "interest rates and monetary policy",
            "bank regulation and supervision",
            "economic conditions and inflation",
            "international finance and foreign exchange"
        ]
        
        for query in test_queries:
            logger.info(f"Testing query: '{query}'")
            results = search.search(query, max_results=3)
            logger.info(f"  Found {results['total_results']} results")
            
            if results['results']:
                top_result = results['results'][0]
                logger.info(f"  Top result: {top_result['filename']} (score: {top_result.get('similarity_score', 'N/A')})")
        
        logger.info("Knowledge base test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Knowledge base test failed: {e}")
        return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Build Fed Minutes knowledge base")
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--force-rebuild", action="store_true", help="Force rebuild of existing data")
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip embedding generation")
    parser.add_argument("--skip-database", action="store_true", help="Skip database creation")
    parser.add_argument("--skip-test", action="store_true", help="Skip testing")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting knowledge base build...")
    
    # Load configuration
    try:
        config = load_config(args.config)
        logger.info(f"Loaded configuration")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1
    
    success = True
    
    # Build embeddings
    if not args.skip_embeddings:
        if not build_embeddings(config, args.force_rebuild):
            logger.error("Failed to build embeddings")
            success = False
    
    # Build vector database
    if success and not args.skip_database:
        if not build_vector_database(config, args.force_rebuild):
            logger.error("Failed to build vector database")
            success = False
    
    # Test knowledge base
    if success and not args.skip_test:
        if not test_knowledge_base(config):
            logger.error("Knowledge base test failed")
            success = False
    
    if success:
        logger.info("Knowledge base build completed successfully!")
        logger.info("")
        logger.info("You can now use the knowledge base for semantic search:")
        logger.info("  from src.phase2_knowledge_base import create_search_interface")
        logger.info("  from src.utils.config import load_config")
        logger.info("  config = load_config()")
        logger.info("  search = create_search_interface(config)")
        logger.info("  results = search.search('your query here')")
        return 0
    else:
        logger.error("Knowledge base build failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())