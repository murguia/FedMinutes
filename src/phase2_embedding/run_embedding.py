"""Create embeddings and vector database."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from src.phase2_embedding.text_chunker import TextChunker
from src.utils.config import load_config
import logging

logger = logging.getLogger(__name__)

def main():
    config = load_config()
    
    # Load parsed data
    df = pd.read_csv(f"{config['paths']['processed_dir']}/meetings_summary.csv")
    
    # Initialize embedder
    model = SentenceTransformer(config['embedding']['model'])
    
    # Initialize ChromaDB
    client = chromadb.PersistentClient(path=config['paths']['vector_db'])
    collection = client.get_or_create_collection("fed_minutes")
    
    # Chunk and embed documents
    chunker = TextChunker(
        chunk_size=config['embedding']['chunk_size'],
        overlap=config['embedding']['chunk_overlap']
    )
    
    for idx, row in df.iterrows():
        chunks = chunker.chunk_text(row['text'])
        
        for i, chunk in enumerate(chunks):
            embedding = model.encode(chunk)
            
            collection.add(
                embeddings=[embedding.tolist()],
                documents=[chunk],
                metadatas=[{
                    'filename': row['filename'],
                    'date': row['date'],
                    'chunk_index': i
                }],
                ids=[f"{row['filename']}_chunk_{i}"]
            )
        
        if idx % 100 == 0:
            logger.info(f"Processed {idx}/{len(df)} documents")
    
    logger.info("✅ Embeddings created successfully!")

if __name__ == "__main__":
    main()


