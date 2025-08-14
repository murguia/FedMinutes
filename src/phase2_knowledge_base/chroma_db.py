"""ChromaDB integration for Fed Minutes knowledge base"""

import os
import json
import uuid
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from .vector_embeddings import DocumentChunk, VectorEmbeddings


class FedMinutesVectorDB:
    """ChromaDB-based vector database for Fed Minutes"""
    
    def __init__(self, db_path: str, collection_name: str = "fed_minutes", embedding_model: str = "all-MiniLM-L6-v2"):
        self.db_path = Path(db_path)
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.logger = logging.getLogger(__name__)
        
        # Initialize ChromaDB client
        self.client = self._initialize_client()
        self.collection = None
        self.embedder = VectorEmbeddings(embedding_model)
    
    def _initialize_client(self):
        """Initialize ChromaDB client"""
        try:
            # Ensure directory exists
            self.db_path.mkdir(parents=True, exist_ok=True)
            
            # Create client with persistent storage
            client = chromadb.PersistentClient(
                path=str(self.db_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            self.logger.info(f"Initialized ChromaDB client at {self.db_path}")
            return client
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ChromaDB client: {e}")
            raise
    
    def create_collection(self, reset_if_exists: bool = False):
        """Create or get the Fed Minutes collection"""
        try:
            # Delete existing collection if reset requested
            if reset_if_exists:
                try:
                    self.client.delete_collection(name=self.collection_name)
                    self.logger.info(f"Deleted existing collection: {self.collection_name}")
                except ValueError:
                    pass  # Collection doesn't exist
            
            # Create collection with sentence transformer embedding function
            embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.embedding_model
            )
            
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=embedding_function,
                metadata={"description": "Fed Minutes document chunks with semantic search"}
            )
            
            self.logger.info(f"Created/retrieved collection: {self.collection_name}")
            return self.collection
            
        except Exception as e:
            self.logger.error(f"Failed to create collection: {e}")
            raise
    
    def add_document_chunks(self, chunks: List[DocumentChunk], batch_size: int = 100):
        """Add document chunks to the vector database"""
        if not self.collection:
            raise RuntimeError("Collection not initialized. Call create_collection() first.")
        
        total_chunks = len(chunks)
        self.logger.info(f"Adding {total_chunks} chunks to database in batches of {batch_size}")
        
        # Process in batches
        for i in range(0, total_chunks, batch_size):
            batch_chunks = chunks[i:i + batch_size]
            self._add_chunk_batch(batch_chunks)
            self.logger.info(f"Added batch {i//batch_size + 1}/{(total_chunks-1)//batch_size + 1}")
        
        self.logger.info(f"Successfully added all {total_chunks} chunks")
    
    def _add_chunk_batch(self, chunks: List[DocumentChunk]):
        """Add a batch of chunks to the database"""
        documents = []
        metadatas = []
        ids = []
        
        for chunk in chunks:
            documents.append(chunk.chunk_text)
            ids.append(chunk.chunk_id)
            
            metadata = {
                "meeting_id": chunk.meeting_id,
                "filename": chunk.filename,
                "date": chunk.date.isoformat() if chunk.date else None,
                "chunk_index": chunk.chunk_index,
                "total_chunks": chunk.total_chunks,
                "meeting_type": chunk.meeting_type,
                "attendees": json.dumps(chunk.attendees),
                "topics": json.dumps(chunk.topics),
                "decisions_summary": chunk.decisions_summary,
                "page_references": json.dumps(chunk.page_references)
            }
            metadatas.append(metadata)
        
        try:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
        except Exception as e:
            self.logger.error(f"Failed to add batch: {e}")
            raise
    
    def semantic_search(self, 
                       query: str, 
                       n_results: int = 10,
                       where: Optional[Dict] = None,
                       include_distances: bool = True) -> Dict[str, Any]:
        """Perform semantic search on the knowledge base"""
        if not self.collection:
            raise RuntimeError("Collection not initialized")
        
        try:
            # Prepare include list
            include = ["documents", "metadatas"]
            if include_distances:
                include.append("distances")
            
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where,
                include=include
            )
            
            # Format results
            formatted_results = self._format_search_results(results, query)
            
            self.logger.info(f"Search for '{query}' returned {len(formatted_results['results'])} results")
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            raise
    
    def _format_search_results(self, raw_results: Dict, query: str) -> Dict[str, Any]:
        """Format raw ChromaDB results into a more usable structure"""
        results = []
        
        documents = raw_results.get('documents', [[]])[0]
        metadatas = raw_results.get('metadatas', [[]])[0]
        distances = raw_results.get('distances', [[]])[0] if 'distances' in raw_results else None
        
        for i, doc in enumerate(documents):
            metadata = metadatas[i] if i < len(metadatas) else {}
            
            result = {
                "chunk_text": doc,
                "similarity_score": 1 - distances[i] if distances and i < len(distances) else None,
                "filename": metadata.get("filename"),
                "date": metadata.get("date"),
                "meeting_type": metadata.get("meeting_type"),
                "chunk_index": metadata.get("chunk_index"),
                "attendees": json.loads(metadata.get("attendees", "[]")),
                "topics": json.loads(metadata.get("topics", "[]")),
                "decisions_summary": metadata.get("decisions_summary"),
                "meeting_id": metadata.get("meeting_id")
            }
            results.append(result)
        
        return {
            "query": query,
            "total_results": len(results),
            "results": results
        }
    
    def filter_by_date_range(self, start_date: str, end_date: str, n_results: int = 100) -> List[Dict]:
        """Get all documents within a date range"""
        if not self.collection:
            raise RuntimeError("Collection not initialized")
        
        # Get all documents and filter client-side
        # ChromaDB doesn't support date range queries on string fields
        all_results = self.collection.get(
            limit=n_results * 3,  # Get more to account for filtering
            include=["documents", "metadatas"]
        )
        
        # Filter by date range
        filtered_docs = []
        filtered_metadatas = []
        
        documents = all_results.get('documents', [])
        metadatas = all_results.get('metadatas', [])
        
        for i, metadata in enumerate(metadatas):
            result_date = metadata.get('date')
            if result_date and start_date <= result_date <= end_date:
                filtered_docs.append(documents[i] if i < len(documents) else "")
                filtered_metadatas.append(metadata)
                
                if len(filtered_docs) >= n_results:
                    break
        
        # Format results
        formatted_results = {
            'documents': filtered_docs,
            'metadatas': filtered_metadatas
        }
        
        return self._format_get_results(formatted_results)
    
    def filter_by_meeting_type(self, meeting_type: str, n_results: int = 100) -> List[Dict]:
        """Get documents by meeting type"""
        if not self.collection:
            raise RuntimeError("Collection not initialized")
        
        results = self.collection.get(
            where={"meeting_type": meeting_type},
            limit=n_results,
            include=["documents", "metadatas"]
        )
        
        return self._format_get_results(results)
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        if not self.collection:
            raise RuntimeError("Collection not initialized")
        
        count = self.collection.count()
        
        # Sample some documents to get date range
        sample_results = self.collection.get(
            limit=min(1000, count),
            include=["metadatas"]
        )
        
        dates = []
        meeting_types = []
        
        for metadata in sample_results.get('metadatas', []):
            if metadata.get('date'):
                dates.append(metadata['date'])
            if metadata.get('meeting_type'):
                meeting_types.append(metadata['meeting_type'])
        
        stats = {
            "total_chunks": count,
            "date_range": {
                "earliest": min(dates) if dates else None,
                "latest": max(dates) if dates else None
            },
            "meeting_types": list(set(meeting_types)),
            "collection_name": self.collection_name,
            "embedding_model": self.embedding_model
        }
        
        return stats
    
    def _format_get_results(self, raw_results: Dict) -> List[Dict]:
        """Format results from get() operations"""
        results = []
        
        documents = raw_results.get('documents', [])
        metadatas = raw_results.get('metadatas', [])
        
        for i, doc in enumerate(documents):
            metadata = metadatas[i] if i < len(metadatas) else {}
            
            result = {
                "chunk_text": doc,
                "filename": metadata.get("filename"),
                "date": metadata.get("date"),
                "meeting_type": metadata.get("meeting_type"),
                "chunk_index": metadata.get("chunk_index"),
                "attendees": json.loads(metadata.get("attendees", "[]")),
                "topics": json.loads(metadata.get("topics", "[]")),
                "decisions_summary": metadata.get("decisions_summary"),
                "meeting_id": metadata.get("meeting_id")
            }
            results.append(result)
        
        return results
    
    def hybrid_search(self, 
                     query: str,
                     date_range: Optional[Tuple[str, str]] = None,
                     meeting_types: Optional[List[str]] = None,
                     n_results: int = 10) -> Dict[str, Any]:
        """Perform hybrid search with semantic similarity and metadata filtering"""
        
        # Build where clause for meeting types only
        # Date filtering will be done client-side due to ChromaDB limitations
        where_clause = None
        if meeting_types:
            where_clause = {"meeting_type": {"$in": meeting_types}}
        
        # Get more results initially to account for date filtering
        search_limit = n_results * 3 if date_range else n_results
        
        # Perform semantic search
        results = self.semantic_search(
            query=query,
            n_results=search_limit,
            where=where_clause,
            include_distances=True
        )
        
        # Apply date filtering client-side
        if date_range:
            start_date, end_date = date_range
            filtered_results = []
            
            for result in results['results']:
                result_date = result.get('date')
                if result_date and start_date <= result_date <= end_date:
                    filtered_results.append(result)
                    
                    # Stop when we have enough results
                    if len(filtered_results) >= n_results:
                        break
            
            # Update results
            results['results'] = filtered_results
            results['total_results'] = len(filtered_results)
        
        return results


def create_vector_db(config_or_path=None, reset: bool = False) -> FedMinutesVectorDB:
    """Factory function to create and initialize vector database
    
    Args:
        config_or_path: Either a config dict or path to config file
        reset: Whether to reset existing collection
    """
    from src.utils.config import load_config
    
    if config_or_path is None or isinstance(config_or_path, str):
        # Load from path
        config = load_config(config_or_path)
    else:
        # Already a config dict
        config = config_or_path
    
    db_path = config['paths']['vector_db']
    embedding_model = config['embedding']['model']
    
    vector_db = FedMinutesVectorDB(
        db_path=db_path,
        embedding_model=embedding_model
    )
    
    vector_db.create_collection(reset_if_exists=reset)
    return vector_db


if __name__ == "__main__":
    # Example usage
    from src.utils.config import load_config
    from .vector_embeddings import create_embeddings_pipeline
    
    config = load_config()
    
    # Create vector database
    vector_db = create_vector_db(config, reset=True)
    
    # Load processed chunks
    processed_dir = Path(config['paths']['processed_dir']) / 'embeddings'
    chunks_file = processed_dir / 'document_chunks.json'
    
    if chunks_file.exists():
        # Load chunks
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
        
        print(f"Loaded {len(chunks)} chunks")
        
        # Add to database
        vector_db.add_document_chunks(chunks)
        
        # Test search
        results = vector_db.semantic_search("interest rates and monetary policy", n_results=5)
        print(f"Found {results['total_results']} results for test query")
        
        # Show stats
        stats = vector_db.get_collection_stats()
        print(f"Database stats: {stats}")
        
    else:
        print(f"No chunks found at {chunks_file}")
        print("Please run the embedding pipeline first")