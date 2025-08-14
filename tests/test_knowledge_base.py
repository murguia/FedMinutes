"""Tests for Phase 2 Knowledge Base functionality"""

import pytest
import tempfile
import shutil
import json
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

from src.phase2_knowledge_base import (
    DocumentChunk, TextChunker, VectorEmbeddings, FedMinutesEmbeddingPipeline,
    FedMinutesVectorDB, FedMinutesSemanticSearch, QueryBuilder
)


class TestDocumentChunk:
    """Test DocumentChunk data class"""
    
    def test_chunk_creation(self):
        chunk = DocumentChunk(
            chunk_id="test_chunk_1",
            meeting_id="meeting_123",
            filename="test.txt",
            date=datetime(1967, 1, 15),
            chunk_text="This is a test chunk",
            chunk_index=0,
            total_chunks=5,
            meeting_type="regular",
            attendees=["Mr. Smith", "Mr. Jones"],
            topics=["Monetary Policy"],
            decisions_summary="Approved budget",
            page_references=[1, 2]
        )
        
        assert chunk.chunk_id == "test_chunk_1"
        assert chunk.meeting_id == "meeting_123"
        assert chunk.filename == "test.txt"
        assert chunk.date.year == 1967
        assert chunk.chunk_text == "This is a test chunk"
        assert chunk.attendees == ["Mr. Smith", "Mr. Jones"]


class TestTextChunker:
    """Test text chunking functionality"""
    
    def test_chunker_initialization(self):
        chunker = TextChunker(chunk_size=500, chunk_overlap=100)
        assert chunker.chunk_size == 500
        assert chunker.chunk_overlap == 100
    
    def test_chunk_by_sentences_simple(self):
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        text = "This is sentence one. This is sentence two. This is sentence three."
        
        chunks = chunker.chunk_by_sentences(text)
        assert len(chunks) > 0
        assert all(len(chunk) <= 60 for chunk in chunks)  # Allow some margin for overlap
    
    def test_chunk_by_sentences_empty(self):
        chunker = TextChunker()
        text = ""
        
        chunks = chunker.chunk_by_sentences(text)
        assert chunks == []
    
    def test_chunk_by_sections(self):
        chunker = TextChunker()
        text = """
        PRESENT: Mr. Smith, Chairman
        The Board approved the application.
        Discussion of monetary policy followed.
        """
        
        sections = chunker.chunk_by_sections(text)
        assert len(sections) > 0
        assert all(isinstance(section, tuple) and len(section) == 2 for section in sections)


class TestVectorEmbeddings:
    """Test vector embedding functionality"""
    
    @pytest.fixture
    def embedder(self):
        # Use a small model for testing
        return VectorEmbeddings("all-MiniLM-L6-v2")
    
    def test_embedder_initialization(self, embedder):
        assert embedder.model is not None
        assert embedder.model_name == "all-MiniLM-L6-v2"
    
    def test_embed_single_text(self, embedder):
        text = "This is a test sentence for embedding."
        embedding = embedder.embed_single_text(text)
        
        assert isinstance(embedding, np.ndarray)
        assert len(embedding.shape) == 1  # 1D array
        assert embedding.shape[0] > 0  # Has dimensions
    
    def test_embed_multiple_texts(self, embedder):
        texts = [
            "First test sentence.",
            "Second test sentence.",
            "Third test sentence."
        ]
        
        embeddings = embedder.embed_texts(texts)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 3  # Three embeddings
        assert embeddings.shape[1] > 0  # Has dimensions
    
    def test_embedding_dimension(self, embedder):
        dimension = embedder.get_embedding_dimension()
        assert isinstance(dimension, int)
        assert dimension > 0


class TestFedMinutesEmbeddingPipeline:
    """Test the embedding pipeline"""
    
    @pytest.fixture
    def config(self):
        return {
            'embedding': {
                'model': 'all-MiniLM-L6-v2',
                'chunk_size': 200,
                'chunk_overlap': 50
            }
        }
    
    @pytest.fixture
    def sample_dataframe(self):
        return pd.DataFrame({
            'filename': ['test1.txt', 'test2.txt'],
            'date': ['2023-01-01', '2023-01-02'],
            'meeting_type': ['regular', 'special'],
            'attendees': ['[{"name": "Mr. Smith"}]', '[{"name": "Mr. Jones"}]'],
            'topics': ['[{"title": "Policy"}]', '[{"title": "Banking"}]'],
            'decisions': ['[{"action": "approved", "subject": "Budget"}]', '[]'],
            'raw_text': [
                'This is the first meeting text. It discusses policy matters.',
                'This is the second meeting text. It covers banking issues.'
            ]
        })
    
    def test_pipeline_initialization(self, config):
        pipeline = FedMinutesEmbeddingPipeline(config)
        assert pipeline.config == config
        assert pipeline.chunker.chunk_size == 200
        assert pipeline.embedder.model_name == 'all-MiniLM-L6-v2'
    
    def test_process_meetings_dataframe(self, config, sample_dataframe):
        pipeline = FedMinutesEmbeddingPipeline(config)
        chunks = pipeline.process_meetings_dataframe(sample_dataframe)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
        assert all(chunk.filename in ['test1.txt', 'test2.txt'] for chunk in chunks)
    
    def test_generate_embeddings(self, config, sample_dataframe):
        pipeline = FedMinutesEmbeddingPipeline(config)
        chunks = pipeline.process_meetings_dataframe(sample_dataframe)
        
        chunks, embeddings = pipeline.generate_embeddings_for_chunks(chunks)
        
        assert len(chunks) == embeddings.shape[0]
        assert embeddings.shape[1] > 0
        assert isinstance(embeddings, np.ndarray)


class TestFedMinutesVectorDB:
    """Test ChromaDB integration"""
    
    @pytest.fixture
    def temp_db_path(self):
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_chunks(self):
        return [
            DocumentChunk(
                chunk_id="chunk_1",
                meeting_id="meeting_1",
                filename="test1.txt",
                date=datetime(1967, 1, 15),
                chunk_text="Discussion of monetary policy and interest rates.",
                chunk_index=0,
                total_chunks=1,
                meeting_type="regular",
                attendees=["Mr. Smith"],
                topics=["Monetary Policy"],
                decisions_summary="Approved rate change",
                page_references=[1]
            ),
            DocumentChunk(
                chunk_id="chunk_2",
                meeting_id="meeting_2",
                filename="test2.txt",
                date=datetime(1967, 2, 15),
                chunk_text="Banking regulation and supervision matters.",
                chunk_index=0,
                total_chunks=1,
                meeting_type="special",
                attendees=["Mr. Jones"],
                topics=["Banking Regulation"],
                decisions_summary="Approved new rules",
                page_references=[1]
            )
        ]
    
    def test_vector_db_initialization(self, temp_db_path):
        db = FedMinutesVectorDB(temp_db_path)
        assert db.db_path == Path(temp_db_path)
        assert db.client is not None
    
    def test_create_collection(self, temp_db_path):
        db = FedMinutesVectorDB(temp_db_path)
        collection = db.create_collection()
        
        assert collection is not None
        assert collection.name == db.collection_name
    
    def test_add_document_chunks(self, temp_db_path, sample_chunks):
        db = FedMinutesVectorDB(temp_db_path)
        db.create_collection()
        
        db.add_document_chunks(sample_chunks)
        
        # Verify chunks were added
        stats = db.get_collection_stats()
        assert stats['total_chunks'] == 2
    
    def test_semantic_search(self, temp_db_path, sample_chunks):
        db = FedMinutesVectorDB(temp_db_path)
        db.create_collection()
        db.add_document_chunks(sample_chunks)
        
        results = db.semantic_search("monetary policy", n_results=5)
        
        assert 'query' in results
        assert 'results' in results
        assert len(results['results']) <= 5
        assert results['query'] == "monetary policy"
    
    def test_filter_by_date_range(self, temp_db_path, sample_chunks):
        db = FedMinutesVectorDB(temp_db_path)
        db.create_collection()
        db.add_document_chunks(sample_chunks)
        
        results = db.filter_by_date_range("1967-01-01", "1967-01-31")
        
        assert len(results) == 1  # Only one meeting in January
        assert results[0]['date'] == "1967-01-15T00:00:00"
    
    def test_hybrid_search(self, temp_db_path, sample_chunks):
        db = FedMinutesVectorDB(temp_db_path)
        db.create_collection()
        db.add_document_chunks(sample_chunks)
        
        results = db.hybrid_search(
            query="banking",
            date_range=("1967-02-01", "1967-02-28"),
            meeting_types=["special"]
        )
        
        assert len(results['results']) == 1
        assert "banking" in results['results'][0]['chunk_text'].lower()


class TestFedMinutesSemanticSearch:
    """Test semantic search interface"""
    
    @pytest.fixture
    def temp_db_path(self):
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def populated_search(self, temp_db_path):
        # Create and populate database
        db = FedMinutesVectorDB(temp_db_path)
        db.create_collection()
        
        chunks = [
            DocumentChunk(
                chunk_id="chunk_1",
                meeting_id="meeting_1",
                filename="test1.txt",
                date=datetime(1968, 1, 15),
                chunk_text="The Federal Reserve discussed monetary policy and interest rate changes.",
                chunk_index=0,
                total_chunks=1,
                meeting_type="regular",
                attendees=["Mr. Martin", "Mr. Robertson"],
                topics=["Monetary Policy"],
                decisions_summary="Approved rate increase",
                page_references=[1]
            ),
            DocumentChunk(
                chunk_id="chunk_2", 
                meeting_id="meeting_2",
                filename="test2.txt",
                date=datetime(1968, 6, 15),
                chunk_text="Banking supervision and regulation of financial institutions was reviewed.",
                chunk_index=0,
                total_chunks=1,
                meeting_type="regular",
                attendees=["Mr. Martin", "Mr. Mitchell"],
                topics=["Banking Regulation"],
                decisions_summary="Approved new guidelines",
                page_references=[1]
            )
        ]
        
        db.add_document_chunks(chunks)
        return FedMinutesSemanticSearch(db)
    
    def test_basic_search(self, populated_search):
        results = populated_search.search("monetary policy", max_results=5)
        
        assert 'query' in results
        assert 'results' in results
        assert 'search_params' in results
        assert results['query'] == "monetary policy"
        assert len(results['results']) <= 5
    
    def test_search_by_topic(self, populated_search):
        results = populated_search.search_by_topic("monetary_policy", max_results=3)
        
        assert results['total_results'] >= 0
        assert 'results' in results
    
    def test_search_decisions(self, populated_search):
        results = populated_search.search_decisions("approved", max_results=3)
        
        assert results['total_results'] >= 0
        assert 'results' in results
    
    def test_temporal_analysis(self, populated_search):
        results = populated_search.temporal_analysis("policy", 1968, 1968)
        
        assert 'query' in results
        assert 'temporal_data' in results
        assert 'analysis_summary' in results
        assert '1968' in results['temporal_data']


class TestQueryBuilder:
    """Test query builder utilities"""
    
    def test_build_date_query(self):
        result = QueryBuilder.build_date_query("2023-01-01", "2023-12-31")
        assert result == ("2023-01-01", "2023-12-31")
    
    def test_build_participant_query(self):
        result = QueryBuilder.build_participant_query(["Mr. Smith", "Mr. Jones"])
        assert '"Mr. Smith"' in result
        assert '"Mr. Jones"' in result
    
    def test_build_topic_query(self):
        result = QueryBuilder.build_topic_query(["policy", "banking"])
        assert "policy OR banking" == result
    
    def test_parse_natural_language_date(self):
        # Test year only
        result = QueryBuilder.parse_natural_language_date("1967")
        assert result == "1967-01-01"
        
        # Test month/year
        result = QueryBuilder.parse_natural_language_date("3/1967")
        assert result == "1967-03-01"
        
        # Test month name and year
        result = QueryBuilder.parse_natural_language_date("January 1967")
        assert result == "1967-01-01"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])