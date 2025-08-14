"""Vector embedding pipeline for Fed Minutes knowledge base"""

import os
import json
import pandas as pd
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging
from datetime import datetime

import numpy as np
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass


@dataclass
class DocumentChunk:
    """Represents a chunk of text with metadata"""
    chunk_id: str
    meeting_id: str
    filename: str
    date: Optional[datetime]
    chunk_text: str
    chunk_index: int
    total_chunks: int
    meeting_type: str
    attendees: List[str]
    topics: List[str]
    decisions_summary: str
    page_references: List[int]


class TextChunker:
    """Handles intelligent chunking of Fed Minutes documents"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.logger = logging.getLogger(__name__)
    
    def chunk_by_sentences(self, text: str) -> List[str]:
        """Split text into chunks respecting sentence boundaries"""
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # Add sentence to current chunk
            test_chunk = current_chunk + sentence + ". "
            
            if len(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                # Current chunk is full, start new chunk with overlap
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap from end of previous chunk
                if self.chunk_overlap > 0 and len(current_chunk) > self.chunk_overlap:
                    overlap_text = current_chunk[-self.chunk_overlap:].strip()
                    current_chunk = overlap_text + " " + sentence + ". "
                else:
                    current_chunk = sentence + ". "
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def chunk_by_sections(self, text: str) -> List[Tuple[str, str]]:
        """Chunk text by logical sections (decisions, discussions, etc.)"""
        sections = []
        current_section = ""
        section_type = "general"
        
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Identify section headers
            if any(keyword in line.lower() for keyword in 
                   ['approved', 'denied', 'authorized', 'decided', 'resolved']):
                if current_section:
                    sections.append((section_type, current_section))
                section_type = "decision"
                current_section = line + "\n"
            elif any(keyword in line.lower() for keyword in 
                     ['discussion', 'report', 'presentation', 'review']):
                if current_section:
                    sections.append((section_type, current_section))
                section_type = "discussion"
                current_section = line + "\n"
            elif line.startswith("PRESENT:") or "attendees" in line.lower():
                if current_section:
                    sections.append((section_type, current_section))
                section_type = "attendees"
                current_section = line + "\n"
            else:
                current_section += line + "\n"
        
        # Add final section
        if current_section:
            sections.append((section_type, current_section))
        
        return sections


class VectorEmbeddings:
    """Handles vector embedding generation and management"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.logger = logging.getLogger(__name__)
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            self.model = SentenceTransformer(self.model_name)
            self.logger.info(f"Loaded embedding model: {self.model_name}")
        except Exception as e:
            self.logger.error(f"Failed to load model {self.model_name}: {e}")
            raise
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        try:
            embeddings = self.model.encode(texts, show_progress_bar=True)
            self.logger.info(f"Generated embeddings for {len(texts)} texts")
            return embeddings
        except Exception as e:
            self.logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    def embed_single_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        return self.embed_texts([text])[0]
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings from this model"""
        if not self.model:
            raise RuntimeError("Model not loaded")
        return self.model.get_sentence_embedding_dimension()


class FedMinutesEmbeddingPipeline:
    """Main pipeline for processing Fed Minutes into vector embeddings"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.chunker = TextChunker(
            chunk_size=config['embedding']['chunk_size'],
            chunk_overlap=config['embedding']['chunk_overlap']
        )
        self.embedder = VectorEmbeddings(config['embedding']['model'])
        self.logger = logging.getLogger(__name__)
    
    def process_meetings_dataframe(self, df: pd.DataFrame) -> List[DocumentChunk]:
        """Convert meetings dataframe into document chunks"""
        chunks = []
        
        for idx, meeting in df.iterrows():
            meeting_chunks = self._process_single_meeting(meeting, idx)
            chunks.extend(meeting_chunks)
        
        self.logger.info(f"Created {len(chunks)} chunks from {len(df)} meetings")
        return chunks
    
    def _process_single_meeting(self, meeting: pd.Series, meeting_idx: int) -> List[DocumentChunk]:
        """Process a single meeting into chunks"""
        raw_text = meeting.get('raw_text', '')
        if not raw_text:
            self.logger.warning(f"No raw text for meeting {meeting.get('filename', 'unknown')}")
            return []
        
        # Extract metadata
        attendees = self._extract_attendee_names(meeting.get('attendees', '[]'))
        topics = self._extract_topic_titles(meeting.get('topics', '[]'))
        decisions_summary = self._create_decisions_summary(meeting.get('decisions', '[]'))
        
        # Chunk the text
        text_chunks = self.chunker.chunk_by_sentences(raw_text)
        
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            chunk = DocumentChunk(
                chunk_id=f"{meeting.get('filename', 'unknown')}_{i}",
                meeting_id=meeting.get('filename', 'unknown'),
                filename=meeting.get('filename', 'unknown'),
                date=pd.to_datetime(meeting.get('date')) if meeting.get('date') else None,
                chunk_text=chunk_text,
                chunk_index=i,
                total_chunks=len(text_chunks),
                meeting_type=meeting.get('meeting_type', 'regular'),
                attendees=attendees,
                topics=topics,
                decisions_summary=decisions_summary,
                page_references=[]  # Could be enhanced with page detection
            )
            chunks.append(chunk)
        
        return chunks
    
    def _extract_attendee_names(self, attendees_json: str) -> List[str]:
        """Extract attendee names from JSON string"""
        try:
            attendees = json.loads(attendees_json) if isinstance(attendees_json, str) else attendees_json
            if isinstance(attendees, list):
                return [att.get('name', '') for att in attendees if isinstance(att, dict)]
            return []
        except:
            return []
    
    def _extract_topic_titles(self, topics_json: str) -> List[str]:
        """Extract topic titles from JSON string"""
        try:
            topics = json.loads(topics_json) if isinstance(topics_json, str) else topics_json
            if isinstance(topics, list):
                return [topic.get('title', '') for topic in topics if isinstance(topic, dict)]
            return []
        except:
            return []
    
    def _create_decisions_summary(self, decisions_json: str) -> str:
        """Create a summary of decisions from JSON string"""
        try:
            decisions = json.loads(decisions_json) if isinstance(decisions_json, str) else decisions_json
            if isinstance(decisions, list) and decisions:
                summaries = []
                for dec in decisions:
                    if isinstance(dec, dict):
                        action = dec.get('action', '')
                        subject = dec.get('subject', '')
                        if action and subject:
                            summaries.append(f"{action}: {subject}")
                return "; ".join(summaries)
            return ""
        except:
            return ""
    
    def generate_embeddings_for_chunks(self, chunks: List[DocumentChunk]) -> Tuple[List[DocumentChunk], np.ndarray]:
        """Generate embeddings for all chunks"""
        texts = [chunk.chunk_text for chunk in chunks]
        embeddings = self.embedder.embed_texts(texts)
        
        self.logger.info(f"Generated embeddings: {embeddings.shape}")
        return chunks, embeddings
    
    def save_processed_data(self, chunks: List[DocumentChunk], embeddings: np.ndarray, output_dir: str):
        """Save processed chunks and embeddings to disk"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save chunks metadata
        chunks_data = []
        for chunk in chunks:
            chunks_data.append({
                'chunk_id': chunk.chunk_id,
                'meeting_id': chunk.meeting_id,
                'filename': chunk.filename,
                'date': chunk.date.isoformat() if chunk.date else None,
                'chunk_text': chunk.chunk_text,
                'chunk_index': chunk.chunk_index,
                'total_chunks': chunk.total_chunks,
                'meeting_type': chunk.meeting_type,
                'attendees': chunk.attendees,
                'topics': chunk.topics,
                'decisions_summary': chunk.decisions_summary,
                'page_references': chunk.page_references
            })
        
        chunks_df = pd.DataFrame(chunks_data)
        chunks_df.to_json(os.path.join(output_dir, 'document_chunks.json'), orient='records', indent=2)
        chunks_df.to_csv(os.path.join(output_dir, 'document_chunks.csv'), index=False)
        
        # Save embeddings
        np.save(os.path.join(output_dir, 'embeddings.npy'), embeddings)
        
        # Save metadata
        metadata = {
            'total_chunks': len(chunks),
            'embedding_dimension': embeddings.shape[1],
            'model_name': self.embedder.model_name,
            'chunk_size': self.chunker.chunk_size,
            'chunk_overlap': self.chunker.chunk_overlap,
            'created_at': datetime.now().isoformat()
        }
        
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Saved {len(chunks)} chunks and embeddings to {output_dir}")


def create_embeddings_pipeline(config_or_path=None):
    """Factory function to create embedding pipeline
    
    Args:
        config_or_path: Either a config dict or path to config file
    """
    from src.utils.config import load_config
    
    if config_or_path is None or isinstance(config_or_path, str):
        # Load from path
        config = load_config(config_or_path)
    else:
        # Already a config dict
        config = config_or_path
        
    return FedMinutesEmbeddingPipeline(config)


if __name__ == "__main__":
    # Example usage
    from src.utils.config import load_config
    
    config = load_config()
    pipeline = FedMinutesEmbeddingPipeline(config)
    
    # Load processed meetings data
    processed_dir = Path(config['paths']['processed_dir'])
    meetings_file = processed_dir / 'meetings_full.json'
    
    if meetings_file.exists():
        df = pd.read_json(meetings_file)
        print(f"Loaded {len(df)} meetings")
        
        # Process into chunks
        chunks = pipeline.process_meetings_dataframe(df)
        print(f"Created {len(chunks)} chunks")
        
        # Generate embeddings
        chunks, embeddings = pipeline.generate_embeddings_for_chunks(chunks)
        
        # Save results
        output_dir = processed_dir / 'embeddings'
        pipeline.save_processed_data(chunks, embeddings, str(output_dir))
        print(f"Saved embeddings to {output_dir}")
    else:
        print(f"No meetings data found at {meetings_file}")
        print("Please run Phase 1 parsing first")