"""Phase 3: AI-Powered Analysis Module

This module provides AI-powered analysis capabilities for Fed Minutes:
- LLM integration for intelligent Q&A
- RAG (Retrieval-Augmented Generation) pipeline
- Automated insight generation
- Research report creation
"""

from .llm_client import (
    BaseLLMClient,
    OpenAIClient,
    AnthropicClient,
    OllamaClient,
    MockLLMClient,
    create_llm_client,
    LLMResponse,
    SYSTEM_PROMPTS
)

from .rag_pipeline import (
    FedMinutesRAG,
    RAGContext,
    RAGResponse,
    create_rag_pipeline
)

from .insight_generator import (
    FedMinutesInsightGenerator,
    Insight,
    InsightReport,
    create_insight_generator
)

__all__ = [
    # LLM Client
    'BaseLLMClient',
    'OpenAIClient',
    'AnthropicClient',
    'OllamaClient',
    'MockLLMClient',
    'create_llm_client',
    'LLMResponse',
    'SYSTEM_PROMPTS',
    
    # RAG Pipeline
    'FedMinutesRAG',
    'RAGContext',
    'RAGResponse',
    'create_rag_pipeline',
    
    # Insight Generator
    'FedMinutesInsightGenerator',
    'Insight',
    'InsightReport',
    'create_insight_generator'
]
