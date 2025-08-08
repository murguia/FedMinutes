# Federal Reserve Minutes Analysis Project

An automated system for parsing and analyzing Federal Reserve meeting minutes from the critical period of 1965-1973, focusing on monetary policy decisions around the Nixon Shock and Bretton Woods collapse.

## 🎯 Project Overview

This project processes over 1,100 Federal Reserve meeting minutes to extract structured insights from a pivotal period in economic history. The system transforms unstructured PDF documents into searchable, analyzable data to understand Fed decision-making during major economic transitions.

## 🏗️ High-Level Architecture

The project follows a 4-phase pipeline design:

```
Phase 1: Data Preparation → Phase 2: Knowledge Base → Phase 3: AI Analysis → Phase 4: Research Questions
     [MOSTLY DONE]              [NEXT STEP]           [FUTURE]            [FUTURE]
```

### Phase 1: Data Preparation ✅ **MOSTLY DONE**
**Status**: Nearly complete with high-quality parsing results

**Capabilities**:
- PDF to structured data extraction
- Attendee identification and role parsing  
- Decision extraction with voting patterns
- Topic categorization and analysis
- Comprehensive validation and quality assessment

**Current Output**:
- ~1,100 meeting records with 95%+ parsing accuracy
- Structured CSV/JSON with attendees, decisions, topics, dates
- Quality validation reports and problem file identification

### Phase 2: Build Searchable Knowledge Base 🔄 **NEXT STEP**  
**Goal**: Create intelligent search and retrieval system

**Planned Components**:
- Vector embeddings of meeting content for semantic search
- ChromaDB or similar vector database integration
- Full-text search capabilities across all documents
- Similarity search for finding related meetings/decisions
- API for programmatic access to the knowledge base

**Success Metrics**:
- Sub-second search response times
- Relevant results for semantic queries
- Comprehensive coverage of all parsed content

### Phase 3: AI-Powered Analysis 🚀 **FUTURE**
**Goal**: Automated insight generation and pattern discovery  

**Planned Features**:
- RAG (Retrieval-Augmented Generation) system for intelligent Q&A
- Trend analysis and pattern recognition
- Automated report generation
- Cross-meeting relationship discovery
- Policy impact analysis

### Phase 4: Specific Research Questions 🎯 **FUTURE**
**Goal**: Answer targeted historical and policy questions

**Research Areas**:
- Nixon Shock decision-making process and Fed response
- Bretton Woods collapse: internal Fed discussions and preparations  
- Voting pattern evolution during economic crises
- Key decision-maker influence and policy advocacy
- International coordination and communication patterns

## 📁 Project Structure

```
FedMinutes/
├── data/
│   ├── raw/
│   │   ├── PDFs/           # Original documents (excluded from git)
│   │   └── TXTs/           # Extracted text (42MB, in git)
│   ├── processed/          # Structured CSV/JSON outputs
│   └── validation/         # Quality reports and analysis
├── src/
│   ├── phase1_parsing/     # ✅ Document parsing (complete)
│   ├── phase2_embedding/   # 🔄 Vector search (next step)  
│   ├── phase4_ai/          # 🚀 AI analysis (future)
│   └── utils/              # Shared configuration and utilities
├── notebooks/
│   ├── 01_exploration.ipynb    # Data exploration and visualization
│   └── 02_validation.ipynb     # Quality assessment
├── scripts/                    # Validation and utility scripts
└── config/                     # System configuration
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Virtual environment (`.venv/` included)

### Quick Start
```bash
# Activate virtual environment
source .venv/bin/activate

# Run the parser (Phase 1)
./run_parser.sh

# Validate results  
./scripts/run_validation.sh

# Explore data
jupyter lab notebooks/
```

## 📊 Current Status & Data Quality

**Phase 1 Results**:
- ✅ 1,100+ documents successfully parsed
- ✅ 95%+ accuracy in data extraction  
- ✅ Comprehensive validation framework
- ✅ High-quality structured output ready for Phase 2

**Key Metrics**:
- Date extraction: >99% success rate
- Average 28 attendees per meeting  
- Average 12 decisions per meeting
- Comprehensive topic categorization
- Financial amounts and voting patterns captured

## 🔄 Next Steps (Phase 2)

**Immediate Priorities**:
1. **Vector Embedding Pipeline**
   - Implement sentence transformers for meeting content
   - Create embeddings for semantic search capabilities

2. **Knowledge Base Infrastructure**  
   - Set up ChromaDB vector database
   - Design efficient indexing and retrieval system

3. **Search Interface**
   - Build query interface for the knowledge base
   - Implement both semantic and keyword search

4. **API Development**
   - Create programmatic access to parsed data
   - Enable integration with analysis tools

**Success Criteria for Phase 2**:
- All meeting content searchable via semantic queries
- Fast retrieval (<1 second) for complex searches  
- Foundation ready for AI-powered analysis (Phase 3)

## 💡 Key Features

**Advanced Parsing Engine**:
- Multi-pattern recognition for complex document structures
- OCR error correction and text normalization
- Structured data models with validation
- Robust error handling and quality assessment

**Comprehensive Data Extraction**:
- Meeting metadata (dates, types, participants)
- Detailed attendee information with roles and organizations
- Decision tracking with voting patterns and financial amounts  
- Topic analysis and categorization
- Document references and cross-links

**Quality Assurance**:
- Automated validation with detailed reporting
- Problem detection and resolution guidance
- Manual verification tools and spot-checking
- Continuous quality monitoring

## 🎯 Research Applications

This system enables analysis of:
- **Monetary Policy Evolution**: Track Fed decision-making during economic crises
- **Institutional Behavior**: Understand internal discussions and debate patterns  
- **Historical Context**: Connect Fed actions to major economic events
- **Policy Network Analysis**: Map relationships and influence patterns
- **Decision Timing**: Analyze response times and preparation for major policy shifts

## 📈 Future Vision

**Phase 3 & 4 Goals**:
- Intelligent Q&A system for historical research
- Automated insight discovery and report generation  
- Comprehensive analysis of Nixon Shock and Bretton Woods periods
- Academic research platform for economic historians
- Policy analysis tools for understanding institutional decision-making

---

**Current Focus**: Completing Phase 2 to build the searchable knowledge base that will enable advanced AI-powered analysis in subsequent phases.