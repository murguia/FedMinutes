# Federal Reserve Minutes Analysis Project

An automated system for parsing and analyzing Federal Reserve meeting minutes from 1967-1973.  These notes were obtained via a [FOIA request by Crisis Notes/Nathan Tankus](https://www.crisesnotes.com/here-are-the-30-000-pages-of-federal-reserve-board-meeting-minutes-i-got-through-foia/).

## Overview

This project processes over 1,100 Federal Reserve meeting minutes to extract structured insights. The system transforms unstructured PDF documents into searchable, analyzable data to understand Fed decision-making during this period.

### 1. Data Preparation

**1. Capabilities**:
1. PDF to structured data extraction
1. Attendee identification and role parsing  
1. Decision extraction with voting patterns
1. Topic categorization and analysis
1. Comprehensive validation and quality assessment

**2. Current Output**:
1. ~1,100 meeting records with 95%+ parsing accuracy
1. Structured CSV/JSON with attendees, decisions, topics, dates
1. Quality validation reports and problem file identification

### 2. Searchable Knowledge Base

**1. Capabilities**:
1. Vector embeddings using sentence-transformers (all-MiniLM-L6-v2)
1. ChromaDB vector database with persistent storage (61,162 chunks)
1. Semantic search capabilities across all document content
1. Intelligent document chunking with metadata preservation
1. Hybrid search combining semantic similarity with metadata filters
1. Temporal analysis and trend detection tools
1. Related meeting discovery and topic analysis
1. Interactive Jupyter notebook for demonstrations

**2. Key Features**:
1. **Semantic search**: Find meetings by meaning, not just keywords
1. **Date range filtering**: Focus on specific time periods (e.g., Nixon Shock era)
1. **Topic-based queries**: Search by monetary policy, banking regulation, etc.
1. **Sub-second performance**: Fast similarity search across 61,162 document chunks
1. **Rich metadata**: Every result includes meeting context, participants, dates
1. **Temporal analysis**: Track how topics evolved over time (1965-1973)
1. **Related content**: Discover meetings with similar themes and discussions

### 3 AI-Powered Analysis

**1. Implemented**:
1. RAG (Retrieval-Augmented Generation) system for intelligent Q&A
1. LLM integration supporting OpenAI and Anthropic APIs
1. Time period analysis and comparative studies
1. Topic evolution tracking across years
1. Automated insight generation and pattern discovery
1. Research-ready Q&A interface with citations
1. Mock LLM support for testing without API keys

**2. Key Capabilities**:
1. **Intelligent Q&A**: Ask natural language questions about Fed policy decisions
1. **Period Summaries**: Generate comprehensive analysis of specific timeframes
1. **Topic Evolution**: Track how Fed thinking evolved on key issues over time
1. **Comparative Analysis**: Compare Fed discussions between different periods
1. **Citation System**: All answers include references to source meetings
1. **Confidence Scoring**: Quality assessment for analytical reliability
1. **Multi-Provider LLM**: Supports OpenAI, Anthropic, Ollama (local), or mock responses

### 4 Report Generation

**1. Features**:
1. **Automated Research Reports**: Generate publication-ready analysis documents
1. **Decision Tree Analysis**: Map Fed decision-making processes and influence networks
1. **Interactive Research Platform**: Web-based interface for historians and researchers
1. **Advanced Pattern Recognition**: Cross-meeting relationship discovery
1. **Export Capabilities**: Academic citation formats and research datasets

**Interesting Topics**:
1. Nixon Shock decision-making process and Fed response analysis
1. Bretton Woods collapse: internal Fed discussions and preparations timeline  
1. Voting pattern evolution and consensus-building during economic crises
1. Key decision-maker influence networks and policy advocacy tracking
1. International coordination and communication pattern studies


### Prerequisites
- Python 3.8+
- Virtual environment

### Quick Start
```bash
# One-command setup (handles everything)
./setup.sh

# Or manual steps:
source .venv/bin/activate
pip install -r requirements.txt
python -m src.phase1_parsing.fed_parser       # Generate parsed data
./scripts/build_knowledge_base.py             # Build embeddings & DB

# Explore the knowledge base
jupyter lab notebooks/03_knowledge_base_demo.ipynb

# Try AI-powered analysis (Phase 3)
jupyter lab notebooks/04_ai_analysis_demo.ipynb
```

**Note**: The repository includes only source text files (42MB). The setup script will generate the parsed data, embeddings, and vector database (~2.8GB total) on first run.

### Local LLM with Ollama (No API Keys Required)

For unlimited, free AI analysis without API costs:

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Download a model (choose one):
ollama pull mistral:7b      # Excellent for analysis, 4GB RAM
ollama pull llama3:8b       # Latest Meta model, 8GB RAM  
ollama pull phi3:mini       # Microsoft, very efficient, 2GB RAM

# Verify installation
ollama list
```

Then edit `config/config.yaml`:
```yaml
llm:
  provider: "ollama"
  model: "mistral:7b"  # or your chosen model
```
**

**Target Deliverables**:
- Complete Nixon Shock decision-making analysis report
- Bretton Woods collapse preparation timeline study
- Fed communication evolution research document
- Interactive dashboard

## Example Results

### **Q: How did the Fed respond to the Nixon Shock in August 1971?**

```
Answer: Based on the Federal Reserve meeting excerpts, the Fed's immediate response 
to Nixon's August 15, 1971 announcement was characterized by careful coordination 
with international partners and domestic market stabilization efforts. The FOMC 
held emergency discussions focusing on:

1. Managing the suspension of dollar-gold convertibility
2. Coordinating with European central banks on exchange rate policies  
3. Ensuring adequate liquidity in domestic markets
4. Monitoring inflation expectations following wage-price controls

The Fed adopted a "wait-and-see" approach while actively engaging in swap 
arrangements with foreign central banks to manage currency pressures.

Confidence: 0.87
Sources: 8 Fed meetings (Aug-Dec 1971)
```

### **Topic Evolution: Inflation Concerns (1969-1973)**

```
The Fed's discussion of inflation evolved dramatically over this period:

1969-1970: Moderate concern, focus on "creeping inflation" around 5%
1971: Escalating worry, discussion of "wage-price spiral" 
1972: Temporary optimism due to controls, but underlying concerns
1973: Alarm over "double-digit inflation" and control failures

Key turning point: August 1971 wage-price freeze initially seen as solution,
but by mid-1972 Fed minutes show growing skepticism about effectiveness.
```

### **Performance Metrics**

- **Semantic Search**: <100ms average query time
- **AI Analysis**: 5-15 seconds with local models (Ollama)
- **Accuracy**: 95%+ parsing accuracy, 0.8+ confidence scores
- **Scale**: 61,162 searchable chunks from 1,124 meetings

## Research Applications

This system enables analysis of:
- **Monetary Policy Evolution**: Track Fed decision-making during economic crises
- **Nixon Shock Impact**: Search Fed discussions around August 1971 policy changes
- **Bretton Woods Collapse**: Analyze Fed preparation and response (1971-1973)
- **Inflation Concerns**: Study evolution of price stability discussions
- **International Coordination**: Find Fed discussions of foreign central bank cooperation
- **Decision Patterns**: Identify voting patterns and consensus-building processes
- **Historical Context**: Connect Fed actions to major economic events with semantic search

Experience the Fed Minutes knowledge base:

```bash
# Clone and setup
git clone [repository-url]
cd FedMinutes
pip install -r requirements.txt

# Launch the interactive knowledge base demo
jupyter lab notebooks/03_knowledge_base_demo.ipynb

# Try the AI-powered analysis (requires API key)
jupyter lab notebooks/04_ai_analysis_demo.ipynb
```

**Sample Queries to Try**:

**Knowledge Base (Phase 2)**:
- "interest rates and monetary policy decisions"
- "inflation concerns around Nixon Shock" (with date filter: 1971-1973)
- "international monetary cooperation bretton woods"
- "wage price controls economic stabilization"

**AI Analysis (Phase 3)**:
- "How did the Fed respond to the Nixon Shock in August 1971?"
- "What were the Fed's main concerns about inflation during 1971-1972?"
- "Compare Fed discussions before and after the Nixon Shock"
- "Analyze the evolution of international monetary policy from 1970-1973"


*Reference:* Data downloaded from https://www.crisesnotes.com/database/.