---
title: Agentic RAG Multimodal Document Intelligence
emoji: 🤖
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: 1.25.0
app_file: app.py
pinned: false
---

# Agentic RAG & Multimodal Document Intelligence 🤖

Production-grade **Agentic Retrieval-Augmentation Generation (RAG)** engine featuring **LangGraph multi-agent state machines**, hybrid vector retrieval (dense semantic embeddings + BM25 sparse keyword search), and self-correcting RAG evaluation loops.

[![LangGraph](https://img.shields.io/badge/LangGraph-State_Machine-00F2FE?style=for-the-badge&logo=python)](https://github.com/langchain-ai/langgraph)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI_Dashboard-FF4B4B?style=for-the-badge&logo=streamlit)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker)](https://www.docker.com/)

---

## ⚡ Multi-Agent Trajectory Graph Architecture

```
[User Query]
     │
     ▼
[Router Node] ──► [Hybrid Vector & BM25 Retriever Node]
                             │
                             ▼
                 [Synthesizer Node (Citations)]
                             │
                             ▼
             [Self-Correction RAG Evaluation Node] ──► [Verified Output]
```

### Key Capabilities

1. **LangGraph State Machine**: Orchestrates query routing, multi-document retrieval, citation formatting, and self-correcting evaluation loops.
2. **Hybrid Vector Search**: Combines dense semantic similarity and BM25 sparse keyword matching over enterprise PDF/text corpora.
3. **Self-Correction & Faithfulness Evaluation**: Measures context relevance and faithfulness scores, preventing model hallucinations.
4. **Streamlit UI Visualizer**: Interactive dashboard for uploading documents, querying the knowledge base, and tracing multi-agent state transitions in real time.

---

## 🛠️ Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run unit test suite
python test_agent.py

# 3. Launch interactive Streamlit Dashboard
streamlit run app.py
```

---

## 🐳 Docker & Cloud Deployment

```bash
docker build -t agentic-rag-app .
docker run -p 8501:8501 agentic-rag-app
```
