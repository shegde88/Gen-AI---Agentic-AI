# Gen AI & Agentic AI — Course Projects

A collection of apps built using LLMs and agentic AI frameworks.

**Author:** Sharan Hegde

## Projects

| Week | Project | Stack | Description |
|---|---|---|---|
| Week 1 | [Soccer Scout Pro](./Week-01-Soccer-Scout-Pro/) | Streamlit · Python | Data-driven football scouting app for Europe's top 5 leagues |
| Week 2 | [FIFA World Cup 2026 RAG](./Week-02-FIFA-World-Cup-RAG/) | LangChain · LangGraph · Pinecone · Nebius · Cohere | RAG-powered chatbot that answers questions about the 2026 World Cup from 85 Wikipedia sources and 6 CSV datasets. 4-node LangGraph pipeline with hallucination checking, Cohere reranking, and a vibe-coded HTML/JS bonus frontend. 87% keyword coverage on 15-question evaluation. |
| Week 3 | [Soccer Scout Pro — Transfer Scout Agent](./Week-03-Soccer-Scout-Pro/) | Streamlit · LangGraph · Anthropic · Tavily | Extends the Week 1 scouting app with a LangGraph-powered Transfer Scout Agent. A 7-node agentic pipeline (parse brief → search players → score players → fetch news → generate report → human review → save report) with human-in-the-loop interrupt, Tavily-powered transfer news, weighted fit scoring (50% overall / 30% age / 20% value for money), and a full Streamlit UI tab. |
