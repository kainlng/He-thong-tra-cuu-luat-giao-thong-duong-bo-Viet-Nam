"""Hệ thống tra cứu Luật Giao thông đường bộ Việt Nam - Hierarchical Legal RAG System.

Modules by Pipeline Stage:
- src.core: Shared configurations and data models
- src.ingestion: Document loading, structure parsing, chunking, and ingestion pipeline
- src.storage: LlamaIndex embeddings, PostgreSQL relational storage, and Qdrant vector database
- src.retrieval: Context expansion, reranker, and hybrid query engine
- src.generation: Legal citation builder and Legal RAG Agent
- src.visualization: Interactive 2D/3D vector space visualizer
"""
