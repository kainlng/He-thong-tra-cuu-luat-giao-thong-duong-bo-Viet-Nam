"""Data Ingestion and Preprocessing stage: PDF loading, legal structure parsing, hierarchical chunking, and ingestion pipeline."""

from src.ingestion.data_loader import (
    clean_legal_markdown,
    load_llamacloud_processor,
    process_all_pdfs,
)
from src.ingestion.legal_parser import HierarchicalLegalParser, calculate_file_hash
from src.ingestion.hierarchical_chunker import (
    HierarchicalLegalChunker,
    remove_vietnamese_accents,
    extract_legal_topics_and_violations,
)
from src.ingestion.ingest_pipeline import run_ingestion

# Alias for backward compatibility
load_llamaindex_processor = load_llamacloud_processor

__all__ = [
    "clean_legal_markdown",
    "load_llamacloud_processor",
    "load_llamaindex_processor",
    "process_all_pdfs",
    "HierarchicalLegalParser",
    "calculate_file_hash",
    "HierarchicalLegalChunker",
    "remove_vietnamese_accents",
    "extract_legal_topics_and_violations",
    "run_ingestion",
]
