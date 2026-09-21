"""Data Ingestion and Preprocessing stage: PDF loading, legal structure parsing, hierarchical chunking, and ingestion pipeline."""

from src.ingestion.xberg_processor import (
    XbergProcessor,
    normalize_xberg_markdown,
)
from src.ingestion.data_loader import (
    clean_legal_markdown,
    process_pdf_xberg,
    process_all_pdfs,
    load_llamacloud_processor,
    load_xberg_processor,
)
from src.ingestion.legal_parser import HierarchicalLegalParser, calculate_file_hash
from src.ingestion.hierarchical_chunker import (
    HierarchicalLegalChunker,
    remove_vietnamese_accents,
    extract_legal_topics_and_violations,
)
from src.ingestion.ingest_pipeline import run_ingestion

# Alias for backward compatibility
load_llamaindex_processor = load_xberg_processor

__all__ = [
    "XbergProcessor",
    "normalize_xberg_markdown",
    "clean_legal_markdown",
    "process_pdf_xberg",
    "process_all_pdfs",
    "load_llamacloud_processor",
    "load_xberg_processor",
    "load_llamaindex_processor",
    "HierarchicalLegalParser",
    "calculate_file_hash",
    "HierarchicalLegalChunker",
    "remove_vietnamese_accents",
    "extract_legal_topics_and_violations",
    "run_ingestion",
]
