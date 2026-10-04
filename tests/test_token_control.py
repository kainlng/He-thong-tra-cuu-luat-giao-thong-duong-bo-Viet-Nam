import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.ingestion.legal_parser import HierarchicalLegalParser
from src.ingestion.hierarchical_chunker import HierarchicalLegalChunker
from src.storage.indexer import LegalTokenizer, process_chunks_token_control

FIXTURES_DIR = BASE_DIR / "tests" / "fixtures"


def test_token_control_on_long_point():
    parser = HierarchicalLegalParser()
    chunker = HierarchicalLegalChunker()
    tokenizer = LegalTokenizer()

    content = (FIXTURES_DIR / "long_point.md").read_text(encoding="utf-8")
    tree = parser.parse_markdown(content, "long_point.md")
    raw_chunks = chunker.create_chunks(tree)

    # Find the very long point chunk (Point a of Điều 15)
    long_pts = [c for c in raw_chunks if c["chunk_type"] == "point" and c["id"].endswith("_a")]
    assert len(long_pts) == 1, "Expected Point a chunk"
    orig_long_pt = long_pts[0]

    # Verify that before subchunking, final tokens exceed target threshold or hard max
    final_tokens = tokenizer.count_final_embedding_tokens(orig_long_pt["content"])
    print(f"Original Point a final tokens: {final_tokens}")
    assert final_tokens > 150, f"Expected long text, got {final_tokens}"

    # Test process_chunks_token_control with a test hard_max to force subchunking if needed
    # Using hard_max_tokens=150 to verify subchunking behavior
    processed_chunks = process_chunks_token_control(
        raw_chunks,
        tokenizer=tokenizer,
        hard_max_tokens=150,
        target_tokens=120,
    )

    # Find subchunks of original point
    subchunks = [c for c in processed_chunks if c["chunk_type"] == "subchunk" and c["source_chunk_id"] == orig_long_pt["id"]]
    assert len(subchunks) >= 2, f"Expected at least 2 subchunks, got {len(subchunks)}"

    # Check Rule 3: subchunk properties
    for idx, sub in enumerate(subchunks):
        assert sub["source_chunk_id"] == orig_long_pt["id"], "source_chunk_id must match original chunk"
        assert sub["parent_chunk_id"] == orig_long_pt["id"], "parent_chunk_id must match original chunk"
        assert sub["subchunk_index"] == idx, f"Expected subchunk_index={idx}, got {sub['subchunk_index']}"
        assert sub["is_embeddable"] is True, "subchunk must be embeddable"
        assert sub["token_count"] <= 150, f"Subchunk token count ({sub['token_count']}) exceeds hard max 150"

    # Check that original chunk has is_embeddable=False
    updated_orig = [c for c in processed_chunks if c["id"] == orig_long_pt["id"]][0]
    assert updated_orig["is_embeddable"] is False, "Original chunk must be marked non-embeddable when subchunked"

    print("✅ test_token_control_on_long_point PASSED")


if __name__ == "__main__":
    test_token_control_on_long_point()
    print("\n🎉 TOKEN CONTROL TESTS PASSED 100%!")
