import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.ingestion.legal_parser import HierarchicalLegalParser
from src.ingestion.hierarchical_chunker import HierarchicalLegalChunker, validate_chunk_tree

FIXTURES_DIR = BASE_DIR / "tests" / "fixtures"


def test_chunker_normal_law():
    parser = HierarchicalLegalParser()
    chunker = HierarchicalLegalChunker()
    
    content = (FIXTURES_DIR / "normal_law.md").read_text(encoding="utf-8")
    tree = parser.parse_markdown(content, "normal_law.md")
    chunks = chunker.create_chunks(tree)

    is_valid, errors = validate_chunk_tree(chunks)
    assert is_valid, f"Validation errors: {errors}"

    articles = [c for c in chunks if c["chunk_type"] == "article"]
    clauses = [c for c in chunks if c["chunk_type"] == "clause"]
    points = [c for c in chunks if c["chunk_type"] == "point"]

    assert len(articles) == 3
    assert len(clauses) >= 6
    assert len(points) == 0

    # Articles must have parent=None and is_embeddable=False
    for a in articles:
        assert a["parent_chunk_id"] is None
        assert a["is_embeddable"] is False

    # Leaf clauses must have parent pointing to Article and is_embeddable=True
    for c in clauses:
        assert c["parent_chunk_id"] is not None
        assert c["is_embeddable"] is True
        assert c["root_article_chunk_id"] == c["parent_chunk_id"]

    print("✅ test_chunker_normal_law PASSED")


def test_chunker_clause_with_points():
    parser = HierarchicalLegalParser()
    chunker = HierarchicalLegalChunker()

    content = (FIXTURES_DIR / "clause_with_points.md").read_text(encoding="utf-8")
    tree = parser.parse_markdown(content, "clause_with_points.md")
    chunks = chunker.create_chunks(tree)

    is_valid, errors = validate_chunk_tree(chunks)
    assert is_valid, f"Validation errors: {errors}"

    clauses_with_pts = [c for c in chunks if c["chunk_type"] == "clause" and "article_168_2024_nd_cp_05" in c["article_id"]]
    # Rule 1 check: When Clause has Points, ClauseChunk.is_embeddable = False
    for cl in clauses_with_pts:
        assert cl["is_embeddable"] is False, f"Clause with points should have is_embeddable=False, got {cl['is_embeddable']}"

    points = [c for c in chunks if c["chunk_type"] == "point"]
    assert len(points) >= 6

    # Rule 1 check: PointChunk parent is ClauseChunk, is_embeddable=True, and carries clause_intro
    for pt in points:
        assert pt["is_embeddable"] is True
        # Parent must be ClauseChunk
        parent_chunk = [c for c in chunks if c["id"] == pt["parent_chunk_id"]][0]
        assert parent_chunk["chunk_type"] == "clause"
        # Root must be ArticleChunk
        assert pt["root_article_chunk_id"] == parent_chunk["parent_chunk_id"]
        # Must carry clause context / intro in pt['content']
        assert "Phạt tiền từ" in pt["content"]

    print("✅ test_chunker_clause_with_points PASSED")


def test_chunker_orphan_point():
    parser = HierarchicalLegalParser()
    chunker = HierarchicalLegalChunker()

    content = (FIXTURES_DIR / "orphan_point.md").read_text(encoding="utf-8")
    tree = parser.parse_markdown(content, "orphan_point.md")
    chunks = chunker.create_chunks(tree)

    is_valid, errors = validate_chunk_tree(chunks)
    assert is_valid, f"Validation errors: {errors}"

    points = [c for c in chunks if c["chunk_type"] == "point"]
    assert len(points) == 4

    for pt in points:
        parent_chunk = [c for c in chunks if c["id"] == pt["parent_chunk_id"]][0]
        assert parent_chunk["chunk_type"] == "clause"
        assert parent_chunk["is_embeddable"] is False
        assert "Cơ quan đăng ký xe" in pt["content"]

    print("✅ test_chunker_orphan_point PASSED")


if __name__ == "__main__":
    test_chunker_normal_law()
    test_chunker_clause_with_points()
    test_chunker_orphan_point()
    print("\n🎉 ALL CHUNKER HIERARCHY TESTS PASSED 100%!")
