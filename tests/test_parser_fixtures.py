import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.ingestion.legal_parser import HierarchicalLegalParser, validate_parsed_tree

FIXTURES_DIR = BASE_DIR / "tests" / "fixtures"


def test_normal_law():
    parser = HierarchicalLegalParser()
    content = (FIXTURES_DIR / "normal_law.md").read_text(encoding="utf-8")
    tree = parser.parse_markdown(content, "normal_law.md")
    
    assert tree["document"]["document_number"] == "36/2024/QH15"
    assert tree["document"]["issued_date"] == "2024-06-27"
    assert tree["document"]["first_effective_date"] == "2025-01-01"
    assert len(tree["articles"]) == 3
    assert len(tree["clauses"]) >= 6
    assert all(c["clause_number"] is not None for c in tree["clauses"])
    
    is_valid, errors = validate_parsed_tree(tree)
    assert is_valid, f"Validation errors: {errors}"
    print("✅ test_normal_law PASSED")


def test_clause_with_points():
    parser = HierarchicalLegalParser()
    content = (FIXTURES_DIR / "clause_with_points.md").read_text(encoding="utf-8")
    tree = parser.parse_markdown(content, "clause_with_points.md")

    assert tree["document"]["document_number"] == "168/2024/NĐ-CP"
    assert len(tree["points"]) >= 6
    
    # Check that clauses with points have clause_intro and clause_full_text
    cl1 = [c for c in tree["clauses"] if c["clause_number"] == "1" and "article_168_2024_nd_cp_05" in c["article_id"]][0]
    assert cl1["clause_intro"] is not None
    assert "Phạt tiền từ 400.000 đồng" in cl1["clause_intro"]
    assert "Không chấp hành hiệu lệnh" in cl1["clause_full_text"]
    
    is_valid, errors = validate_parsed_tree(tree)
    assert is_valid, f"Validation errors: {errors}"
    print("✅ test_clause_with_points PASSED")


def test_article_only():
    parser = HierarchicalLegalParser()
    content = (FIXTURES_DIR / "article_only.md").read_text(encoding="utf-8")
    tree = parser.parse_markdown(content, "article_only.md")

    assert len(tree["articles"]) == 3
    # All clauses should be synthetic with clause_number=None
    for cl in tree["clauses"]:
        assert cl["is_synthetic"] is True, "Expected synthetic clause"
        assert cl["clause_number"] is None, f"Expected None clause_number, got {cl['clause_number']}"

    is_valid, errors = validate_parsed_tree(tree)
    assert is_valid, f"Validation errors: {errors}"
    print("✅ test_article_only PASSED")


def test_orphan_point():
    parser = HierarchicalLegalParser()
    content = (FIXTURES_DIR / "orphan_point.md").read_text(encoding="utf-8")
    tree = parser.parse_markdown(content, "orphan_point.md")

    # Article 1 has points a, b, c, d without numbered clauses
    pts = [p for p in tree["points"] if "article_15_2024_tt_bca_01" in p["article_id"]]
    assert len(pts) == 4, f"Expected 4 points, got {len(pts)}"
    labels = [p["point_label"] for p in pts]
    assert labels == ["a", "b", "c", "d"], f"Unexpected labels: {labels}"

    # The parent clause should be synthetic with clause_number=None
    parent_clause = [c for c in tree["clauses"] if c["id"] == pts[0]["clause_id"]][0]
    assert parent_clause["is_synthetic"] is True
    assert parent_clause["clause_number"] is None
    assert "Cơ quan đăng ký xe" in parent_clause["clause_intro"]

    is_valid, errors = validate_parsed_tree(tree)
    assert is_valid, f"Validation errors: {errors}"
    print("✅ test_orphan_point PASSED")


def test_missing_dates():
    parser = HierarchicalLegalParser()
    content = (FIXTURES_DIR / "missing_dates.md").read_text(encoding="utf-8")
    tree = parser.parse_markdown(content, "missing_dates.md")

    # Document has no issued date closing formula and no effective date clause
    # It must NOT fall back to 2024-01-01 or 2025-01-01!
    assert tree["document"]["issued_date"] is None, f"Expected None issued_date, got {tree['document']['issued_date']}"
    assert tree["document"]["first_effective_date"] is None, f"Expected None effective_date, got {tree['document']['first_effective_date']}"

    is_valid, errors = validate_parsed_tree(tree)
    assert is_valid, f"Validation errors: {errors}"
    print("✅ test_missing_dates PASSED")


def test_noisy_ocr():
    parser = HierarchicalLegalParser()
    content = (FIXTURES_DIR / "noisy_ocr.md").read_text(encoding="utf-8")
    tree = parser.parse_markdown(content, "noisy_ocr.md")

    # Clauses with bullets - 1., - 2.
    assert len(tree["clauses"]) >= 3
    # Points with * a), * b), • c)
    assert len(tree["points"]) >= 3

    is_valid, errors = validate_parsed_tree(tree)
    assert is_valid, f"Validation errors: {errors}"
    print("✅ test_noisy_ocr PASSED")


if __name__ == "__main__":
    test_normal_law()
    test_clause_with_points()
    test_article_only()
    test_orphan_point()
    test_missing_dates()
    test_noisy_ocr()
    print("\n🎉 ALL FIXTURE TESTS PASSED 100%!")
