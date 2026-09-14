import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.ingestion.legal_parser import HierarchicalLegalParser


def test_parser_luat():
    parser = HierarchicalLegalParser()
    sample = """
    ## LUẬT TRẬT TỰ, AN TOÀN GIAO THÔNG ĐƯỜNG BỘ
    Luật số: 36/2024/QH15
    ### Chương I: NHỮNG QUY ĐỊNH CHUNG
    **Điều 1. Phạm vi điều chỉnh**
    Nội dung điều 1...
    """
    info = parser.extract_document_info(sample, "36-2024-qh15.md")
    assert info["document_type"] == "luat", f"Expected 'luat', got {info['document_type']}"
    assert info["document_number"] == "36/2024/QH15", f"Expected '36/2024/QH15', got {info['document_number']}"
    assert info["id"] == "doc_36_2024_qh15", f"Expected 'doc_36_2024_qh15', got {info['id']}"
    assert info["issuing_agency"]["code"] == "QUOC_HOI", f"Expected 'QUOC_HOI', got {info['issuing_agency']['code']}"
    assert "trật tự, an toàn giao thông đường bộ" in info["title"].lower(), f"Unexpected title: {info['title']}"
    print("✅ test_parser_luat PASSED")


def test_parser_nghi_dinh():
    parser = HierarchicalLegalParser()
    sample = """
    # CHÍNH PHỦ
    Số: 168/2024/NĐ-CP
    # NGHỊ ĐỊNH
    **Quy định xử phạt vi phạm hành chính về trật tự, an toàn giao thông trong lĩnh vực giao thông đường bộ**
    ### Chương I: NHỮNG QUY ĐỊNH CHUNG
    **Điều 1. Phạm vi điều chỉnh**
    Nội dung điều 1...
    """
    info = parser.extract_document_info(sample, "168-2024-nd-cp_llamacloud.md")
    assert info["document_type"] == "nghi_dinh", f"Expected 'nghi_dinh', got {info['document_type']}"
    assert info["document_number"] == "168/2024/NĐ-CP", f"Expected '168/2024/NĐ-CP', got {info['document_number']}"
    assert info["id"] == "doc_168_2024_nd_cp", f"Expected 'doc_168_2024_nd_cp', got {info['id']}"
    assert info["issuing_agency"]["code"] == "CHINH_PHU", f"Expected 'CHINH_PHU', got {info['issuing_agency']['code']}"
    assert "xử phạt vi phạm hành chính" in info["title"].lower(), f"Unexpected title: {info['title']}"
    print("✅ test_parser_nghi_dinh PASSED")


def test_parser_thong_tu():
    parser = HierarchicalLegalParser()
    sample = """
    # BỘ CÔNG AN
    Số: 02/2024/TT-BCA
    # THÔNG TƯ
    Quy định về tuần tra, kiểm soát giao thông đường bộ
    **Điều 1. Phạm vi điều chỉnh**
    Nội dung điều 1...
    """
    info = parser.extract_document_info(sample, "02-2024-tt-bca.md")
    assert info["document_type"] == "thong_tu", f"Expected 'thong_tu', got {info['document_type']}"
    assert info["document_number"] == "02/2024/TT-BCA", f"Expected '02/2024/TT-BCA', got {info['document_number']}"
    assert info["id"] == "doc_02_2024_tt_bca", f"Expected 'doc_02_2024_tt_bca', got {info['id']}"
    assert info["issuing_agency"]["code"] == "BCA", f"Expected 'BCA', got {info['issuing_agency']['code']}"
    print("✅ test_parser_thong_tu PASSED")


if __name__ == "__main__":
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    test_parser_luat()
    test_parser_nghi_dinh()
    test_parser_thong_tu()
    print("\n🎉 ALL TESTS PASSED SUCCESSFULLY!")
