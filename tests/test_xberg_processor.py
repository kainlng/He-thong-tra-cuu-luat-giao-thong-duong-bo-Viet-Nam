import sys
import unittest
from pathlib import Path
import pypdf

# Ensure project root in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.ingestion.xberg_processor import XbergProcessor, resolve_ocr_strategy, normalize_xberg_markdown
from src.ingestion.legal_parser import HierarchicalLegalParser


class TestXbergProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw_dir = BASE_DIR / "data" / "raw"
        cls.test_tmp_dir = BASE_DIR / "tests" / "fixtures" / "tmp_pdf"
        cls.test_tmp_dir.mkdir(parents=True, exist_ok=True)
        
        # Tạo tệp hybrid PDF 2 trang thử nghiệm từ 168-2024-nd-cp.pdf (Trang 1: digital text, Trang 2: scanned image)
        cls.source_pdf = cls.raw_dir / "168-2024-nd-cp.pdf"
        cls.hybrid_pdf = cls.test_tmp_dir / "hybrid_test.pdf"
        if cls.source_pdf.exists():
            reader = pypdf.PdfReader(str(cls.source_pdf))
            writer = pypdf.PdfWriter()
            writer.add_page(reader.pages[0])
            writer.add_page(reader.pages[1])
            with open(cls.hybrid_pdf, "wb") as f:
                writer.write(f)

    @classmethod
    def tearDownClass(cls):
        # Dọn dẹp tệp tạm
        if cls.hybrid_pdf.exists():
            try:
                cls.hybrid_pdf.unlink()
            except Exception:
                pass
        if cls.test_tmp_dir.exists():
            try:
                cls.test_tmp_dir.rmdir()
            except Exception:
                pass

    def test_01_resolve_ocr_strategy(self):
        """Kiểm tra hàm resolve_ocr_strategy chuẩn hóa cho native Rust binding."""
        res_auto = resolve_ocr_strategy("auto")
        self.assertEqual(res_auto, "auto")

        res_scanned = resolve_ocr_strategy("scanned_pages", min_confidence=0.5)
        # Phải là enum _rust.OcrStrategy hoặc dict {"mode": "scanned_pages", "min_confidence": 0.5}
        if isinstance(res_scanned, dict):
            self.assertEqual(res_scanned.get("mode"), "scanned_pages")
            self.assertEqual(res_scanned.get("min_confidence"), 0.5)
        else:
            self.assertIn("scanned_pages", repr(res_scanned))

    def test_02_diagnose_scanned_pages_utility(self):
        """Kiểm tra utility diagnose_scanned_pages nhận diện đúng trang scan qua pypdf."""
        if not self.hybrid_pdf.exists():
            self.skipTest("Hybrid PDF test fixture không khả dụng.")
        scanned_pages = XbergProcessor.diagnose_scanned_pages(self.hybrid_pdf)
        # Trang 1 là digital (chữ ký số), Trang 2 là scan
        self.assertIn(2, scanned_pages)
        self.assertNotIn(1, scanned_pages)

    def test_03_hybrid_pdf_scanned_pages_extraction(self):
        """Kiểm tra Xberg trích xuất hybrid PDF bằng native scanned_pages mà không cần pypdf can thiệp."""
        if not self.hybrid_pdf.exists():
            self.skipTest("Hybrid PDF test fixture không khả dụng.")

        processor = XbergProcessor(ocr_strategy="scanned_pages", min_confidence=0.5, force_ocr=False)
        extracted_text = processor.extract_pdf(self.hybrid_pdf)

        self.assertGreater(len(extracted_text), 100)
        # Phải có marker cho cả 2 trang
        self.assertIn("<!-- PAGE 1 -->", extracted_text)
        self.assertIn("<!-- PAGE 2 -->", extracted_text)
        # Trang 1: Text chữ ký số
        self.assertTrue("THONGTINCHINHPHU" in extracted_text.upper() or "THỜI GIAN KÝ" in extracted_text.upper())
        # Trang 2: Có text được OCR từ Tesseract (ví dụ: 'Cảnh cáo' hoặc 'Phạt tiền' từ Điều 3/4)
        self.assertTrue("CẢNH CÁO" in extracted_text.upper() or "PHẠT TIỀN" in extracted_text.upper())

    def test_04_force_ocr_flag(self):
        """Kiểm tra chế độ force_ocr=True kích hoạt thành công."""
        if not self.hybrid_pdf.exists():
            self.skipTest("Hybrid PDF test fixture không khả dụng.")

        processor = XbergProcessor(force_ocr=True)
        extracted_text = processor.extract_pdf(self.hybrid_pdf)

        self.assertGreater(len(extracted_text), 100)
        self.assertIn("<!-- PAGE 1 -->", extracted_text)
        self.assertIn("<!-- PAGE 2 -->", extracted_text)

    def test_05_config_dict_provenance(self):
        """Kiểm tra get_config_dict và security_limits trả về đúng cấu hình bảo vệ tài nguyên."""
        processor = XbergProcessor(ocr_strategy="scanned_pages", min_confidence=0.5)
        cfg_dict = processor.get_config_dict()
        self.assertEqual(cfg_dict["extraction_backend"], "xberg")
        self.assertEqual(cfg_dict["ocr_backend"], "tesseract")
        self.assertEqual(cfg_dict["ocr_strategy"], "scanned_pages")
        self.assertEqual(cfg_dict["page_scan_detection"], "xberg_scanned_pages")
        self.assertEqual(cfg_dict["max_content_size"], 2 * 1024 * 1024 * 1024)

        ext_cfg = processor.build_extraction_config()
        self.assertIsNotNone(ext_cfg.security_limits)
        self.assertEqual(ext_cfg.security_limits.max_content_size, 2 * 1024 * 1024 * 1024)

    def test_06_markdown_normalization_and_parser_feed(self):
        """Kiểm tra Markdown sau khi normalize được HierarchicalLegalParser nhận diện chuẩn xác."""
        sample_raw = """
<!-- PAGE 1 -->
Người ký: VĂN PHÒNG CHÍNH PHỦ
Số: 99/2025/NĐ-CP
Hà Nội, ngày 10 tháng 01 năm 2025
NGHỊ ĐỊNH
Quy định về bảo đảm an toàn giao thông đường bộ

Chương I NHỮNG QUY ĐỊNH CHUNG Điều 1. Phạm vi điều chỉnh
1. Nghị định này quy định về trật tự giao thông.
2. Các tổ chức, cá nhân có liên quan phải tuân thủ:
- a) Tuân thủ tín hiệu đèn giao thông.
- b) Chấp hành hiệu lệnh của người chỉ huy.
"""
        normalized = normalize_xberg_markdown(sample_raw)
        self.assertIn("<!-- PAGE 1 -->", normalized)
        self.assertIn("Chương I", normalized)
        self.assertIn("Điều 1. Phạm vi điều chỉnh", normalized)
        # Gỡ bỏ markdown list syntax
        self.assertIn("a) Tuân thủ", normalized)

        # Feed vào parser
        parser = HierarchicalLegalParser()
        parsed_tree = parser.parse_markdown(normalized, file_name="99_2025_nd_cp_xberg.md")
        
        self.assertEqual(parsed_tree["document"]["document_number"], "99/2025/NĐ-CP")
        self.assertEqual(len(parsed_tree["articles"]), 1)
        self.assertEqual(len(parsed_tree["clauses"]), 2)
        self.assertEqual(len(parsed_tree["points"]), 2)
        # Provenance check
        src = parsed_tree["document"]["source"]
        self.assertEqual(src["extraction_backend"], "xberg")
        self.assertEqual(src["ocr_backend"], "tesseract")
        self.assertEqual(src["ocr_strategy"], "scanned_pages")

    def test_07_manifest_config_hash_change(self):
        """Kiểm tra thay đổi OCR strategy sang scanned_pages làm thay đổi config_hash (kích hoạt reprocess)."""
        from src.ingestion.data_loader import calculate_config_hash
        old_cfg = {
            "engine": "xberg",
            "ocr_strategy": "auto",
            "force_ocr": False,
        }
        new_cfg = XbergProcessor(ocr_strategy="scanned_pages", min_confidence=0.5).get_config_dict(force_ocr=False)

        old_hash = calculate_config_hash(old_cfg)
        new_hash = calculate_config_hash(new_cfg)

        self.assertNotEqual(old_hash, new_hash)
        self.assertTrue(new_hash.startswith("cfg:"))


if __name__ == "__main__":
    unittest.main()
