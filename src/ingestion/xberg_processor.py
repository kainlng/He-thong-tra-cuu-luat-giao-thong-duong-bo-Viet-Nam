import re
import sys
import asyncio
import unicodedata
from pathlib import Path
from typing import Optional, List, Dict, Any

from rich import print
from xberg import (
    extract,
    ExtractInput,
    ExtractionConfig,
    PageConfig,
    OcrConfig,
    SecurityLimits,
)

from src.core.config import (
    resolve_tessdata_dir,
    XBERG_OCR_STRATEGY,
    XBERG_OCR_MIN_CONFIDENCE,
    XBERG_OCR_LANGUAGE,
    XBERG_MAX_CONTENT_SIZE,
)


def normalize_xberg_markdown(text: str) -> str:
    """
    Chuẩn hóa Markdown sinh ra từ Xberg để tương thích hoàn toàn với HierarchicalLegalParser:
    1. Đảm bảo Unicode NFC (tiếng Việt có dấu chuẩn).
    2. Bảo tồn các thẻ marker trang <!-- PAGE {n} -->.
    3. Tách rời các tiêu đề Chương và Điều bị dính liền trên 1 dòng.
    4. Gỡ bỏ cú pháp list của Xberg (- 1\\. -> 1., - a) -> a)) để parser nhận diện đúng Khoản và Điểm.
    5. Dọn dẹp header/footer Công báo và số trang rời.
    """
    if not text:
        return ""

    # 1. Chuẩn hóa Unicode NFC
    text = unicodedata.normalize("NFC", text)

    # 2. Tách Chương và Điều nếu bị Xberg gom chung vào một dòng / block in đậm:
    # Ví dụ: **Chương I NHỮNG QUY ĐỊNH CHUNG Điều 1. Phạm vi điều chỉnh**
    # -> **Chương I NHỮNG QUY ĐỊNH CHUNG**\n\n**Điều 1. Phạm vi điều chỉnh**
    def split_chapter_article(m):
        prefix = m.group(1) or ""
        chap = m.group(2).strip()
        art = m.group(3).strip()
        suffix = m.group(4) or ""
        if prefix.startswith("**") and suffix.endswith("**"):
            return f"**{chap}**\n\n**{art}**"
        return f"{chap}\n\n{art}"

    text = re.sub(
        r"(\*{0,2})(Chương\s+[IVXLCDM\d]+[^\n]*?)\s+(Điều\s+\d+[A-Za-z]?[\.:\s\-][^\n]*?)(\*{0,2})(?=\n|$)",
        split_chapter_article,
        text,
        flags=re.IGNORECASE,
    )

    # Đảm bảo trước mỗi "Điều \d+" luôn có ít nhất 2 dấu xuống dòng nếu không phải đầu văn bản
    text = re.sub(r"([^\n])\n(\*{0,2}Điều\s+\d+)", r"\1\n\n\2", text, flags=re.IGNORECASE)

    # 3. Gỡ bỏ cú pháp list markdown (- 1\. hoặc - 1.) về dạng chuẩn pháp luật (1.)
    # Ví dụ: "- 1\\. Nội dung" -> "1. Nội dung"
    text = re.sub(r"(?m)^\s*[-*•]\s*(\d+)[\.\\]+\s*", r"\1. ", text)
    text = re.sub(r"(?m)^\s*(\d+)\\\.\s*", r"\1. ", text)

    # 4. Gỡ bỏ cú pháp list markdown cho Điểm (- a) hoặc - a\.) về dạng chuẩn (a))
    # Ví dụ: "- a) Nội dung" -> "a) Nội dung"
    text = re.sub(r"(?m)^\s*[-*•]\s*([a-đA-Đ])[\.\)]\s*", r"\1) ", text)
    text = re.sub(r"(?m)^\s*([a-đA-Đ])\\\.\s*", r"\1) ", text)

    # 5. Dọn dẹp header/footer Công báo và số trang tạp chí
    # Lưu ý: Giữ nguyên thẻ <!-- PAGE \d+ -->
    text = re.sub(r"(?im)^\s*#+\s*CÔNG BÁO\s*/[^\n]*\n?", "", text)
    text = re.sub(r"(?im)^\s*CÔNG BÁO\s*/[^\n]*\n?", "", text)
    text = re.sub(r"(?im)^\s*\d+\s+CÔNG BÁO\s*/[^\n]*\n?", "", text)
    text = re.sub(r"(?im)^\s*VĂN BẢN QUY PHẠM PHÁP LUẬT\s*\n?", "", text)
    text = re.sub(r"(?im)^\s*\(?Tiếp theo Công báo[^\n]*\)?\s*\n?", "", text)
    text = re.sub(r"(?im)^\s*#+\s*(?:CHỦ TỊCH NƯỚC\s*-\s*QUỐC HỘI|QUỐC HỘI|CÔNG BÁO).*?\n", "", text)
    text = re.sub(r"(?im)^\s*(?:Trang\s+\d+(?:\s*[\/\-]\s*\d+)?|[-–—]\s*\d{1,4}\s*[-–—])\s*$", "", text)

    # 6. Chuẩn hóa khoảng trống và dòng trống
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def resolve_ocr_strategy(strategy: Any, min_confidence: float = 0.5) -> Any:
    """
    Chuẩn hóa ocr_strategy sang định dạng native mà Rust binding của Xberg hỗ trợ.
    Ưu tiên trực tiếp native API `_rust.OcrStrategy.scanned_pages(min_confidence=...)`
    của phiên bản Xberg đang cài đặt, sau đó fallback sang dict `{"mode": "scanned_pages", ...}`.
    """
    if strategy == "scanned_pages" or strategy == {"mode": "scanned_pages"}:
        try:
            import xberg._xberg as _rust
            if hasattr(_rust, "OcrStrategy") and hasattr(_rust.OcrStrategy, "scanned_pages") and callable(_rust.OcrStrategy.scanned_pages):
                return _rust.OcrStrategy.scanned_pages(min_confidence=min_confidence)
        except Exception:
            pass
        return {"mode": "scanned_pages", "min_confidence": min_confidence}
    return strategy


class XbergProcessor:
    """
    Bộ xử lý bóc tách tài liệu PDF sử dụng framework Xberg kết hợp Tesseract OCR.
    Xberg sử dụng scanned_pages strategy để tự đánh giá các trang có đặc điểm giống scan
    và OCR có chọn lọc; các trang còn lại được xử lý bằng native extraction.
    Bảo tồn cấu trúc phân trang <!-- PAGE {page_num} --> và chuẩn hóa tương thích cho Legal Parser.
    """

    def __init__(
        self,
        language: Optional[List[str]] = None,
        ocr_strategy: Optional[str] = None,
        min_confidence: Optional[float] = None,
        force_ocr: bool = False,
        tessdata_path: Optional[str] = None,
        max_content_size: Optional[int] = None,
    ):
        self.language = language or XBERG_OCR_LANGUAGE
        self.ocr_strategy = ocr_strategy or XBERG_OCR_STRATEGY
        self.min_confidence = min_confidence if min_confidence is not None else XBERG_OCR_MIN_CONFIDENCE
        self.force_ocr = force_ocr
        self.max_content_size = max_content_size if max_content_size is not None else XBERG_MAX_CONTENT_SIZE

        resolved_tess = resolve_tessdata_dir() if tessdata_path is None else Path(tessdata_path)
        self.tessdata_path = str(resolved_tess.resolve())

    def get_config_dict(self, force_ocr: Optional[bool] = None) -> Dict[str, Any]:
        """Trả về từ điển cấu hình để sinh mã hash config cho IngestionManifest."""
        return {
            "extraction_backend": "xberg",
            "ocr_backend": "tesseract",
            "language": self.language,
            "ocr_strategy": self.ocr_strategy,
            "min_confidence": self.min_confidence,
            "page_scan_detection": "xberg_scanned_pages",
            "force_ocr": self.force_ocr if force_ocr is None else force_ocr,
            "max_content_size": self.max_content_size,
            "tessdata_path": self.tessdata_path,
        }

    @staticmethod
    def diagnose_scanned_pages(pdf_path: Path) -> List[int]:
        """
        Diagnostic utility: Phát hiện các trang nghi ngờ là scan ảnh dựa trên heuristic pypdf.
        Chỉ dùng phục vụ benchmark/so sánh chẩn đoán, không nằm trên luồng xử lý chính.
        """
        scanned_pages = []
        try:
            import pypdf
            reader = pypdf.PdfReader(str(pdf_path))
            for idx, page in enumerate(reader.pages):
                text = (page.extract_text() or "").strip()
                if len(text) < 50 and len(page.images) > 0:
                    scanned_pages.append(idx + 1)
        except Exception:
            pass
        return scanned_pages

    def build_extraction_config(
        self,
        force_ocr: Optional[bool] = None,
        force_ocr_pages: Optional[List[int]] = None,
    ) -> ExtractionConfig:
        """Khởi tạo đối tượng ExtractionConfig của Xberg với các tham số tối ưu cho Legal RAG."""
        is_force_ocr = self.force_ocr if force_ocr is None else force_ocr

        # Security limits: Tăng giới hạn uncompressed image memory lên 2GB (tránh huỷ OCR trên PDF scan nhiều trang)
        sec_limits = SecurityLimits(
            max_content_size=self.max_content_size,
            max_pages=None,
        )

        ocr_config = OcrConfig(
            backend="tesseract",
            language=self.language,
            tessdata_path=self.tessdata_path,
            security_limits=sec_limits,
        )

        page_config = PageConfig(
            extract_pages=True,
            insert_page_markers=True,
            marker_format="\n\n<!-- PAGE {page_num} -->\n\n",
        )

        resolved_strategy = resolve_ocr_strategy(self.ocr_strategy, self.min_confidence)

        return ExtractionConfig(
            output_format="markdown",
            ocr_strategy=resolved_strategy,
            force_ocr=is_force_ocr,
            force_ocr_pages=force_ocr_pages,
            ocr=ocr_config,
            pages=page_config,
            security_limits=sec_limits,
            enable_quality_processing=True,
        )

    async def extract_pdf_async(
        self,
        pdf_path: Path,
        force_ocr: Optional[bool] = None,
        force_ocr_pages: Optional[List[int]] = None,
    ) -> str:
        """
        Bóc tách PDF bất đồng bộ qua Xberg và trả về nội dung Markdown đã chuẩn hóa.
        Xberg sử dụng scanned_pages strategy để tự đánh giá các trang có đặc điểm giống scan
        và OCR có chọn lọc; các trang còn lại được xử lý bằng native extraction.
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file PDF: {pdf_path}")

        is_force_ocr = self.force_ocr if force_ocr is None else force_ocr

        # Xberg tự quản lý phân loại trang dựa trên scanned_pages (force_ocr_pages chỉ dùng khi manual debug override)
        config = self.build_extraction_config(
            force_ocr=is_force_ocr,
            force_ocr_pages=force_ocr_pages,
        )
        extract_input = ExtractInput(kind="uri", uri=str(pdf_path.resolve()))

        result = await extract(extract_input, config=config)
        if not result.results:
            raise RuntimeError(f"Xberg không trích xuất được nội dung nào từ {pdf_path.name}")

        raw_markdown = result.results[0].content or ""
        normalized_markdown = normalize_xberg_markdown(raw_markdown)
        return normalized_markdown

    def extract_pdf(
        self,
        pdf_path: Path,
        force_ocr: Optional[bool] = None,
        force_ocr_pages: Optional[List[int]] = None,
    ) -> str:
        """Wrapper đồng bộ gọi extract_pdf_async qua asyncio.run()."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(
                self.extract_pdf_async(pdf_path, force_ocr=force_ocr, force_ocr_pages=force_ocr_pages)
            )

        return asyncio.run(
            self.extract_pdf_async(pdf_path, force_ocr=force_ocr, force_ocr_pages=force_ocr_pages)
        )
