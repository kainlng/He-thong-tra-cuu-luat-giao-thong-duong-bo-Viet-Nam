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
)

from src.core.config import (
    resolve_tessdata_dir,
    XBERG_OCR_STRATEGY,
    XBERG_OCR_LANGUAGE,
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


class XbergProcessor:
    """
    Bộ xử lý bóc tách tài liệu PDF sử dụng framework Xberg kết hợp Tesseract OCR.
    Hỗ trợ auto-detect OCR, trích xuất cấu trúc trang có gắn thẻ <!-- PAGE {page_num} -->,
    và chuẩn hóa tương thích tuyệt đối với parser cấu trúc phân cấp pháp luật.
    """

    def __init__(
        self,
        language: Optional[List[str]] = None,
        ocr_strategy: Optional[str] = None,
        force_ocr: bool = False,
        tessdata_path: Optional[str] = None,
    ):
        self.language = language or XBERG_OCR_LANGUAGE
        self.ocr_strategy = ocr_strategy or XBERG_OCR_STRATEGY
        self.force_ocr = force_ocr

        resolved_tess = resolve_tessdata_dir() if tessdata_path is None else Path(tessdata_path)
        self.tessdata_path = str(resolved_tess.resolve())

    def get_config_dict(self, force_ocr: Optional[bool] = None) -> Dict[str, Any]:
        """Trả về từ điển cấu hình để sinh mã hash config cho IngestionManifest."""
        return {
            "engine": "xberg",
            "ocr_backend": "tesseract",
            "language": self.language,
            "ocr_strategy": self.ocr_strategy,
            "force_ocr": self.force_ocr if force_ocr is None else force_ocr,
            "tessdata_path": self.tessdata_path,
        }

    @staticmethod
    def detect_scanned_pages(pdf_path: Path) -> List[int]:
        """Tự động phát hiện các trang scan dạng ảnh (thiếu text layer) trong PDF bằng pypdf."""
        scanned_pages = []
        try:
            import pypdf
            reader = pypdf.PdfReader(str(pdf_path))
            for idx, page in enumerate(reader.pages):
                text = (page.extract_text() or "").strip()
                # Nếu trang có ít hơn 50 ký tự text và có chứa đối tượng hình ảnh, đánh dấu là trang scan
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

        ocr_config = OcrConfig(
            backend="tesseract",
            language=self.language,
            tessdata_path=self.tessdata_path,
        )

        page_config = PageConfig(
            extract_pages=True,
            insert_page_markers=True,
            marker_format="\n\n<!-- PAGE {page_num} -->\n\n",
        )

        return ExtractionConfig(
            output_format="markdown",
            ocr_strategy=self.ocr_strategy,
            force_ocr=is_force_ocr,
            force_ocr_pages=force_ocr_pages,
            ocr=ocr_config,
            pages=page_config,
            enable_quality_processing=True,
        )

    async def extract_pdf_async(
        self,
        pdf_path: Path,
        force_ocr: Optional[bool] = None,
    ) -> str:
        """Bóc tách PDF bất đồng bộ qua Xberg và trả về nội dung Markdown đã chuẩn hóa.
        Hỗ trợ Hybrid PDF: tự động phát hiện các trang scan ảnh để ép OCR trang đó,
        trong khi các trang text được trích xuất trực tiếp chuẩn xác 100%.
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file PDF: {pdf_path}")

        is_force_ocr = self.force_ocr if force_ocr is None else force_ocr
        target_ocr_pages = None

        if not is_force_ocr:
            scanned_pages = self.detect_scanned_pages(pdf_path)
            if scanned_pages:
                print(f"  [Hybrid PDF] Tự động phát hiện {len(scanned_pages)} trang scan ảnh: {scanned_pages[:10]}...")
                target_ocr_pages = scanned_pages

        config = self.build_extraction_config(force_ocr=is_force_ocr, force_ocr_pages=target_ocr_pages)
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
    ) -> str:
        """Wrapper đồng bộ gọi extract_pdf_async qua asyncio.run()."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
                return loop.run_until_complete(self.extract_pdf_async(pdf_path, force_ocr=force_ocr))
        except Exception:
            pass

        return asyncio.run(self.extract_pdf_async(pdf_path, force_ocr=force_ocr))
