import os
import sys
import re
import json
import hashlib
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List

from rich import print

# Ensure UTF-8 stdout encoding for Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root directory to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.core.config import RAW_DIR, PROCESSED_DIR
from src.ingestion.xberg_processor import XbergProcessor, normalize_xberg_markdown


MANIFEST_FILE_NAME = ".data_loader_manifest.json"


def calculate_pdf_hash(pdf_path: Path) -> str:
    """Calculate SHA-256 hash of a PDF file to detect content changes."""
    sha256 = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return f"sha256:{sha256.hexdigest()}"


def calculate_config_hash(config_dict: Dict[str, Any]) -> str:
    """Calculate hash of extraction configuration to detect parameter changes (strategy, language, psm...)."""
    raw = json.dumps(config_dict, sort_keys=True, ensure_ascii=False)
    return f"cfg:{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:12]}"


class IngestionManifest:
    """Manages ingestion manifest cache to enable resume and track OCR config changes."""

    def __init__(self, processed_dir: Path = PROCESSED_DIR):
        self.manifest_path = processed_dir / MANIFEST_FILE_NAME
        self.data: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if self.manifest_path.exists():
            try:
                return json.loads(self.manifest_path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def save(self):
        try:
            self.manifest_path.write_text(
                json.dumps(self.data, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
        except Exception as exc:
            print(f"[Manifest Warning] Không thể ghi file manifest: {exc}")

    def get_entry(self, pdf_name: str) -> Optional[Dict[str, Any]]:
        return self.data.get(pdf_name)

    def update_entry(
        self,
        pdf_name: str,
        file_hash: str,
        config_hash: str,
        output_file: str,
        method: str,
        char_count: int,
        ocr_details: Optional[Dict[str, Any]] = None,
    ):
        entry = {
            "file_hash": file_hash,
            "config_hash": config_hash,
            "output_file": output_file,
            "processed_at": datetime.now().isoformat(),
            "method": method,
            "char_count": char_count,
        }
        if ocr_details:
            entry.update(ocr_details)
        self.data[pdf_name] = entry
        self.save()

    def should_skip(
        self,
        pdf_path: Path,
        output_path: Path,
        config_hash: str,
        overwrite: bool = False,
    ) -> bool:
        """Determines if a PDF should be skipped based on content hash and config hash."""
        if overwrite:
            return False

        if not output_path.exists() or output_path.stat().st_size == 0:
            return False

        entry = self.get_entry(pdf_path.name)
        if not entry:
            # File exists on disk but not in manifest -> reprocess to record config hash
            return False

        current_file_hash = calculate_pdf_hash(pdf_path)
        # Skip only if BOTH PDF content and OCR extraction config match
        if entry.get("file_hash") == current_file_hash and entry.get("config_hash") == config_hash:
            return True

        return False


def clean_legal_markdown(text: str) -> str:
    """Clean extracted Vietnamese legal gazette markdown while preserving page markers."""
    if not text:
        return ""

    # 1. Remove markdown code block markers
    text = re.sub(r"(?im)^\s*```(?:markdown)?\s*$", "", text)
    # 2. Remove LlamaCloud page tags if any exist
    text = re.sub(r"(?im)^\s*</?page_(?:header|footer)[^>]*>\s*\n?", "", text)
    # 3. Remove CÔNG BÁO headers and variations (leaving <!-- PAGE \d+ --> intact)
    text = re.sub(r"(?im)^\s*#+\s*CÔNG BÁO\s*/[^\n]*\n?", "", text)
    text = re.sub(r"(?im)^\s*CÔNG BÁO\s*/[^\n]*\n?", "", text)
    text = re.sub(r"(?im)^\s*\d+\s+CÔNG BÁO\s*/[^\n]*\n?", "", text)
    text = re.sub(r"(?im)^\s*\d+\s*\n(?=\s*CÔNG BÁO\s*/)", "", text)
    # 4. Remove standalone page numbers (e.g. '12', '- 12 -', 'Trang 12', 'Trang 12/45')
    text = re.sub(r"(?im)^\s*(?:Trang\s+\d+(?:\s*[\/\-]\s*\d+)?|[-–—]\s*\d{1,4}\s*[-–—])\s*$", "", text)
    # 5. Remove legal text header banners
    text = re.sub(r"(?im)^\s*VĂN BẢN QUY PHẠM PHÁP LUẬT\s*\n?", "", text)
    text = re.sub(r"(?im)^\s*\(?Tiếp theo Công báo[^\n]*\)?\s*\n?", "", text)
    text = re.sub(r"(?im)^\s*#+\s*(?:CHỦ TỊCH NƯỚC\s*-\s*QUỐC HỘI|QUỐC HỘI|CÔNG BÁO).*?\n", "", text)
    # 6. Normalize 3+ consecutive newlines to 2 newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def process_pdf_xberg(
    pdf_path: Path,
    processor: Optional[XbergProcessor] = None,
    force_ocr: bool = False,
) -> str:
    """Extract and normalize a single PDF using XbergProcessor."""
    if processor is None:
        processor = XbergProcessor(force_ocr=force_ocr)
    return processor.extract_pdf(pdf_path, force_ocr=force_ocr)


# Backward compatibility wrappers
def load_llamacloud_processor():
    """Backward compatibility alias returning a callable process_pdf(pdf_path)."""
    processor = XbergProcessor()
    return lambda pdf_path: process_pdf_xberg(pdf_path, processor=processor)


load_xberg_processor = load_llamacloud_processor


def process_all_pdfs(
    overwrite: bool = False,
    force_ocr: bool = False,
    raw_dir: Path = RAW_DIR,
    processed_dir: Path = PROCESSED_DIR,
) -> Dict[str, int]:
    """
    Process all PDFs in data/raw using XBerg with Tesseract OCR (vie).
    Saves outputs as *_xberg.md in data/processed and updates manifest with config hash.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    pdf_files = sorted(raw_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"Không tìm thấy file PDF nào trong {raw_dir}")
        return {"total": 0, "processed": 0, "skipped": 0, "failed": 0}

    processor = XbergProcessor(force_ocr=force_ocr)
    manifest = IngestionManifest(processed_dir=processed_dir)
    config_dict = processor.get_config_dict(force_ocr=force_ocr)
    config_hash = calculate_config_hash(config_dict)

    print("=" * 70)
    print(f"BẮT ĐẦU DATALOADER BÓC TÁCH PDF BẰNG XBERG ({len(pdf_files)} tệp tin)")
    print(f"- OCR Strategy: {processor.ocr_strategy} (Tesseract vie.traineddata)")
    print(f"- Force OCR: {'BẬT (Ép OCR toàn bộ trang)' if force_ocr else 'TẮT (Auto-detect trang scan/thiếu text)'}")
    print(f"- Config Hash: {config_hash}")
    print(f"- Resume Mode: {'TẮT (Ép bóc tách lại toàn bộ)' if overwrite else 'BẬT (Bỏ qua file đã xử lý)'}")
    print("=" * 70)

    stats = {
        "total": len(pdf_files),
        "processed": 0,
        "skipped": 0,
        "failed": 0,
    }

    for pdf_path in pdf_files:
        output_path = processed_dir / f"{pdf_path.stem}_xberg.md"
        pdf_hash = calculate_pdf_hash(pdf_path)

        # Check Resume status with both pdf hash and config hash
        if manifest.should_skip(pdf_path, output_path, config_hash=config_hash, overwrite=overwrite):
            print(f"[Resume Skip] '{pdf_path.name}' -> Đã xử lý với cấu hình tương đương ({output_path.name}).")
            stats["skipped"] += 1
            continue

        print(f"\n[Xberg Ingestion] Đang bóc tách: {pdf_path.name}")
        try:
            cleaned_text = processor.extract_pdf(pdf_path, force_ocr=force_ocr)
            output_path.write_text(cleaned_text, encoding="utf-8")

            manifest.update_entry(
                pdf_name=pdf_path.name,
                file_hash=pdf_hash,
                config_hash=config_hash,
                output_file=output_path.name,
                method="xberg_tesseract",
                char_count=len(cleaned_text),
                ocr_details={
                    "ocr_backend": "tesseract",
                    "ocr_strategy": processor.ocr_strategy,
                    "ocr_language": processor.language,
                    "force_ocr": force_ocr,
                },
            )

            stats["processed"] += 1
            print(f"Đã lưu kết quả tại: {output_path} ({len(cleaned_text):,} ký tự)")

        except Exception as exc:
            print(f"[Xberg Error] Lỗi khi bóc tách {pdf_path.name}: {exc}")
            stats["failed"] += 1

    print("\n" + "=" * 70)
    print("TỔNG KẾT BÓC TÁCH TÀI LIỆU XBERG:")
    print(f"- Tổng số file PDF: {stats['total']}")
    print(f"- Đã bỏ qua (Resume - Không thay đổi): {stats['skipped']}")
    print(f"- Đã bóc tách thành công: {stats['processed']}")
    print(f"- Thất bại: {stats['failed']}")
    print("=" * 70)

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vietnamese Legal PDF DataLoader with XBerg & Tesseract OCR")
    parser.add_argument("--force", "--reprocess", action="store_true", help="Force re-processing all PDFs")
    parser.add_argument("--force-ocr", action="store_true", help="Force OCR on all pages using Tesseract")
    args = parser.parse_args()

    process_all_pdfs(overwrite=args.force, force_ocr=args.force_ocr)
