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

from src.core.config import RAW_DIR, PROCESSED_DIR, get_llama_cloud_api_keys


MANIFEST_FILE_NAME = ".data_loader_manifest.json"


def calculate_pdf_hash(pdf_path: Path) -> str:
    """Calculate SHA-256 hash of a PDF file to detect content changes."""
    sha256 = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return f"sha256:{sha256.hexdigest()}"


class IngestionManifest:
    """Manages ingestion manifest cache to enable resume and skip redundant token usage."""

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
        output_file: str,
        method: str,
        char_count: int
    ):
        self.data[pdf_name] = {
            "file_hash": file_hash,
            "output_file": output_file,
            "processed_at": datetime.now().isoformat(),
            "method": method,
            "char_count": char_count,
        }
        self.save()

    def should_skip(self, pdf_path: Path, output_path: Path, overwrite: bool = False) -> bool:
        """Determines if a PDF file should be skipped to save tokens."""
        if overwrite:
            return False

        if not output_path.exists() or output_path.stat().st_size == 0:
            return False

        entry = self.get_entry(pdf_path.name)
        if not entry:
            # Output file exists and non-empty, even if not in manifest -> resume safe
            return True

        current_hash = calculate_pdf_hash(pdf_path)
        if entry.get("file_hash") == current_hash:
            return True

        # Hash mismatch -> PDF was updated -> need re-process
        return False


class LlamaCloudKeyPool:
    """Manages pool of LlamaCloud API keys, supporting key failover rotation when keys run out of tokens."""

    def __init__(self):
        self.keys: List[str] = get_llama_cloud_api_keys()
        self.exhausted_keys: set = set()

    def mask_key(self, key: str) -> str:
        if len(key) <= 8:
            return "*****"
        return f"{key[:4]}...{key[-4:]}"

    def get_working_keys(self) -> List[Tuple[int, str]]:
        """Returns indexed list of available keys: [(1, 'key1'), (2, 'key2'), ...]"""
        return [(idx + 1, key) for idx, key in enumerate(self.keys) if key not in self.exhausted_keys]

    def mark_exhausted(self, key: str):
        self.exhausted_keys.add(key)


def clean_legal_markdown(text: str) -> str:
    """Clean extracted Vietnamese legal gazette markdown."""
    # 1. Remove markdown code block markers
    text = re.sub(r"(?im)^\s*```(?:markdown)?\s*$", "", text)
    # 2. Remove LlamaCloud page tags <page_header>, </page_header>, <page_footer>, </page_footer>
    text = re.sub(r"(?im)^\s*</?page_(?:header|footer)[^>]*>\s*\n?", "", text)
    # 3. Remove CÔNG BÁO headers and variations
    text = re.sub(r"(?im)^\s*CÔNG BÁO\s*/[^\n]*\n?", "", text)
    text = re.sub(r"(?im)^\s*\d+\s*\n(?=\s*CÔNG BÁO\s*/)", "", text)
    text = re.sub(r"(?im)^\s*\d+\s+CÔNG BÁO\s*/[^\n]*\n?", "", text)
    # 4. Remove standalone page numbers (e.g. '12', '- 12 -', 'Trang 12', 'Trang 12/45')
    text = re.sub(r"(?im)^\s*(?:Trang\s+\d+(?:\s*[\/\-]\s*\d+)?|[-–—]?\s*\d{1,4}\s*[-–—]?)\s*$", "", text)
    # 5. Remove legal text header banners
    text = re.sub(r"(?im)^\s*VĂN BẢN QUY PHẠM PHÁP LUẬT\s*\n?", "", text)
    text = re.sub(r"(?im)^\s*\(?Tiếp theo Công báo[^\n]*\)?\s*\n?", "", text)
    text = re.sub(r"(?im)^\s*#+\s*(?:CHỦ TỊCH NƯỚC\s*-\s*QUỐC HỘI|QUỐC HỘI|CÔNG BÁO).*?\n", "", text)
    # 6. Normalize 3+ consecutive newlines to 2 newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()



def process_pdf_cloud(pdf_path: Path, llama_key: str) -> str:
    """Process a PDF via LlamaCloud SDK using a specific API key."""
    from llama_cloud import LlamaCloud
    client = LlamaCloud(api_key=llama_key)
    file_obj = client.files.create(file=pdf_path, purpose="parse")
    print(f"  -> [LlamaCloud API] Đã tạo file_id: {file_obj.id}. Đang phân tích (tier='agentic')...")

    result = client.parsing.parse(
        file_id=file_obj.id,
        tier="agentic",
        version="latest",
        expand=["markdown"],
    )

    pages = result.markdown.pages if (result.markdown and result.markdown.pages) else []
    print(f"  -> [LlamaCloud API] Hoàn tất bóc tách {len(pages)} trang từ {pdf_path.name}.")

    raw_text = "\n\n".join([p.markdown for p in pages if p.markdown])
    return clean_legal_markdown(raw_text)


def process_pdf_local(pdf_path: Path) -> str:
    """Process a PDF via local PDF reader fallback (LlamaIndex PDFReader or PyPDF)."""
    try:
        from llama_index.readers.file import PDFReader
        print(f"  -> [Local Fallback] Đang đọc {pdf_path.name} bằng LlamaIndex PDFReader...")
        reader = PDFReader()
        documents = reader.load_data(pdf_path)
        raw_text = "\n\n".join([doc.text for doc in documents])
        return clean_legal_markdown(raw_text)
    except Exception as exc:
        print(f"  -> [Local Fallback Warning] LlamaIndex PDFReader không khả dụng ({exc}). Thử pypdf...")
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(pdf_path))
            raw_text = "\n\n".join([page.extract_text() or "" for page in reader.pages])
            return clean_legal_markdown(raw_text)
        except Exception as exc_inner:
            raise RuntimeError(f"Không thể bóc tách PDF bằng bất kỳ Local Fallback reader nào: {exc_inner}")


def process_pdf_with_fallback(
    pdf_path: Path,
    force_local: bool = False,
    key_pool: Optional[LlamaCloudKeyPool] = None,
) -> Tuple[str, str]:
    """
    Attempt to process PDF using LlamaCloud API keys from KeyPool (if not force_local).
    Rotates through available LlamaCloud API keys (e.g. LLAMA_CLOUD_API_KEY_1, LLAMA_CLOUD_API_KEY_2...)
    on token exhaustion or API errors.
    If ALL LlamaCloud API keys fail, automatically falls back to Local PDF Reader per-file.
    Returns: (cleaned_text, method_used)
    """
    if key_pool is None:
        key_pool = LlamaCloudKeyPool()

    if not force_local:
        working_keys = key_pool.get_working_keys()
        if working_keys:
            for key_num, llama_key in working_keys:
                masked = key_pool.mask_key(llama_key)
                try:
                    print(f"  -> [LlamaCloud API Pool] Thử bóc tách với Key #{key_num} ({masked})...")
                    cleaned_text = process_pdf_cloud(pdf_path, llama_key)
                    return cleaned_text, f"llamacloud_key_{key_num}"
                except Exception as exc:
                    print(f"[LlamaCloud API Error] Key #{key_num} ({masked}) gặp lỗi: {exc}")
                    key_pool.mark_exhausted(llama_key)
                    remaining = len(key_pool.get_working_keys())
                    if remaining > 0:
                        print(f"[Failover Rotation] Tự động chuyển sang API Key tiếp theo ({remaining} keys còn khả dụng)...")
                    else:
                        print("Tất cả LlamaCloud API Keys trong pool đều đã hết token hoặc gặp lỗi.")

    if force_local:
        print(f"[Local Mode] Ép buộc sử dụng Local PDF Reader cho {pdf_path.name}")
    else:
        print(f"[Local Fallback] Chuyển sang Local PDF Reader cho {pdf_path.name}")

    cleaned_text = process_pdf_local(pdf_path)
    return cleaned_text, "local_fallback"


def load_llamacloud_processor():
    """Backward compatibility wrapper returning a single callable process_pdf(pdf_path)."""
    key_pool = LlamaCloudKeyPool()

    def process_pdf(pdf_path: Path) -> str:
        text, _ = process_pdf_with_fallback(pdf_path, key_pool=key_pool)
        return text

    return process_pdf


def process_all_pdfs(
    overwrite: bool = False,
    force_local: bool = False,
    raw_dir: Path = RAW_DIR,
    processed_dir: Path = PROCESSED_DIR,
) -> Dict[str, int]:
    """
    Process all PDFs in data/raw using Resume (skip existing), Multi-key Rotation, and Local Fallback.
    Saves outputs as *_llamacloud.md in data/processed and updates manifest.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    pdf_files = sorted(raw_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"Không tìm thấy file PDF nào trong {raw_dir}")
        return {"total": 0, "processed": 0, "skipped": 0, "fallback": 0, "failed": 0}

    manifest = IngestionManifest(processed_dir=processed_dir)
    key_pool = LlamaCloudKeyPool()

    print("=" * 70)
    print(f"BẮT ĐẦU DATALOADER BÓC TÁCH PDF ({len(pdf_files)} tệp tin)")
    print(f"- Resume Mode: {'TẮT (Ép bóc tách lại toàn bộ)' if overwrite else 'BẬT (Tiết kiệm token LlamaCloud)'}")
    print(f"- LlamaCloud Key Pool: {len(key_pool.keys)} API Keys khả dụng (LLAMA_CLOUD_API_KEY_1, _2...)")
    print(f"- Parser Strategy: {'Local Only' if force_local else 'Multi-Key LlamaCloud Pool -> Local Reader Fallback'}")
    print("=" * 70)

    stats = {
        "total": len(pdf_files),
        "processed": 0,
        "skipped": 0,
        "fallback": 0,
        "failed": 0,
    }

    for pdf_path in pdf_files:
        output_path = processed_dir / f"{pdf_path.stem}_llamacloud.md"
        pdf_hash = calculate_pdf_hash(pdf_path)

        # Check Resume status
        if manifest.should_skip(pdf_path, output_path, overwrite=overwrite):
            print(f"[Resume Skip] '{pdf_path.name}' -> Đã tồn tại ({output_path.name}). Tiết kiệm LlamaCloud Token!")
            stats["skipped"] += 1
            continue

        print(f"\n[Processing] Đang bóc tách: {pdf_path.name}")
        try:
            text, method = process_pdf_with_fallback(pdf_path, force_local=force_local, key_pool=key_pool)
            output_path.write_text(text, encoding="utf-8")

            manifest.update_entry(
                pdf_name=pdf_path.name,
                file_hash=pdf_hash,
                output_file=output_path.name,
                method=method,
                char_count=len(text),
            )

            if method == "local_fallback":
                stats["fallback"] += 1
            else:
                stats["processed"] += 1

            print(f"Đã lưu kết quả tại: {output_path} ({len(text):,} ký tự, method='{method}')")

        except Exception as exc:
            print(f"Lỗi khi bóc tách {pdf_path.name}: {exc}")
            stats["failed"] += 1

    print("\n" + "=" * 70)
    print("TỔNG KẾT BÓC TÁCH TÀI LIỆU:")
    print(f"- Tổng số file PDF: {stats['total']}")
    print(f"- Đã bỏ qua (Resume - Tiết kiệm Token): {stats['skipped']}")
    print(f"- Đã bóc tách qua LlamaCloud API (Multi-Key Pool): {stats['processed']}")
    print(f"- Đã bóc tách qua Local Fallback: {stats['fallback']}")
    print(f"- Thất bại: {stats['failed']}")
    print("=" * 70)

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vietnamese Legal PDF DataLoader with Resume & Multi-Key Fallback")
    parser.add_argument("--force", "--overwrite", action="store_true", help="Force re-processing all PDFs even if already processed")
    parser.add_argument("--local", "--force-local", action="store_true", help="Force local PDFReader fallback without invoking LlamaCloud API")
    args = parser.parse_args()

    process_all_pdfs(overwrite=args.force, force_local=args.local)
