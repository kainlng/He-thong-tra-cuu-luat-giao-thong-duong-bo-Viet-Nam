import os
import sys
import re
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv():
        return False

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def clean_legal_markdown(text: str) -> str:
    """Strip running gazette headers, footers, and redundant page titles from extracted markdown."""
    # 1. Remove CÔNG BÁO running headers
    text = re.sub(r"(?im)^\s*CÔNG BÁO/[^\n]*\n?", "", text)
    # 2. Remove VĂN BẢN QUY PHẠM PHÁP LUẬT
    text = re.sub(r"(?im)^\s*VĂN BẢN QUY PHẠM PHÁP LUẬT\s*\n?", "", text)
    # 3. Remove (Tiếp theo Công báo...)
    text = re.sub(r"(?im)^\s*\(?Tiếp theo Công báo[^\n]*\)?\s*\n?", "", text)
    # 4. Remove CHỦ TỊCH NƯỚC - QUỐC HỘI header banners
    text = re.sub(r"(?im)^\s*#+\s*(?:CHỦ TỊCH NƯỚC\s*-\s*QUỐC HỘI|QUỐC HỘI|CÔNG BÁO).*?\n", "", text)
    # 5. Normalize multiple consecutive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_llamaindex_processor():
    """Load LlamaIndex / LlamaParse PDF processor configured to ignore headers/footers."""
    llama_key = os.getenv("LLAMA_PARSE_API_KEY")
    if llama_key:
        try:
            from llama_parse import LlamaParse
            print("Khởi tạo bộ bóc tách tài liệu LlamaParse Cloud...")
            parser = LlamaParse(
                api_key=llama_key,
                result_type="markdown",
                language="vi",
                verbose=True,
                parsing_instruction=(
                    "Ignore all page headers, page footers, running headers, and gazette metadata "
                    "(such as 'CÔNG BÁO/Số...', 'VĂN BẢN QUY PHẠM PHÁP LUẬT', 'CHỦ TỊCH NƯỚC - QUỐC HỘI', "
                    "'Tiếp theo Công báo số...'). Do not include them in the markdown output. "
                    "Only extract the actual legal chapters, articles, clauses, and points."
                ),
            )

            def process_pdf_cloud(pdf_path: Path) -> str:
                print(f"Gửi tài liệu {pdf_path.name} đến LlamaParse Cloud (Bỏ qua Header/Footer)...")
                documents = parser.load_data(str(pdf_path))
                raw_text = "\n\n".join([doc.text for doc in documents])
                cleaned = clean_legal_markdown(raw_text)
                return cleaned

            return process_pdf_cloud
        except Exception as exc:
            print(f"Cảnh báo LlamaParse: {exc}. Chuyển sang LlamaIndex PDFReader...")

    # Fallback to local LlamaIndex PDFReader
    try:
        from llama_index.readers.file import PDFReader
        print("Khởi tạo bộ đọc LlamaIndex PDFReader...")
        reader = PDFReader()

        def process_pdf_local(pdf_path: Path) -> str:
            documents = reader.load_data(pdf_path)
            raw_text = "\n\n".join([doc.text for doc in documents])
            cleaned = clean_legal_markdown(raw_text)
            return cleaned

        return process_pdf_local
    except Exception as exc:
        raise ImportError(f"Không thể khởi tạo LlamaIndex PDF Reader: {exc}")


def process_all_pdfs():
    """Process all PDFs in data/raw using LlamaIndex and save to data/processed."""
    pdf_files = sorted(RAW_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"Không tìm thấy file PDF nào trong {RAW_DIR}")
        return

    print(f"Tìm thấy {len(pdf_files)} file PDF trong {RAW_DIR}")
    try:
        process_pdf = load_llamaindex_processor()
    except Exception as exc:
        print(f"Không thể khởi động LlamaIndex processor: {exc}")
        return

    for pdf_path in pdf_files:
        output_path = PROCESSED_DIR / f"{pdf_path.stem}_llamaindex.md"
        print(f"\n[LlamaIndex] Đang xử lý: {pdf_path.name}")
        try:
            text = process_pdf(pdf_path)
            output_path.write_text(text, encoding="utf-8")
            print(f"-> Đã lưu kết quả tại: {output_path}")
        except Exception as exc:
            print(f"Lỗi khi xử lý {pdf_path.name}: {exc}")

    print("\nHoàn tất xử lý tài liệu bằng LlamaIndex!")


if __name__ == "__main__":
    process_all_pdfs()
