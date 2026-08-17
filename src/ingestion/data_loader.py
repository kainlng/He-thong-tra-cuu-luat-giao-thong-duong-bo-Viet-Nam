import os
import sys
import re
from pathlib import Path

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

from src.core.config import RAW_DIR, PROCESSED_DIR, LLAMA_CLOUD_API_KEY


def clean_legal_markdown(text: str) -> str:
    """
    Clean extracted Vietnamese legal gazette markdown.
    Removes:
    - CÔNG BÁO running headers
    - page numbers attached to CÔNG BÁO headers
    - VĂN BẢN QUY PHẠM PHÁP LUẬT
    - 'Tiếp theo Công báo...' lines
    - redundant QUỐC HỘI / CHỦ TỊCH NƯỚC banners
    - markdown fences
    - excessive blank lines
    """

    # 1. Remove markdown fences
    text = re.sub(
        r"(?im)^\s*```(?:markdown)?\s*$",
        "",
        text
    )

    # 2. Remove CÔNG BÁO header
    text = re.sub(
        r"(?im)^\s*CÔNG BÁO\s*/[^\n]*\n?",
        "",
        text
    )

    # 3. Remove page number immediately before CÔNG BÁO header
    text = re.sub(
        r"(?im)^\s*\d+\s*\n(?=\s*CÔNG BÁO\s*/)",
        "",
        text
    )

    # 4. Remove page number + CÔNG BÁO when extracted on the same line
    text = re.sub(
        r"(?im)^\s*\d+\s+CÔNG BÁO\s*/[^\n]*\n?",
        "",
        text
    )

    # 5. Remove VĂN BẢN QUY PHẠM PHÁP LUẬT
    text = re.sub(
        r"(?im)^\s*VĂN BẢN QUY PHẠM PHÁP LUẬT\s*\n?",
        "",
        text
    )

    # 6. Remove "(Tiếp theo Công báo...)"
    text = re.sub(
        r"(?im)^\s*\(?Tiếp theo Công báo[^\n]*\)?\s*\n?",
        "",
        text
    )

    # 7. Remove QUỐC HỘI / CHỦ TỊCH NƯỚC banners
    text = re.sub(
        r"(?im)^\s*#+\s*(?:CHỦ TỊCH NƯỚC\s*-\s*QUỐC HỘI|QUỐC HỘI|CÔNG BÁO).*?\n",
        "",
        text
    )

    # 8. Normalize excessive blank lines
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


def load_llamacloud_processor():
    """Load LlamaCloud processor using the official LlamaCloud SDK."""
    llama_key = LLAMA_CLOUD_API_KEY or os.getenv("LLAMA_CLOUD_API_KEY") or os.getenv("LLAMA_PARSE_API_KEY")
    if llama_key:
        try:
            from llama_cloud import LlamaCloud
            print("Khởi tạo LlamaCloud client (SDK chính thức)...")
            client = LlamaCloud(api_key=llama_key)

            def process_pdf_cloud(pdf_path: Path) -> str:
                print(f"-> [LlamaCloud] Tải lên tập tin {pdf_path.name} (purpose='parse')...")
                file_obj = client.files.create(file=pdf_path, purpose="parse")
                print(f"-> [LlamaCloud] Đã tạo file_id: {file_obj.id}. Đang phân tích (tier='agentic', version='latest')...")
                
                result = client.parsing.parse(
                    file_id=file_obj.id,
                    tier="agentic",
                    version="latest",
                    expand=["markdown"],
                )
                
                pages = result.markdown.pages if (result.markdown and result.markdown.pages) else []
                print(f"-> [LlamaCloud] Hoàn tất bóc tách {len(pages)} trang từ {pdf_path.name}.")
                
                raw_text = "\n\n".join([p.markdown for p in pages if p.markdown])
                cleaned_text = clean_legal_markdown(raw_text)
                return cleaned_text

            return process_pdf_cloud
        except Exception as exc:
            print(f"Cảnh báo LlamaCloud SDK: {exc}. Chuyển sang chế độ local fallback...")

    # Fallback to local PDFReader
    try:
        from llama_index.readers.file import PDFReader
        print("Khởi tạo bộ đọc LlamaIndex PDFReader (Local Fallback)...")
        reader = PDFReader()

        def process_pdf_local(pdf_path: Path) -> str:
            documents = reader.load_data(pdf_path)
            raw_text = "\n\n".join([doc.text for doc in documents])
            cleaned = clean_legal_markdown(raw_text)
            return cleaned

        return process_pdf_local
    except Exception as exc:
        raise ImportError(f"Không thể khởi tạo Llama Cloud PDF Reader: {exc}")


def process_all_pdfs():
    """Process all PDFs in data/raw using Llama Cloud and save as *_llamacloud.md in data/processed."""
    pdf_files = sorted(RAW_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"Không tìm thấy file PDF nào trong {RAW_DIR}")
        return

    print(f"Tìm thấy {len(pdf_files)} file PDF trong {RAW_DIR}")
    try:
        process_pdf = load_llamacloud_processor()
    except Exception as exc:
        print(f"Không thể khởi động Llama Cloud processor: {exc}")
        return

    for pdf_path in pdf_files:
        output_path = PROCESSED_DIR / f"{pdf_path.stem}_llamacloud.md"
        print(f"\n[Llama Cloud Ingestion] Đang xử lý: {pdf_path.name}")
        try:
            text = process_pdf(pdf_path)
            output_path.write_text(text, encoding="utf-8")
            print(f"-> Đã lưu kết quả tại: {output_path}")
        except Exception as exc:
            print(f"Lỗi khi xử lý {pdf_path.name}: {exc}")

    print("\nHoàn tất bóc tách toàn bộ tài liệu.")


if __name__ == "__main__":
    process_all_pdfs()
