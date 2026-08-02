import importlib.util
from pathlib import Path
import sys

# Ensure UTF-8 output encoding for Windows terminal
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


def load_marker_processor():
    """Attempt to initialize Marker PDF converter if available."""
    try:
        if (
            importlib.util.find_spec("marker.converters.pdf") is not None
            and importlib.util.find_spec("marker.models") is not None
            and importlib.util.find_spec("marker.output") is not None
        ):
            pdf_module = importlib.import_module("marker.converters.pdf")
            model_module = importlib.import_module("marker.models")
            output_module = importlib.import_module("marker.output")

            PdfConverter = pdf_module.PdfConverter
            create_model_dict = model_module.create_model_dict
            text_from_rendered = output_module.text_from_rendered

            print("Đang nạp các mô hình AI của Marker v2 vào RAM...")
            model_dict = create_model_dict()
            converter = PdfConverter(artifact_dict=model_dict)

            def process_pdf(pdf_path: Path) -> str:
                rendered = converter(str(pdf_path))
                text, _, _ = text_from_rendered(rendered)
                return text

            return process_pdf

        if importlib.util.find_spec("marker.convert") is not None:
            convert_module = importlib.import_module("marker.convert")
            models_module = importlib.import_module("marker.models")

            convert_single_pdf = convert_module.convert_single_pdf
            load_all_models = models_module.load_all_models

            print("Đang tải các mô hình AI của Surya/Marker...")
            model_lst = load_all_models()

            def process_pdf(pdf_path: Path) -> str:
                full_text, _, _ = convert_single_pdf(str(pdf_path), model_lst)
                return full_text

            return process_pdf

    except Exception as exc:
        raise RuntimeError(f"Lỗi khởi động Marker: {exc}")

    raise ImportError("Marker package is not installed.")


def process_pdf_with_docling(pdf_path: Path) -> str:
    """Fallback converter using Docling."""
    if importlib.util.find_spec("docling.document_converter") is None:
        raise ImportError("Docling is not installed.")
    docling_module = importlib.import_module("docling.document_converter")
    DocumentConverter = docling_module.DocumentConverter
    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))
    return result.document.export_to_markdown()


def process_pdf_with_pypdfium2(pdf_path: Path) -> str:
    """Fallback converter using pypdfium2."""
    if importlib.util.find_spec("pypdfium2") is None:
        raise ImportError("pypdfium2 is not installed.")
    pdfium_module = importlib.import_module("pypdfium2")
    pdf = pdfium_module.PdfDocument(str(pdf_path))
    pages_text = []
    for page in pdf:
        text = page.get_textpage().get_text_range()
        if text and text.strip():
            pages_text.append(text.strip())
    return "\n\n".join(pages_text)


def process_pdf_with_pypdf(pdf_path: Path) -> str:
    """Fallback converter using pypdf."""
    if importlib.util.find_spec("pypdf") is None:
        raise ImportError("pypdf is not installed.")
    pypdf_module = importlib.import_module("pypdf")
    reader = pypdf_module.PdfReader(str(pdf_path))
    pages_text = []
    for page in reader.pages:
        text = page.extract_text()
        if text and text.strip():
            pages_text.append(text.strip())
    return "\n\n".join(pages_text)


def convert_pdf(pdf_path: Path, marker_processor=None) -> tuple[str, str]:
    """Convert PDF file to text/markdown with automatic fallbacks if Marker/llama-server is unavailable."""
    # 1. Try Marker
    if marker_processor is not None:
        try:
            text = marker_processor(pdf_path)
            return text, "Marker"
        except Exception as exc:
            print(f"  [Thông báo] Marker không thể xử lý {pdf_path.name}: {exc}")
            print("  -> Chuyển sang phương thức chuyển đổi thay thế (Fallback)...")

    # 2. Try pypdfium2 fallback (Fast & clean text extraction)
    try:
        text = process_pdf_with_pypdfium2(pdf_path)
        return text, "pypdfium2"
    except Exception:
        pass

    # 3. Try pypdf fallback
    try:
        text = process_pdf_with_pypdf(pdf_path)
        return text, "pypdf"
    except Exception:
        pass

    # 4. Try Docling fallback
    try:
        text = process_pdf_with_docling(pdf_path)
        return text, "Docling"
    except Exception:
        pass

    raise RuntimeError(f"Không thể chuyển đổi file {pdf_path.name} bằng bất kỳ công cụ nào.")


def main():
    marker_processor = None
    try:
        marker_processor = load_marker_processor()
    except Exception as exc:
        message = str(exc).strip()
        print(f"Không thể khởi động Marker (do thiếu llama-server hoặc lý do khác): {message}")
        print("Hệ thống sẽ tự động sử dụng bộ chuyển đổi PDF thay thế.\n")

    pdf_files = sorted(RAW_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"Không tìm thấy file PDF nào trong {RAW_DIR}")
        return

    for pdf_path in pdf_files:
        print(f"Đang xử lý: {pdf_path.name}")
        try:
            text, engine_used = convert_pdf(pdf_path, marker_processor=marker_processor)

            output_path = PROCESSED_DIR / f"{pdf_path.stem}_marker.md"
            output_path.write_text(text, encoding="utf-8")
            print(f"  [Thành công] Đã lưu kết quả ({engine_used}) tại: {output_path}\n")

        except Exception as exc:
            print(f"  [Lỗi] Không thể xử lý {pdf_path.name}: {exc}\n")

    print("Hoàn tất xử lý tất cả file PDF!")


if __name__ == "__main__":
    main()

