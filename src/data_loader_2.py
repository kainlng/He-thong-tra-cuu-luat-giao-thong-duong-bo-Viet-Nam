import importlib.util
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(r"E:\Python\He-thong-tra-cuu-luat-giao-thong-duong-bo-Viet-Nam")
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def load_marker_processor():
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

    raise ImportError(
        "Marker could not be imported. Install the correct Marker package version or select the right Python environment."
    )


def main():
    process_pdf = load_marker_processor()

    pdf_files = sorted(RAW_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"Không tìm thấy file PDF nào trong {RAW_DIR}")
        return

    for pdf_path in pdf_files:
        print(f"\nĐang xử lý bằng Marker: {pdf_path.name}")
        try:
            text = process_pdf(pdf_path)

            output_path = PROCESSED_DIR / f"{pdf_path.stem}_marker.md"
            output_path.write_text(text, encoding="utf-8")
            print(f"Đã lưu kết quả tại: {output_path}")

        except Exception as exc:
            message = str(exc).strip()
            if "llama-server binary not found" in message.lower() or "llama_cpp" in message.lower():
                print(f"Bỏ qua {pdf_path.name}: thiếu thành phần llama.cpp cần thiết cho Marker.")
            else:
                print(f"Lỗi khi xử lý {pdf_path.name}: {message}")

    print("\nHoàn tất xử lý!")


if __name__ == "__main__":
    main()
