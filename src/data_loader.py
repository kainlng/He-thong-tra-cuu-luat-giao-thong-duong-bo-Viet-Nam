import os
from dotenv import load_dotenv
from pathlib import Path

# Thêm các import cần thiết để cấu hình Docling
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

load_dotenv()

BASE_DIR = Path(r"E:\Python\He-thong-tra-cuu-luat-giao-thong-duong-bo-Viet-Nam")
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# 1. Khởi tạo tùy chọn pipeline cho PDF
pipeline_options = PdfPipelineOptions()

# 2. TẮT tính năng OCR để chống tràn RAM (vì file Luật đã có sẵn chữ)
pipeline_options.do_ocr = False 

# 3. Vẫn GIỮ nguyên tính năng nhận diện bảng biểu (rất quan trọng cho file Luật)
pipeline_options.do_table_structure = True 

# 4. Khởi tạo converter với cấu hình vừa tạo
converter = DocumentConverter(
    allowed_formats=[InputFormat.PDF],
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)
# ------------------------------

pdf_files = sorted(RAW_DIR.glob("*.pdf"))

if not pdf_files:
    print(f"No PDF files found in {RAW_DIR}")
else:
    for pdf_path in pdf_files:
        print(f"\nProcessing: {pdf_path.name}")
        try:
            result = converter.convert(str(pdf_path))
            markdown_text = result.document.export_to_markdown()

            output_path = PROCESSED_DIR / f"{pdf_path.stem}.md"
            output_path.write_text(markdown_text, encoding="utf-8")
            print(f"Saved: {output_path}")
        except Exception as exc:
            print(f"Failed to process {pdf_path.name}: {exc}")

    print("\nProcessing completed")