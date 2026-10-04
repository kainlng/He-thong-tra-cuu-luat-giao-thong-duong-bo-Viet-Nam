import hashlib
from pathlib import Path
from typing import Union


def calculate_file_sha256(file_path: Union[str, Path], chunk_size: int = 65536) -> str:
    """Tính mã băm SHA-256 của tệp tin để phát hiện thay đổi nội dung và chống nạp trùng (Idempotent).

    Args:
        file_path: Đường dẫn tới tệp tin (str hoặc Path).
        chunk_size: Kích thước khối đệm đọc nhị phân (mặc định 64KB = 65536 bytes).

    Returns:
        Chuỗi mã băm chuẩn hóa dạng: 'sha256:<hexdigest>'.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy tệp để tính hash: {file_path}")

    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return f"sha256:{hasher.hexdigest()}"


# Backward compatibility aliases
calculate_file_hash = calculate_file_sha256
calculate_pdf_hash = calculate_file_sha256


import unicodedata
import re

def remove_vietnamese_accents(text: str) -> str:
    """Loại bỏ dấu tiếng Việt để phục vụ normalize search & indexing."""
    text = unicodedata.normalize("NFD", text)
    text = re.sub(r"[\u0300-\u036f]", "", text)
    text = text.replace("đ", "d").replace("Đ", "D")
    return text.lower().strip()
