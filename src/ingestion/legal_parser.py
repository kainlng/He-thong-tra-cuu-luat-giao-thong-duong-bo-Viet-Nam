import re
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from src.ingestion.data_loader import clean_legal_markdown
from src.core.legal_rules import LEGAL_AGENCIES_CONFIG, DEFAULT_AGENCY


def calculate_file_hash(file_path: Path) -> str:
    """Tính mã băm SHA-256 của file để đảm bảo tính Idempotent (chống nạp trùng)."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return f"sha256:{hasher.hexdigest()}"


class HierarchicalLegalParser:
    """Parser phân tích cú pháp văn bản pháp luật Việt Nam:
    Văn bản -> Chương -> Điều -> Khoản -> Điểm.
    """

    def __init__(self):
        # Regex nhận diện Chương: Chương I, Chương II, Chương 1...
        self.chapter_pattern = re.compile(
            r"(?:^|\n)(?:#+\s*)?(Chương\s+([IVXLCDM\d]+)[\.:\s]*([^\n]*))",
            re.IGNORECASE,
        )

        # Regex nhận diện Điều: Điều 1, Điều 2., **Điều 3.**...
        self.article_pattern = re.compile(
            r"(?:^|\n)(?:#+\s*|\*{1,2}\s*)?(Điều\s+(\d+)[A-Za-z]?[\.:\s\-]*([^\n\*\#]*))(?:\*{1,2})?",
            re.IGNORECASE,
        )

        # Regex nhận diện Khoản: 1. Nội dung, Khoản 1. Nội dung...
        self.clause_pattern = re.compile(
            r"(?:^|\n)(?:Khoản\s+)?(\d+)[\.\)]\s+([^\n]+(?:\n(?!\s*(?:(?:Khoản\s+)?\d+[\.\)]|[a-đ]\)|\*{1,2}?Điều|\#+))[^\n]+)*)",
            re.IGNORECASE,
        )

        # Regex nhận diện Điểm: a) Nội dung, Điểm a) Nội dung...
        self.point_pattern = re.compile(
            r"(?:^|\n)(?:Điểm\s+)?([a-đA-Đ])[\.\)]\s+([^\n]+(?:\n(?!\s*(?:(?:Khoản\s+)?\d+[\.\)]|[a-đ]\)|\*{1,2}?Điều|\#+))[^\n]+)*)",
            re.IGNORECASE,
        )

    def extract_issuing_agency(self, doc_num: str, text: str, file_name: str) -> Tuple[Dict[str, str], str]:
        """Tự động xác định cơ quan ban hành và loại văn bản dựa trên file cấu hình LEGAL_AGENCIES_CONFIG."""
        doc_num_upper = doc_num.upper()
        file_name_lower = file_name.lower()
        header_sample = "\n".join(text.split("\n")[:25]).upper()

        # 1. Khớp theo hậu tố số hiệu văn bản (Độ chính xác cao nhất)
        for cfg in LEGAL_AGENCIES_CONFIG:
            if any(re.search(pat, doc_num_upper) for pat in cfg["num_patterns"]):
                return {"name": cfg["name"], "code": cfg["code"]}, cfg["default_doc_type"]

        # 2. Khớp theo tên file
        for cfg in LEGAL_AGENCIES_CONFIG:
            if any(pat in file_name_lower for pat in cfg["filename_patterns"]):
                return {"name": cfg["name"], "code": cfg["code"]}, cfg["default_doc_type"]

        # 3. Khớp theo tiêu ngữ đầu văn bản (Header)
        for cfg in LEGAL_AGENCIES_CONFIG:
            if any(kw in header_sample for kw in cfg["header_keywords"]):
                return {"name": cfg["name"], "code": cfg["code"]}, cfg["default_doc_type"]

        return {"name": DEFAULT_AGENCY["name"], "code": DEFAULT_AGENCY["code"]}, DEFAULT_AGENCY["default_doc_type"]

    def extract_dates(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """Bóc tách ngày ban hành (issued_date) và ngày có hiệu lực (effective_from)."""
        issued_date = None
        effective_from = None

        # 1. Bóc tách ngày ban hành
        match_issued = re.search(
            r"(?:ngày|thông qua ngày)\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(20\d{2}|19\d{2})",
            text[:2500],
            re.IGNORECASE,
        )
        if not match_issued:
            match_issued = re.search(
                r"(?:thông qua ngày|ngày)\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(20\d{2}|19\d{2})",
                text[-2000:],
                re.IGNORECASE,
            )

        if match_issued:
            d, m, y = match_issued.group(1), match_issued.group(2), match_issued.group(3)
            issued_date = f"{y}-{m.zfill(2)}-{d.zfill(2)}"

        # 2. Bóc tách ngày có hiệu lực (thường ở Điều khoản thi hành)
        match_eff = re.search(
            r"(?:có hiệu lực|hiệu lực thi hành)(?:\s+từ|\s+kể từ)?\s+ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(20\d{2}|19\d{2})",
            text,
            re.IGNORECASE,
        )
        if match_eff:
            d, m, y = match_eff.group(1), match_eff.group(2), match_eff.group(3)
            effective_from = f"{y}-{m.zfill(2)}-{d.zfill(2)}"

        return issued_date, effective_from

    def extract_document_info(self, text: str, file_name: str) -> Dict[str, Any]:
        """Trích xuất thông tin tổng quan của văn bản từ tên file hoặc phần đầu văn bản."""
        doc_num = ""
        title = ""

        # 1. Trích xuất số hiệu văn bản (text trước, file_name sau)
        num_match = re.search(
            r"(?:Số|Luật số|Nghị định số|Thông tư số|Quyết định số|Nghị quyết số)[\s:]*([0-9]+[/-][0-9]+[/-][A-Za-z0-9Đđ\-_]+)",
            text,
            re.IGNORECASE,
        )
        if not num_match:
            # Thử tìm dạng: Luật số 36/2024/QH15, Nghị định số 151/2024/NĐ-CP
            num_match = re.search(
                r"(?:Luật|Nghị định|Thông tư|Quyết định|Nghị quyết)\s+số\s+([0-9]+[/-][0-9]+[/-][A-Za-z0-9Đđ\-_]+)",
                text,
                re.IGNORECASE,
            )

        if num_match:
            raw_num = num_match.group(1).upper()
            parts = re.split(r"[/_-]", raw_num, maxsplit=2)
            if len(parts) == 3:
                doc_num = f"{parts[0]}/{parts[1]}/{parts[2]}"
            else:
                doc_num = raw_num
        else:
            # 2. Nếu không thấy trong nội dung, phát hiện từ tên file (hỗ trợ cả dấu - và _)
            fn_match = re.search(r"(\d+)[-_](\d+)[-_]([a-zA-Z0-9Đđ]+(?:[-_][a-zA-Z0-9Đđ]+)?)", file_name)
            if fn_match:
                doc_num = f"{fn_match.group(1)}/{fn_match.group(2)}/{fn_match.group(3).upper().replace('_', '-')}"
            else:
                doc_num = file_name.replace(".md", "").replace(".pdf", "").upper()

        # 2. Xác định cơ quan ban hành & loại văn bản tự động từ config
        issuing_agency, doc_type = self.extract_issuing_agency(doc_num, text, file_name)

        # 3. Trích xuất tiêu đề chính (title)
        raw_lines = [l.strip("#* \t\r") for l in text.split("\n")[:35]]
        clean_lines = []
        for l in raw_lines:
            if not l or l.startswith(">") or l.startswith("<") or l.startswith("<!--"):
                continue
            if any(ign in l.upper() for ign in ["CỘNG HÒA", "ĐỘC LẬP", "HÀ NỘI, NGÀY", "THÔNG TIN ĐIỆN TỬ", "NGƯỜI KÝ", "CƠ QUAN:", "THỜI GIAN KÝ:"]):
                continue
            clean_lines.append(l)

        for i, line in enumerate(clean_lines):
            # Kiểm tra nếu dòng có dạng "Luật số ...", "Nghị định số ..."
            m_named_num = re.match(r"^(Luật|Nghị định|Thông tư|Quyết định|Nghị quyết)\s+số\b.*", line, re.IGNORECASE)
            if m_named_num:
                kind_name = m_named_num.group(1).capitalize()
                if i + 1 < len(clean_lines):
                    next_l = clean_lines[i + 1]
                    if (
                        not next_l.startswith("Chương")
                        and not next_l.startswith("Điều")
                        and not next_l.startswith("Số:")
                        and next_l.upper() not in ["LUẬT", "NGHỊ ĐỊNH", "THÔNG TƯ", "QUYẾT ĐỊNH", "NGHỊ QUYẾT"]
                    ):
                        title = f"{kind_name} {next_l.capitalize() if next_l.isupper() else next_l}"
                        break
                continue

            # Bỏ qua các dòng chỉ chứa số hiệu (Số: ...)
            if re.match(r"^Số[\s:]", line, re.IGNORECASE):
                continue

            upper_l = line.upper()
            if upper_l in ["NGHỊ ĐỊNH", "LUẬT", "THÔNG TƯ", "QUYẾT ĐỊNH", "NGHỊ QUYẾT"]:
                if i + 1 < len(clean_lines):
                    sub = clean_lines[i + 1]
                    title = f"{line.capitalize()} {sub.capitalize() if sub.isupper() else sub}"
                else:
                    title = line
                break
            elif any(upper_l.startswith(k) for k in ["NGHỊ ĐỊNH", "LUẬT", "THÔNG TƯ", "QUYẾT ĐỊNH", "NGHỊ QUYẾT", "QUY ĐỊNH"]):
                title = line
                break

        if not title:
            type_labels = {
                "nghi_dinh": "Nghị định",
                "luat": "Luật",
                "thong_tu": "Thông tư",
                "quyet_dinh": "Quyết định",
                "nghi_quyet": "Nghị quyết",
            }
            title = f"{type_labels.get(doc_type, 'Văn bản')} {doc_num}"

        # 4. Trích xuất ngày ban hành và ngày hiệu lực
        extracted_issued, extracted_effective = self.extract_dates(text)

        doc_id = "doc_" + re.sub(r"[^a-zA-Z0-9_]", "_", doc_num.replace("Đ", "D").replace("đ", "d")).lower().strip("_")

        return {
            "id": doc_id,
            "document_number": doc_num,
            "title": title,
            "short_title": f"{doc_type.capitalize()} {doc_num}",
            "document_type": doc_type,
            "issuing_agency": issuing_agency,
            "domain": ["giao_thong_duong_bo"],
            "current_status": "active",
            "issued_date": extracted_issued or "2024-01-01",
            "effective_from": extracted_effective or "2025-01-01",
            "effective_to": None,
            "is_current": True,
        }

    def parse_clauses_and_points(
        self, article_text: str, doc_id: str, ver_id: str, chap_id: Optional[str], art_id: str
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Tách các Khoản và Điểm bên trong một Điều luật."""
        clauses = []
        points = []
        seen_clause_ids = set()
        seen_point_ids = set()

        # Tách theo từng dòng bắt đầu bằng số thứ tự Khoản: 1., 2., 3...
        clause_blocks = re.split(r"(?:^|\n)(?=(?:Khoản\s+)?\d+[\.\)])", article_text)

        clause_order = 1
        for block in clause_blocks:
            block = block.strip()
            if not block:
                continue

            match_clause = re.match(r"^(?:Khoản\s+)?(\d+)[\.\)]\s*(.*)", block, re.DOTALL)
            if match_clause:
                clause_num = match_clause.group(1)
                clause_content = match_clause.group(2).strip()
                base_clause_id = f"clause_{art_id.replace('article_', '')}_{clause_num.zfill(2)}"
                clause_id = base_clause_id
                c_idx = 2
                while clause_id in seen_clause_ids:
                    clause_id = f"{base_clause_id}_{c_idx}"
                    c_idx += 1
                seen_clause_ids.add(clause_id)

                # Kiểm tra xem trong Khoản này có chứa các Điểm: a), b), c)... không
                point_blocks = re.split(r"(?:^|\n)(?=(?:Điểm\s+)?[a-đA-Đ][\.\)])", clause_content)
                
                point_order = 1
                clause_body = point_blocks[0].strip() if point_blocks else clause_content

                if len(point_blocks) > 1:
                    for p_block in point_blocks[1:]:
                        p_block = p_block.strip()
                        if not p_block:
                            continue
                        match_point = re.match(r"^(?:Điểm\s+)?([a-đA-Đ])[\.\)]\s*(.*)", p_block, re.DOTALL)
                        if match_point:
                            p_label = match_point.group(1).lower()
                            p_content = match_point.group(2).strip()
                            base_point_id = f"point_{clause_id.replace('clause_', '')}_{p_label}"
                            point_id = base_point_id
                            p_idx = 2
                            while point_id in seen_point_ids:
                                point_id = f"{base_point_id}_{p_idx}"
                                p_idx += 1
                            seen_point_ids.add(point_id)

                            points.append({
                                "id": point_id,
                                "document_id": doc_id,
                                "version_id": ver_id,
                                "chapter_id": chap_id,
                                "article_id": art_id,
                                "clause_id": clause_id,
                                "point_label": p_label,
                                "content": p_content,
                                "order_num": point_order,
                                "parent_id": clause_id,
                            })
                            point_order += 1

                clauses.append({
                    "id": clause_id,
                    "document_id": doc_id,
                    "version_id": ver_id,
                    "chapter_id": chap_id,
                    "article_id": art_id,
                    "clause_number": clause_num,
                    "content": clause_body,
                    "order_num": clause_order,
                    "parent_id": art_id,
                })
                clause_order += 1
            else:
                # Nếu Điều không chia thành các Khoản đánh số (chỉ có 1 đoạn văn chung)
                if not clauses:
                    clause_id = f"clause_{art_id.replace('article_', '')}_01"
                    if clause_id not in seen_clause_ids:
                        seen_clause_ids.add(clause_id)
                        clauses.append({
                            "id": clause_id,
                            "document_id": doc_id,
                            "version_id": ver_id,
                            "chapter_id": chap_id,
                            "article_id": art_id,
                            "clause_number": "1",
                            "content": block,
                            "order_num": 1,
                            "parent_id": art_id,
                        })
                clause_order += 1

        return clauses, points

    def parse_markdown(self, markdown_text: str, file_name: str, file_hash: str = "") -> Dict[str, Any]:
        """Phân tích toàn bộ văn bản markdown thành cây cấu trúc pháp lý hoàn chỉnh."""
        cleaned_text = clean_legal_markdown(markdown_text)
        doc_info = self.extract_document_info(cleaned_text, file_name)

        doc_id = doc_info["id"]
        ver_id = f"version_{doc_id.replace('doc_', '')}_v1"

        # 1. Tạo LegalDocument
        document = {
            "id": doc_id,
            "document_number": doc_info["document_number"],
            "title": doc_info["title"],
            "short_title": doc_info["short_title"],
            "document_type": doc_info["document_type"],
            "issuing_agency": doc_info["issuing_agency"],
            "domain": doc_info["domain"],
            "issued_date": doc_info["issued_date"],
            "first_effective_date": doc_info["effective_from"],
            "current_status": doc_info["current_status"],
            "language": "vi",
            "source": {
                "file_name": file_name,
                "file_hash": file_hash,
                "source_type": "markdown",
            },
        }

        # 2. Tạo LegalVersion
        version = {
            "id": ver_id,
            "document_id": doc_id,
            "version_number": 1,
            "effective_from": doc_info["effective_from"],
            "effective_to": doc_info["effective_to"],
            "status": doc_info["current_status"],
            "is_current": doc_info["is_current"],
            "amended_by": [],
            "amends": [],
            "replaces": None,
            "legal_basis": [],
        }

        # Map to store unique chapters and merged articles
        chapters_dict = {}
        articles_dict = {}

        current_chapter_id = None
        current_chapter_num = ""
        chapter_order = 1
        article_order = 1

        article_matches = list(self.article_pattern.finditer(cleaned_text))

        for i, match in enumerate(article_matches):
            raw_art_header = match.group(1).strip()
            art_num = match.group(2).strip()
            art_title = match.group(3).strip() if match.group(3) else ""

            art_id = f"article_{doc_id.replace('doc_', '')}_{art_num.zfill(2)}"

            # Lấy nội dung của Điều
            start_pos = match.end()
            end_pos = article_matches[i + 1].start() if i + 1 < len(article_matches) else len(cleaned_text)
            art_body = cleaned_text[start_pos:end_pos].strip()

            # Kiểm tra xem có Chương mới nằm giữa Điều trước và Điều hiện tại không
            preceding_text = cleaned_text[article_matches[i - 1].end() if i > 0 else 0 : match.start()]
            chap_match = self.chapter_pattern.search(preceding_text)
            if chap_match:
                chap_raw = chap_match.group(1).strip()
                chap_num = chap_match.group(2).strip()
                chap_title = chap_match.group(3).strip() if chap_match.group(3) else ""
                current_chapter_id = f"chapter_{doc_id.replace('doc_', '')}_{chap_num.lower()}"
                current_chapter_num = chap_num

                if current_chapter_id not in chapters_dict:
                    chapters_dict[current_chapter_id] = {
                        "id": current_chapter_id,
                        "document_id": doc_id,
                        "version_id": ver_id,
                        "chapter_number": chap_num,
                        "title": chap_title or chap_raw,
                        "order_num": chapter_order,
                        "parent_id": None,
                    }
                    chapter_order += 1

            if art_id not in articles_dict:
                articles_dict[art_id] = {
                    "id": art_id,
                    "document_id": doc_id,
                    "version_id": ver_id,
                    "chapter_id": current_chapter_id,
                    "article_number": art_num,
                    "title": art_title or f"Điều {art_num}",
                    "order_num": article_order,
                    "parent_id": current_chapter_id,
                    "raw_text": f"Điều {art_num}. {art_title}\n{art_body}".strip(),
                    "body_segments": [art_body] if art_body else [],
                }
                article_order += 1
            else:
                # Nếu Điều này bị ngắt trang lặp lại tiêu đề, gộp phần nội dung tiếp theo
                if art_body:
                    articles_dict[art_id]["body_segments"].append(art_body)
                    articles_dict[art_id]["raw_text"] += "\n" + art_body

        articles = list(articles_dict.values())
        chapters = list(chapters_dict.values())
        all_clauses = []
        all_points = []

        # Phân tích Khoản và Điểm cho từng Điều đã được hợp nhất hoàn chỉnh
        for art in articles:
            combined_body = "\n".join(art.get("body_segments", []))
            clauses, points = self.parse_clauses_and_points(
                combined_body, doc_id, ver_id, art.get("chapter_id"), art["id"]
            )
            all_clauses.extend(clauses)
            all_points.extend(points)

        return {
            "document": document,
            "version": version,
            "chapters": chapters,
            "articles": articles,
            "clauses": all_clauses,
            "points": all_points,
        }
