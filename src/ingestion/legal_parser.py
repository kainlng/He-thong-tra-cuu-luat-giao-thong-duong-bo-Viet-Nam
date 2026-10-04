import re
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from src.ingestion.data_loader import clean_legal_markdown
from src.ingestion.xberg_processor import normalize_xberg_markdown
from src.core.legal_rules import LEGAL_AGENCIES_CONFIG, DEFAULT_AGENCY
from src.core.utils import calculate_file_sha256

PARSER_VERSION = "2.0.0"

# Backward compatibility alias
calculate_file_hash = calculate_file_sha256


def validate_parsed_tree(parsed_tree: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Xác thực tính toàn vẹn ngữ nghĩa và cấu trúc quan hệ của cây pháp lý sau khi parse.
    
    Kiểm tra:
    - [x] Mọi Clause.article_id trỏ tới Article tồn tại
    - [x] Mọi Point.clause_id trỏ tới Clause tồn tại
    - [x] Clause.parent_id trỏ tới Article hợp lệ
    - [x] Point.parent_id trỏ tới Clause hợp lệ
    - [x] Không có duplicate IDs
    - [x] Synthetic node được đánh dấu rõ và clause_number là None
    """
    errors: List[str] = []

    doc = parsed_tree.get("document")
    if not doc or not doc.get("id"):
        errors.append("Document hoặc Document.id bị thiếu.")

    articles = parsed_tree.get("articles", [])
    clauses = parsed_tree.get("clauses", [])
    points = parsed_tree.get("points", [])

    art_ids = {a["id"] for a in articles}
    cl_ids = {c["id"] for c in clauses}
    pt_ids = {p["id"] for p in points}

    # 1. Kiểm tra duplicate IDs
    all_ids = []
    for a in articles:
        all_ids.append(a["id"])
    for c in clauses:
        all_ids.append(c["id"])
    for p in points:
        all_ids.append(p["id"])

    seen_ids = set()
    for uid in all_ids:
        if uid in seen_ids:
            errors.append(f"Duplicate node ID phát hiện: '{uid}'")
        seen_ids.add(uid)

    # 2. Kiểm tra quan hệ Clause -> Article
    for cl in clauses:
        if cl.get("article_id") not in art_ids:
            errors.append(f"Clause '{cl.get('id')}' trỏ tới article_id không tồn tại: '{cl.get('article_id')}'")
        if cl.get("parent_id") and cl.get("parent_id") not in art_ids:
            errors.append(f"Clause '{cl.get('id')}' có parent_id không khớp với Article: '{cl.get('parent_id')}'")
        if cl.get("is_synthetic") and cl.get("clause_number") is not None:
            errors.append(f"Synthetic Clause '{cl.get('id')}' không được phép có clause_number: {cl.get('clause_number')}")

    # 3. Kiểm tra quan hệ Point -> Clause
    for pt in points:
        if pt.get("clause_id") not in cl_ids:
            errors.append(f"Point '{pt.get('id')}' trỏ tới clause_id không tồn tại: '{pt.get('clause_id')}'")
        if pt.get("parent_id") and pt.get("parent_id") not in cl_ids:
            errors.append(f"Point '{pt.get('id')}' có parent_id không khớp với Clause: '{pt.get('parent_id')}'")
        if pt.get("article_id") not in art_ids:
            errors.append(f"Point '{pt.get('id')}' trỏ tới article_id không tồn tại: '{pt.get('article_id')}'")

    is_valid = len(errors) == 0
    return is_valid, errors


class HierarchicalLegalParser:
    """Parser phân tích cú pháp văn bản pháp luật Việt Nam:
    Văn bản -> Chương -> Điều -> Khoản -> Điểm.
    Phiên bản 2.0: Sử dụng compiled header patterns + boundary slicing,
    bảo toàn clause_intro, nhận diện orphan points, và trích xuất ngày/cơ quan tin cậy.
    """

    def __init__(self):
        self.version = PARSER_VERSION

        # Regex nhận diện Chương: Chương I, Chương II, Chương 1, **Chương I...**
        self.chapter_pattern = re.compile(
            r"(?:^|\n)(?:#+\s*|\*{1,2}\s*)?(Chương\s+([IVXLCDM\d]+)[\.:\s]*([^\n\*#]*))(?:\*{1,2})?",
            re.IGNORECASE,
        )

        # Regex nhận diện Điều: Điều 1, Điều 2., **Điều 3.**...
        self.article_pattern = re.compile(
            r"(?:^|\n)(?:#+\s*|\*{1,2}\s*)?(Điều\s+(\d+)[A-Za-z]?[\.:\s\-]*([^\n\*\#]*))(?:\*{1,2})?",
            re.IGNORECASE,
        )

        # Regex nhận diện vị trí bắt đầu Khoản: 1., Khoản 1., - 1., • 1.
        self.clause_header_pattern = re.compile(
            r"(?:^|\n)\s*(?:[-*•]\s*)?(?:Khoản\s+)?(\d+)[\.\)]\s*",
            re.IGNORECASE,
        )

        # Regex nhận diện vị trí bắt đầu Điểm: a), Điểm a), - a), • a)
        self.point_header_pattern = re.compile(
            r"(?:^|\n)\s*(?:[-*•]\s*)?(?:Điểm\s+)?([a-đA-Đ])[\.\)]\s*",
            re.IGNORECASE,
        )

    def extract_issuing_agency(self, doc_num: str, text: str, file_name: str) -> Tuple[Dict[str, Any], str]:
        """Tự động xác định cơ quan ban hành và loại văn bản dựa trên cross-check đa nguồn.
        Trả về: (issuing_agency_dict, doc_type) kèm source và confidence.
        """
        doc_num_upper = doc_num.upper()
        file_name_lower = file_name.lower()
        header_sample = "\n".join(text.split("\n")[:30]).upper()

        num_match_cfg = None
        header_match_cfg = None
        fn_match_cfg = None

        for cfg in LEGAL_AGENCIES_CONFIG:
            if not num_match_cfg and any(re.search(pat, doc_num_upper) for pat in cfg["num_patterns"]):
                num_match_cfg = cfg
            if not header_match_cfg and any(kw in header_sample for kw in cfg["header_keywords"]):
                header_match_cfg = cfg
            if not fn_match_cfg and any(pat in file_name_lower for pat in cfg["filename_patterns"]):
                fn_match_cfg = cfg

        # Cross-check ưu tiên
        if num_match_cfg and header_match_cfg and num_match_cfg["code"] == header_match_cfg["code"]:
            return {
                "name": num_match_cfg["name"],
                "code": num_match_cfg["code"],
                "confidence": "high",
                "source": "header+doc_num",
            }, num_match_cfg["default_doc_type"]
        elif num_match_cfg:
            return {
                "name": num_match_cfg["name"],
                "code": num_match_cfg["code"],
                "confidence": "high",
                "source": "doc_num",
            }, num_match_cfg["default_doc_type"]
        elif header_match_cfg:
            return {
                "name": header_match_cfg["name"],
                "code": header_match_cfg["code"],
                "confidence": "medium",
                "source": "header",
            }, header_match_cfg["default_doc_type"]
        elif fn_match_cfg:
            return {
                "name": fn_match_cfg["name"],
                "code": fn_match_cfg["code"],
                "confidence": "low",
                "source": "filename",
            }, fn_match_cfg["default_doc_type"]

        return {
            "name": None,
            "code": None,
            "confidence": "unknown",
            "source": "unknown",
        }, DEFAULT_AGENCY["default_doc_type"]

    def extract_dates(self, text: str) -> Dict[str, Any]:
        """Bóc tách ngày ban hành (issued_date) và ngày có hiệu lực (effective_from) kèm nguồn và độ tin cậy.
        Tuyệt đối không bịa ngày mặc định nếu không tìm thấy.
        """
        issued_date = None
        issued_source = None
        issued_conf = "unknown"

        effective_from = None
        effective_source = None
        effective_conf = "unknown"

        # 1. Tìm ngày ban hành ở phần ký kết / tiêu ngữ cuối hoặc đầu văn bản (tránh bắt nhầm ngày viện dẫn căn cứ)
        m_closing = re.search(
            r"(?:Hà Nội|TP\.\s*Hồ Chí Minh|[A-ZĐa-đ\s]{2,20}),\s*ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})",
            text,
            re.IGNORECASE,
        )
        if m_closing:
            d, m, y = m_closing.group(1), m_closing.group(2), m_closing.group(3)
            issued_date = f"{y}-{m.zfill(2)}-{d.zfill(2)}"
            issued_source = "closing_formula"
            issued_conf = "high"
        else:
            # Tìm ở đầu văn bản (không nằm trong dòng 'Căn cứ')
            header_lines = text.split("\n")[:25]
            for hline in header_lines:
                hline_clean = hline.strip()
                if hline_clean.lower().startswith("căn cứ"):
                    continue
                m_head_date = re.search(r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})", hline_clean, re.IGNORECASE)
                if m_head_date:
                    d, m, y = m_head_date.group(1), m_head_date.group(2), m_head_date.group(3)
                    issued_date = f"{y}-{m.zfill(2)}-{d.zfill(2)}"
                    issued_source = "header"
                    issued_conf = "medium"
                    break

        # 2. Tìm ngày có hiệu lực thi hành (thường ở Điều khoản thi hành)
        m_eff = re.search(
            r"(?:có hiệu lực|hiệu lực thi hành)(?:\s+từ|\s+kể từ)?\s+ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})",
            text,
            re.IGNORECASE,
        )
        if m_eff:
            eff_pos = m_eff.start()
            pre_window = text[max(0, eff_pos - 30):eff_pos].lower()
            if "hết" not in pre_window:
                d, m, y = m_eff.group(1), m_eff.group(2), m_eff.group(3)
                effective_from = f"{y}-{m.zfill(2)}-{d.zfill(2)}"
                effective_source = "effective_clause"
                effective_conf = "high"

        return {
            "issued_date": issued_date,
            "issued_date_source": issued_source,
            "issued_date_confidence": issued_conf,
            "effective_from": effective_from,
            "effective_date_source": effective_source,
            "effective_date_confidence": effective_conf,
        }

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

        # 4. Trích xuất ngày ban hành và ngày hiệu lực có nguồn & độ tin cậy
        date_info = self.extract_dates(text)

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
            "issued_date": date_info["issued_date"],
            "issued_date_source": date_info["issued_date_source"],
            "issued_date_confidence": date_info["issued_date_confidence"],
            "effective_from": date_info["effective_from"],
            "effective_date_source": date_info["effective_date_source"],
            "effective_date_confidence": date_info["effective_date_confidence"],
            "effective_to": None,
            "is_current": True,
        }

    def parse_clauses_and_points(
        self,
        article_text: str,
        doc_id: str,
        ver_id: str,
        chap_id: Optional[str],
        art_id: str,
        page_start: int = 1,
        page_end: int = 1,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Tách các Khoản và Điểm bên trong một Điều luật bằng Boundary Slicing."""
        clauses = []
        points = []
        seen_clause_ids = set()
        seen_point_ids = set()

        # 1. Tìm tất cả các boundary bắt đầu của Khoản
        clause_matches = list(self.clause_header_pattern.finditer(article_text))

        if clause_matches:
            # Điều có các Khoản đánh số thực tế
            clause_order = 1
            for i, c_match in enumerate(clause_matches):
                clause_num = c_match.group(1)
                base_clause_id = f"clause_{art_id.replace('article_', '')}_{clause_num.zfill(2)}"
                clause_id = base_clause_id
                c_idx = 2
                while clause_id in seen_clause_ids:
                    clause_id = f"{base_clause_id}_{c_idx}"
                    c_idx += 1
                seen_clause_ids.add(clause_id)

                # Boundary slicing cho nội dung Khoản
                c_start = c_match.start()
                c_end = clause_matches[i + 1].start() if i + 1 < len(clause_matches) else len(article_text)
                raw_clause_str = article_text[c_start:c_end].strip()

                # Nội dung sau header Khoản
                c_body_start = c_match.end()
                clause_body = article_text[c_body_start:c_end].strip()

                # Tìm Điểm bên trong Khoản này
                point_matches = list(self.point_header_pattern.finditer(clause_body))

                if point_matches:
                    clause_intro = clause_body[:point_matches[0].start()].strip()
                    point_order = 1

                    for j, p_match in enumerate(point_matches):
                        p_label = p_match.group(1).lower()
                        base_point_id = f"point_{clause_id.replace('clause_', '')}_{p_label}"
                        point_id = base_point_id
                        p_idx = 2
                        while point_id in seen_point_ids:
                            point_id = f"{base_point_id}_{p_idx}"
                            p_idx += 1
                        seen_point_ids.add(point_id)

                        p_content_start = p_match.end()
                        p_content_end = point_matches[j + 1].start() if j + 1 < len(point_matches) else len(clause_body)
                        p_content = clause_body[p_content_start:p_content_end].strip()

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
                            "page_start": page_start,
                            "page_end": page_end,
                        })
                        point_order += 1

                    clauses.append({
                        "id": clause_id,
                        "document_id": doc_id,
                        "version_id": ver_id,
                        "chapter_id": chap_id,
                        "article_id": art_id,
                        "clause_number": clause_num,
                        "clause_intro": clause_intro,
                        "content": clause_intro,
                        "clause_full_text": raw_clause_str,
                        "is_synthetic": False,
                        "synthetic_reason": None,
                        "order_num": clause_order,
                        "parent_id": art_id,
                        "page_start": page_start,
                        "page_end": page_end,
                    })
                else:
                    clauses.append({
                        "id": clause_id,
                        "document_id": doc_id,
                        "version_id": ver_id,
                        "chapter_id": chap_id,
                        "article_id": art_id,
                        "clause_number": clause_num,
                        "clause_intro": None,
                        "content": clause_body,
                        "clause_full_text": raw_clause_str,
                        "is_synthetic": False,
                        "synthetic_reason": None,
                        "order_num": clause_order,
                        "parent_id": art_id,
                        "page_start": page_start,
                        "page_end": page_end,
                    })

                clause_order += 1

        else:
            # Điều không có Khoản đánh số
            # Kiểm tra xem có Điểm xuất hiện trực tiếp không (Orphan Points)
            point_matches = list(self.point_header_pattern.finditer(article_text))
            synth_clause_id = f"clause_{art_id.replace('article_', '')}_body"

            if point_matches:
                # Case E: Có Điểm trực tiếp mà không có Khoản đánh số
                clause_intro = article_text[:point_matches[0].start()].strip()
                point_order = 1

                for j, p_match in enumerate(point_matches):
                    p_label = p_match.group(1).lower()
                    base_point_id = f"point_{synth_clause_id.replace('clause_', '')}_{p_label}"
                    point_id = base_point_id
                    p_idx = 2
                    while point_id in seen_point_ids:
                        point_id = f"{base_point_id}_{p_idx}"
                        p_idx += 1
                    seen_point_ids.add(point_id)

                    p_content_start = p_match.end()
                    p_content_end = point_matches[j + 1].start() if j + 1 < len(point_matches) else len(article_text)
                    p_content = article_text[p_content_start:p_content_end].strip()

                    points.append({
                        "id": point_id,
                        "document_id": doc_id,
                        "version_id": ver_id,
                        "chapter_id": chap_id,
                        "article_id": art_id,
                        "clause_id": synth_clause_id,
                        "point_label": p_label,
                        "content": p_content,
                        "order_num": point_order,
                        "parent_id": synth_clause_id,
                        "page_start": page_start,
                        "page_end": page_end,
                    })
                    point_order += 1

                clauses.append({
                    "id": synth_clause_id,
                    "document_id": doc_id,
                    "version_id": ver_id,
                    "chapter_id": chap_id,
                    "article_id": art_id,
                    "clause_number": None,  # Không bịa Khoản 1
                    "clause_intro": clause_intro,
                    "content": clause_intro,
                    "clause_full_text": article_text.strip(),
                    "is_synthetic": True,
                    "synthetic_reason": "article_without_numbered_clause_with_points",
                    "order_num": 1,
                    "parent_id": art_id,
                    "page_start": page_start,
                    "page_end": page_end,
                })

            else:
                # Case C: Điều chỉ có văn bản thông thường (không có Khoản, không có Điểm)
                clauses.append({
                    "id": synth_clause_id,
                    "document_id": doc_id,
                    "version_id": ver_id,
                    "chapter_id": chap_id,
                    "article_id": art_id,
                    "clause_number": None,  # Không bịa Khoản 1
                    "clause_intro": None,
                    "content": article_text.strip(),
                    "clause_full_text": article_text.strip(),
                    "is_synthetic": True,
                    "synthetic_reason": "article_without_numbered_clause",
                    "order_num": 1,
                    "parent_id": art_id,
                    "page_start": page_start,
                    "page_end": page_end,
                })

        return clauses, points

    def parse_markdown(
        self,
        markdown_text: str,
        file_name: str,
        file_hash: str = "",
        extraction_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Phân tích toàn bộ văn bản markdown thành cây cấu trúc pháp lý hoàn chỉnh kèm page tracking."""
        cleaned_text = clean_legal_markdown(normalize_xberg_markdown(markdown_text))
        doc_info = self.extract_document_info(cleaned_text, file_name)

        # Trích xuất vị trí các marker trang <!-- PAGE \d+ --> để map số trang thực tế
        page_markers: List[Tuple[int, int]] = []
        for m_page in re.finditer(r"<!--\s*PAGE\s+(\d+)\s*-->", cleaned_text):
            page_markers.append((m_page.start(), int(m_page.group(1))))

        def get_page_range(start_idx: int, end_idx: int) -> Tuple[int, int]:
            if not page_markers:
                return 1, 1
            p_start = page_markers[0][1]
            for pos, pnum in page_markers:
                if pos <= start_idx:
                    p_start = pnum
                else:
                    break
            p_end = p_start
            for pos, pnum in page_markers:
                if pos <= end_idx:
                    p_end = max(p_end, pnum)
                else:
                    break
            return p_start, p_end

        doc_page_start = page_markers[0][1] if page_markers else 1
        doc_page_end = page_markers[-1][1] if page_markers else 1

        doc_id = doc_info["id"]
        ver_id = f"version_{doc_id.replace('doc_', '')}_v1"

        # Thiết lập provenance thực tế
        provenance = {
            "file_name": file_name,
            "file_hash": file_hash,
            "source_type": extraction_metadata.get("source_type", "pdf") if extraction_metadata else "pdf",
            "page_start": doc_page_start,
            "page_end": doc_page_end,
            "extraction_backend": extraction_metadata.get("extraction_backend", "xberg") if extraction_metadata else "xberg",
            "ocr_backend": extraction_metadata.get("ocr_backend", "tesseract") if extraction_metadata else "tesseract",
            "ocr_strategy": extraction_metadata.get("ocr_strategy", "scanned_pages") if extraction_metadata else "scanned_pages",
            "parser_version": self.version,
        }

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
            "issued_date_source": doc_info.get("issued_date_source"),
            "issued_date_confidence": doc_info.get("issued_date_confidence"),
            "first_effective_date": doc_info["effective_from"],
            "effective_date_source": doc_info.get("effective_date_source"),
            "effective_date_confidence": doc_info.get("effective_date_confidence"),
            "current_status": doc_info["current_status"],
            "language": "vi",
            "source": provenance,
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

            start_pos = match.end()
            end_pos = article_matches[i + 1].start() if i + 1 < len(article_matches) else len(cleaned_text)
            art_body = cleaned_text[start_pos:end_pos].strip()

            art_page_start, art_page_end = get_page_range(match.start(), end_pos)

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
                    chap_p_start, chap_p_end = get_page_range(
                        (article_matches[i - 1].end() if i > 0 else 0) + chap_match.start(),
                        (article_matches[i - 1].end() if i > 0 else 0) + chap_match.end(),
                    )
                    chapters_dict[current_chapter_id] = {
                        "id": current_chapter_id,
                        "document_id": doc_id,
                        "version_id": ver_id,
                        "chapter_number": chap_num,
                        "title": chap_title or chap_raw,
                        "order_num": chapter_order,
                        "parent_id": None,
                        "page_start": chap_p_start,
                        "page_end": chap_p_end,
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
                    "page_start": art_page_start,
                    "page_end": art_page_end,
                }
                article_order += 1
            else:
                if art_body:
                    articles_dict[art_id]["body_segments"].append(art_body)
                    articles_dict[art_id]["raw_text"] += "\n" + art_body
                    articles_dict[art_id]["page_end"] = max(articles_dict[art_id].get("page_end", 1), art_page_end)

        articles = list(articles_dict.values())
        chapters = list(chapters_dict.values())
        all_clauses = []
        all_points = []

        for art in articles:
            combined_body = "\n".join(art.get("body_segments", []))
            clauses, points = self.parse_clauses_and_points(
                combined_body,
                doc_id,
                ver_id,
                art.get("chapter_id"),
                art["id"],
                page_start=art.get("page_start", 1),
                page_end=art.get("page_end", 1),
            )
            all_clauses.extend(clauses)
            all_points.extend(points)

        parsed_tree = {
            "document": document,
            "version": version,
            "chapters": chapters,
            "articles": articles,
            "clauses": all_clauses,
            "points": all_points,
            "parser_version": self.version,
        }

        # Kiểm tra tính toàn vẹn
        is_valid, validation_errors = validate_parsed_tree(parsed_tree)
        if not is_valid:
            print(f"[Parser Cảnh báo Validation]: Phát hiện {len(validation_errors)} vấn đề: {validation_errors}")

        return parsed_tree
