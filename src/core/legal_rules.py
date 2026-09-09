"""Quy tắc phân loại văn bản và cơ quan ban hành trong hệ thống pháp luật Việt Nam."""

from typing import List, Dict, Any

LEGAL_AGENCIES_CONFIG: List[Dict[str, Any]] = [
    {
        "code": "QUOC_HOI",
        "name": "Quốc hội",
        "default_doc_type": "luat",
        "num_patterns": [r"QH\d*"],
        "filename_patterns": ["qh", "luat"],
        "header_keywords": ["QUỐC HỘI"],
        "signer_keywords": ["CHỦ TỊCH QUỐC HỘI", "PHÓ CHỦ TỊCH QUỐC HỘI"],
    },
    {
        "code": "UBTVQH",
        "name": "Ủy ban Thường vụ Quốc hội",
        "default_doc_type": "nghi_quyet",
        "num_patterns": [r"UBTVQH\d*"],
        "filename_patterns": ["ubtvqh"],
        "header_keywords": ["ỦY BAN THƯỜNG VỤ QUỐC HỘI"],
        "signer_keywords": ["CHỦ TỊCH QUỐC HỘI", "PHÓ CHỦ TỊCH QUỐC HỘI"],
    },
    {
        "code": "CHINH_PHU",
        "name": "Chính phủ",
        "default_doc_type": "nghi_dinh",
        "num_patterns": [r"NĐ-CP", r"ND-CP"],
        "filename_patterns": ["nd-cp", "nghi_dinh"],
        "header_keywords": ["CHÍNH PHỦ"],
        "signer_keywords": ["THỦ TƯỚNG", "KT. THỦ TƯỚNG", "PHÓ THỦ TƯỚNG"],
    },
    {
        "code": "TTG",
        "name": "Thủ tướng Chính phủ",
        "default_doc_type": "quyet_dinh",
        "num_patterns": [r"QĐ-TTG", r"QD-TTG"],
        "filename_patterns": ["qd-ttg"],
        "header_keywords": ["THỦ TƯỚNG CHÍNH PHỦ"],
        "signer_keywords": ["THỦ TƯỚNG"],
    },
    {
        "code": "BGTVT",
        "name": "Bộ Giao thông vận tải",
        "default_doc_type": "thong_tu",
        "num_patterns": [r"TT-BGTVT", r"BGTVT"],
        "filename_patterns": ["tt-bgtvt", "bgtvt"],
        "header_keywords": ["BỘ GIAO THÔNG VẬN TẢI"],
        "signer_keywords": ["BỘ TRƯỞNG BỘ GIAO THÔNG VẬN TẢI", "KT. BỘ TRƯỞNG"],
    },
    {
        "code": "BCA",
        "name": "Bộ Công an",
        "default_doc_type": "thong_tu",
        "num_patterns": [r"TT-BCA", r"BCA"],
        "filename_patterns": ["tt-bca", "bca"],
        "header_keywords": ["BỘ CÔNG AN"],
        "signer_keywords": ["BỘ TRƯỞNG BỘ CÔNG AN", "KT. BỘ TRƯỞNG"],
    },
    {
        "code": "BTC",
        "name": "Bộ Tài chính",
        "default_doc_type": "thong_tu",
        "num_patterns": [r"TT-BTC", r"BTC"],
        "filename_patterns": ["tt-btc", "btc"],
        "header_keywords": ["BỘ TÀI CHÍNH"],
        "signer_keywords": ["BỘ TRƯỞNG BỘ TÀI CHÍNH", "KT. BỘ TRƯỞNG"],
    },
    {
        "code": "BYT",
        "name": "Bộ Y tế",
        "default_doc_type": "thong_tu",
        "num_patterns": [r"TT-BYT", r"BYT"],
        "filename_patterns": ["tt-byt", "byt"],
        "header_keywords": ["BỘ Y TẾ"],
        "signer_keywords": ["BỘ TRƯỞNG BỘ Y TẾ", "KT. BỘ TRƯỞNG"],
    },
    {
        "code": "BXD",
        "name": "Bộ Xây dựng",
        "default_doc_type": "thong_tu",
        "num_patterns": [r"TT-BXD", r"BXD"],
        "filename_patterns": ["tt-bxd", "bxd"],
        "header_keywords": ["BỘ XÂY DỰNG"],
        "signer_keywords": ["BỘ TRƯỞNG BỘ XÂY DỰNG", "KT. BỘ TRƯỞNG"],
    },
]

DEFAULT_AGENCY = {
    "name": "Cơ quan nhà nước",
    "code": "OTHER",
    "default_doc_type": "van_ban",
}
