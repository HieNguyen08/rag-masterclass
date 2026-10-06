"""
Tiện ích dùng chung cho các lab: đọc dữ liệu, làm sạch email, tách từ cho BM25,
nhận diện ngôn ngữ đơn giản. Chỉ dùng thư viện chuẩn.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

LAB_DIR = Path(__file__).resolve().parent
DATA_DIR = LAB_DIR / "data"


# ---------------------------------------------------------------------------
# Đọc dữ liệu
# ---------------------------------------------------------------------------
def read_jsonl(path: Path | str) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Không thấy {path}. Hãy chạy: python data/generate_data.py (trong thư mục labs/)")
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_articles() -> list[dict]:
    return read_jsonl(DATA_DIR / "help_center.jsonl")


def load_emails() -> list[dict]:
    return read_jsonl(DATA_DIR / "emails.jsonl")


def article_text(a: dict) -> str:
    """Văn bản dùng để index một bài: tiêu đề + nội dung (bỏ ký hiệu ###)."""
    body = a["body"].replace("### ", "")
    return f"{a['title']}\n{body}"


# ---------------------------------------------------------------------------
# Làm sạch email: bỏ quoted reply, chữ ký, disclaimer
# ---------------------------------------------------------------------------
# Dòng mở đầu đoạn trích dẫn của các client email phổ biến (Gmail/Outlook vi/en/ja)
_QUOTE_HEADER = re.compile(
    r"^(On .+wrote:|Vào .+đã viết:|\d{4}年\d{1,2}月\d{1,2}日.*:|-----Original Message-----|From: .+)$",
    re.IGNORECASE,
)
# Dấu hiệu bắt đầu chữ ký / lời chào cuối thư
_SIGNATURE_START = re.compile(
    r"^(Trân trọng|Tran trong|Thân mến|Best regards|Kind regards|Regards|Thanks,?|Thank you,?|Cheers|"
    r"よろしくお願いいたします|よろしくお願いします|--\s*|---+)\s*[,.。]?\s*$",
    re.IGNORECASE,
)


def clean_email(body: str) -> str:
    """Cắt quoted reply + chữ ký. Đơn giản, dựa trên quy tắc (xem Module 04 cho bản đầy đủ).

    Thứ tự: cắt từ dòng header trích dẫn trở xuống; bỏ mọi dòng bắt đầu bằng '>';
    cắt từ dòng chào cuối thư trở xuống (chữ ký thường nằm sau đó).
    """
    lines = body.splitlines()
    kept: list[str] = []
    for line in lines:
        s = line.strip()
        if _QUOTE_HEADER.match(s):
            break
        if s.startswith(">"):
            continue
        if _SIGNATURE_START.match(s):
            break
        kept.append(line)
    text = "\n".join(kept).strip()
    return text or body.strip()  # nếu cắt hết thì giữ nguyên để không mất câu hỏi


def email_query(email: dict) -> str:
    """Query dùng cho retrieval = tiêu đề (bỏ Re:/Fwd:) + thân email đã làm sạch."""
    subject = re.sub(r"^((re|fw|fwd|tl)\s*:\s*)+", "", email.get("subject", ""), flags=re.I)
    return f"{subject}\n{clean_email(email['body'])}".strip()


# ---------------------------------------------------------------------------
# Tách từ cho BM25 (tiếng Việt / Anh / Nhật)
# ---------------------------------------------------------------------------
_CJK = r"぀-ヿ㐀-䶿一-鿿ｦ-ﾟ"
_CJK_RUN = re.compile(f"[{_CJK}]+")
_TOKEN = re.compile(r"\w+", re.UNICODE)

# Stopword rất ngắn: chỉ để minh họa. Thực tế nên lấy từ thống kê DF của corpus.
STOPWORDS = {
    # vi (có dấu và không dấu)
    "và", "là", "của", "có", "cho", "các", "những", "được", "thì", "mà", "với", "này", "đó", "ạ", "ơi",
    "va", "la", "cua", "co", "cac", "nhung", "duoc", "thi", "ma", "voi", "nay", "do", "a",
    "mình", "minh", "em", "anh", "chị", "chi", "bạn", "ban", "tôi", "toi", "ad", "vậy", "vay",
    # en
    "the", "a", "an", "is", "are", "to", "of", "and", "or", "in", "on", "for", "we", "our", "you",
    "your", "it", "i", "my", "be", "can", "do", "does", "with", "this", "that", "hi", "please",
}


def strip_accents(s: str) -> str:
    """Bỏ dấu tiếng Việt: 'mật khẩu' -> 'mat khau'. Xử lý riêng đ/Đ."""
    s = s.replace("đ", "d").replace("Đ", "D")
    nfd = unicodedata.normalize("NFD", s)
    return "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")


def tokenize(text: str, fold: str = "add", bigrams: bool = True,
             remove_stopwords: bool = True) -> list[str]:
    """Tách từ đơn giản cho BM25.

    - Chuẩn hóa Unicode NFC + lowercase (tránh lỗi tổ hợp dấu khác nhau).
    - Tiếng Việt: tách theo âm tiết (khoảng trắng), thêm bigram âm tiết 'mật_khẩu'
      để xấp xỉ từ ghép mà không cần bộ tách từ (underthesea/pyvi).
    - fold: 'none' giữ dấu; 'add' thêm phiên bản không dấu (khớp email gõ không dấu);
      'only' chỉ dùng không dấu.
    - Tiếng Nhật (không có khoảng trắng): dùng bigram ký tự cho mỗi đoạn CJK.
    """
    text = unicodedata.normalize("NFC", text).lower()
    # Tách các đoạn CJK ra khỏi chữ Latin xung quanh
    text = _CJK_RUN.sub(lambda m: f" {m.group(0)} ", text)
    raw = _TOKEN.findall(text)

    tokens: list[str] = []
    syll: list[str] = []          # chuỗi âm tiết Latin liên tiếp để tạo bigram
    syll_folded: list[str] = []

    def flush() -> None:
        if bigrams:
            base = syll_folded if fold == "only" else syll
            bg = [f"{x}_{y}" for x, y in zip(base, base[1:])]
            tokens.extend(bg)
            if fold == "add":
                # thêm bigram không dấu, chỉ khi khác bản có dấu (tránh đếm tf hai lần)
                bg_f = [f"{x}_{y}" for x, y in zip(syll_folded, syll_folded[1:])]
                tokens.extend(f for f, o in zip(bg_f, bg) if f != o)
        syll.clear()
        syll_folded.clear()

    for tok in raw:
        if _CJK_RUN.fullmatch(tok):
            flush()
            if len(tok) == 1:
                tokens.append(tok)
            else:
                tokens.extend(tok[i:i + 2] for i in range(len(tok) - 1))
            continue
        if remove_stopwords and tok in STOPWORDS:
            # stopword làm đứt chuỗi bigram (tránh bigram vô nghĩa kiểu 'của_tôi')
            flush()
            continue
        folded = strip_accents(tok)
        if fold == "only":
            tokens.append(folded)
        else:
            tokens.append(tok)
            if fold == "add" and folded != tok:
                tokens.append(folded)
        syll.append(tok if fold != "only" else folded)
        syll_folded.append(folded)
    flush()
    return tokens


# ---------------------------------------------------------------------------
# Nhận diện ngôn ngữ thô (đủ cho lab; production nên dùng fastText/lingua)
# ---------------------------------------------------------------------------
_VI_CHARS = re.compile(r"[ăâđêôơưàảãáạằẳẵắặầẩẫấậèẻẽéẹềểễếệìỉĩíịòỏõóọồổỗốộờởỡớợùủũúụừửữứựỳỷỹýỵ]", re.I)
_KANA = re.compile(r"[぀-ヿ]")
_VI_NO_ACCENT_HINTS = {"khong", "duoc", "minh", "cho", "hoi", "nhan", "vien", "ho", "tro", "lam", "sao",
                       "toi", "ban", "gap", "nguoi", "can", "muon", "san", "pham", "nhap", "kho"}


def detect_lang(text: str) -> str:
    if _KANA.search(text) or len(_CJK_RUN.findall(text)) >= 3:
        return "ja"
    if _VI_CHARS.search(text):
        return "vi"
    words = set(re.findall(r"[a-z]+", text.lower()))
    if len(words & _VI_NO_ACCENT_HINTS) >= 3:
        return "vi"
    return "en"
