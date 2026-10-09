"""
Lab 08 — Ingestion: làm sạch email, khử trùng lặp bằng MinHash + LSH, che PII nhất quán.
Chỉ dùng thư viện chuẩn (+ numpy cho thống kê). Chạy trong thư mục labs/:

    python lab08_ingestion_dedup_pii.py            # chạy cả ba phần
    python lab08_ingestion_dedup_pii.py --part dedup
    python lab08_ingestion_dedup_pii.py --part pii --show E-001
"""
from __future__ import annotations

import argparse
import hashlib
import random
import re
import unicodedata
from itertools import combinations

import numpy as np

from common import clean_email, load_articles, load_emails, tokenize

# ---------------------------------------------------------------------------
# Phần A — làm sạch email (Module 04, mục 4)
# ---------------------------------------------------------------------------
def part_clean() -> None:
    emails = load_emails()
    raw = [len(tokenize(e["body"])) for e in emails]
    cln = [len(tokenize(clean_email(e["body"]))) for e in emails]
    lost_q = [e["id"] for e in emails if "?" in e["body"].split("\n>")[0] and "?" not in clean_email(e["body"])]
    print(f"[A] Token BM25 trung bình mỗi email: {np.mean(raw):.0f} → {np.mean(cln):.0f} sau làm sạch "
          f"(giảm {100 * (1 - sum(cln) / sum(raw)):.0f}%)")
    print(f"[A] Email mất dấu '?' sau làm sạch (nghi cắt nhầm câu hỏi): {lost_q or 'không có'}")


# ---------------------------------------------------------------------------
# Phần B — MinHash + LSH (Module 04, mục 5)
# ---------------------------------------------------------------------------
P = (1 << 61) - 1                      # số nguyên tố Mersenne cho hàm băm phổ quát


def shingles(text: str, n: int = 5) -> set[str]:
    t = re.sub(r"\s+", " ", unicodedata.normalize("NFC", text).lower()).strip()
    return {t[i:i + n] for i in range(max(1, len(t) - n + 1))}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 1.0


class MinHasher:
    """h_i(x) = (a_i * x + b_i) mod P — họ băm phổ quát xấp xỉ hoán vị ngẫu nhiên."""

    def __init__(self, k: int = 128, seed: int = 1):
        rng = random.Random(seed)
        self.ab = [(rng.randrange(1, P), rng.randrange(0, P)) for _ in range(k)]

    def signature(self, sh: set[str]) -> list[int]:
        xs = [int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big") for s in sh]
        return [min((a * x + b) % P for x in xs) for a, b in self.ab]


def est_jaccard(s1: list[int], s2: list[int]) -> float:
    return sum(x == y for x, y in zip(s1, s2)) / len(s1)


def lsh_candidates(sigs: dict[str, list[int]], b: int, r: int) -> set[tuple[str, str]]:
    """Chia chữ ký thành b dải, mỗi dải r hàng; hai tài liệu là ứng viên nếu trùng ít nhất một dải."""
    cand = set()
    for band in range(b):
        buckets: dict[tuple, list[str]] = {}
        for doc, s in sigs.items():
            buckets.setdefault(tuple(s[band * r:(band + 1) * r]), []).append(doc)
        for docs in buckets.values():
            cand.update(tuple(sorted(p)) for p in combinations(docs, 2))
    return cand


def make_near_duplicates(articles: list[dict], seed: int = 7) -> list[tuple[str, str]]:
    """Tạo bản gần trùng có kiểm soát: bỏ một câu, đổi một con số, thêm câu chào — mô phỏng
    phiên bản cũ/mới của cùng một bài hoặc macro được sao chép rồi sửa nhẹ."""
    rng = random.Random(seed)
    out = []
    for a in articles[:20]:
        sents = re.split(r"(?<=[.。])\s", a["body"])
        if len(sents) < 3:
            continue
        s = sents[:]
        s.pop(rng.randrange(1, len(s)))                                   # bỏ một câu
        s = [re.sub(r"\d+", lambda m: str(int(m.group()) + 1), x, count=1) if i == 0 else x for i, x in enumerate(s)]
        out.append((a["id"] + "-v2", "Cập nhật: " + " ".join(s)))
    return out


def part_dedup(k: int = 128, b: int = 16, r: int = 8) -> None:
    arts = load_articles()
    docs = {a["id"]: a["body"] for a in arts}
    docs.update(make_near_duplicates(arts))
    sh = {d: shingles(t) for d, t in docs.items()}
    mh = MinHasher(k)
    sigs = {d: mh.signature(s) for d, s in sh.items()}

    pairs = list(combinations(sorted(docs), 2))
    true_j = {p: jaccard(sh[p[0]], sh[p[1]]) for p in pairs}
    est = {p: est_jaccard(sigs[p[0]], sigs[p[1]]) for p in pairs}
    err = np.array([est[p] - true_j[p] for p in pairs])
    print(f"[B] {len(docs)} tài liệu, {len(pairs)} cặp; sai số ước lượng Jaccard: TB {err.mean():+.4f}, "
          f"độ lệch chuẩn {err.std():.4f} (lý thuyết ≤ 1/(2√k) = {1 / (2 * k ** 0.5):.4f})")

    for thr in (0.5, 0.7):
        truth = {p for p in pairs if true_j[p] >= thr}
        print(f"[B] Cặp có Jaccard thật ≥ {thr}: {len(truth)}")
    cand = lsh_candidates(sigs, b, r)
    truth = {p for p in pairs if true_j[p] >= 0.7}
    found = truth & cand
    print(f"[B] LSH b={b}, r={r}: {len(cand)} cặp ứng viên (so với {len(pairs)} cặp nếu so tất cả); "
          f"bắt được {len(found)}/{len(truth)} cặp J ≥ 0,7")
    for j in (0.5, 0.7, 0.8, 0.9):
        print(f"      P(thành ứng viên | J = {j}) = 1 - (1 - J^r)^b = {1 - (1 - j ** r) ** b:.3f}")
    worst = sorted(truth, key=lambda p: true_j[p])[:3]
    print("[B] Cặp gần trùng khó nhất:", ", ".join(f"{p[0]}~{p[1]} J={true_j[p]:.2f}" for p in worst))


# ---------------------------------------------------------------------------
# Phần C — che PII nhất quán theo tài liệu (Module 04, mục 6)
# ---------------------------------------------------------------------------
PII_PATTERNS = [
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
    ("EMAIL", re.compile(r"[\w.+-]+\s*\[(?:at|a còng)\]\s*[\w-]+(?:\s*\[(?:dot|chấm)\]\s*[\w-]+)+", re.I)),
    ("CARD", re.compile(r"\b\d(?:[ -]?\d){12,18}\b")),                    # kiểm tra Luhn bên dưới
    ("PHONE", re.compile(r"(?<!\d)(?:\+84|0084|0)(?:[\s.-]?\d){9,10}(?!\d)")),
    ("PHONE", re.compile(r"(?<!\d)(?:\+81[\s-]?|0)\d{1,4}-\d{1,4}-\d{3,4}(?!\d)")),
    ("TAX", re.compile(r"\b(?:MST|mã số thuế|tax id)\s*[:#]?\s*(\d{10}(?:-\d{3})?)\b", re.I)),
]


def luhn_ok(s: str) -> bool:
    d = [int(c) for c in re.sub(r"\D", "", s)][::-1]
    return len(d) >= 13 and sum(x if i % 2 == 0 else (x * 2 - 9 if x * 2 > 9 else x * 2) for i, x in enumerate(d)) % 10 == 0


def redact(text: str) -> tuple[str, dict[str, str]]:
    """Thay PII bằng nhãn nhất quán trong một tài liệu: cùng giá trị → cùng nhãn (EMAIL_1, PHONE_2...).
    Nhất quán quan trọng để LLM vẫn hiểu 'số điện thoại này' được nhắc lại ở đâu (Module 04, mục 6.3)."""
    mapping: dict[str, str] = {}
    counts: dict[str, int] = {}
    spans = []
    for kind, rx in PII_PATTERNS:
        for m in rx.finditer(text):
            val = m.group(1) if kind == "TAX" else m.group(0)
            st = m.start(1) if kind == "TAX" else m.start()
            if kind == "CARD" and not luhn_ok(val):
                continue
            if any(st < e and st + len(val) > s for s, e, _ in spans):      # bỏ trùng lặp vùng
                continue
            spans.append((st, st + len(val), kind))
    out, last = [], 0
    for s, e, kind in sorted(spans):
        val = text[s:e]
        key = re.sub(r"[\s.+-]", "", val.lower())
        if key not in mapping:
            counts[kind] = counts.get(kind, 0) + 1
            mapping[key] = f"<{kind}_{counts[kind]}>"
        out += [text[last:s], mapping[key]]
        last = e
    out.append(text[last:])
    return "".join(out), mapping


# Bộ kiểm thử nhỏ có nhãn: (văn bản, số PII thật cần che)
PII_TESTS = [
    ("Gọi em qua 0900 000 101 hoặc 0900.000.101 nhé", 2),
    ("SĐT: +84 912 345 678", 1),
    ("Liên hệ mai.tran@anphat.example.com", 1),
    ("email: mai.tran [at] anphat [dot] com", 1),
    ("MST: 0312345678", 1),
    ("Mã số thuế 0312345678-001 của chi nhánh", 1),
    ("Thẻ 4111 1111 1111 1111 bị trừ tiền hai lần", 1),
    ("電話: 03-1234-5678", 1),
    ("Đơn hàng số 2026100512345 bị lỗi", 0),                 # mã đơn 13 số: không phải thẻ (Luhn sai)
    ("Phiên bản 4.12.0 phát hành ngày 05/10/2026", 0),
    ("Hạn mức 1.000.000 lượt/tháng", 0),
    ("Ticket #1234567 đã đóng", 0),
    ("Hotline 1900 1234 của Mekong", 0),                      # số tổng đài công khai: không phải PII cá nhân
    ("Gửi tới 0987654321 và cc anh.bui@anhduong.example.com", 2),
]


def part_pii(show: str | None = None) -> None:
    tp = fp = fn = 0
    for text, n_true in PII_TESTS:
        red, mapping = redact(text)
        n_found = len(re.findall(r"<[A-Z]+_\d+>", red))
        tp += min(n_found, n_true)
        fp += max(0, n_found - n_true)
        fn += max(0, n_true - n_found)
        if n_found != n_true:
            print(f"[C]   lệch: {text!r} → {red!r}")
    print(f"[C] Bộ kiểm thử {len(PII_TESTS)} câu: precision {tp / max(1, tp + fp):.2f}, recall {tp / max(1, tp + fn):.2f} "
          f"(TP {tp}, FP {fp}, FN {fn})")

    emails = load_emails()
    leaked = 0
    for e in emails:
        red, _ = redact(e["body"])
        if e["from"].lower() in red.lower() or re.search(r"0900 000 \d{3}", red):
            leaked += 1
    print(f"[C] 60 email: còn sót email người gửi hoặc SĐT chữ ký trong {leaked} email sau khi che")
    if show:
        e = next(x for x in emails if x["id"] == show)
        red, mapping = redact(e["body"])
        print("---\n" + red + "\n---\nÁnh xạ (lưu riêng, có kiểm soát truy cập):", mapping)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["clean", "dedup", "pii", "all"], default="all")
    ap.add_argument("--show")
    a = ap.parse_args()
    if a.part in ("clean", "all"):
        part_clean()
    if a.part in ("dedup", "all"):
        part_dedup()
    if a.part in ("pii", "all"):
        part_pii(a.show)


if __name__ == "__main__":
    main()
