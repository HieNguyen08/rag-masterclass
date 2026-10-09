"""
Lab 09 — Xử lý query cho email: làm sạch, tách câu hỏi + RRF, mở rộng query (PRF / HyDE),
và đo từng bước trên 53 email có nhãn.

Chạy (trong thư mục labs/):
    python lab09_query_rewriting.py                 # chế độ không cần LLM
    python lab09_query_rewriting.py --show E-037    # xem các query con của một email
    LLM_MODE=openai LLM_BASE_URL=http://localhost:11434/v1 LLM_MODEL=qwen3:4b-instruct-2507-q4_K_M \
        python lab09_query_rewriting.py --hyde      # thêm HyDE và tách câu hỏi bằng LLM thật

Không có LLM, "mở rộng query" dùng pseudo-relevance feedback (PRF): lấy vài câu đầu của tài liệu
top-1 nối vào query — tổ tiên không cần LLM của HyDE (Module 06, mục 5).
"""
from __future__ import annotations

import argparse
import os
import re

import numpy as np

from common import (SimpleBM25, article_text, clean_email, email_query, load_articles, load_emails,
                    mrr_at_k, recall_at_k, rrf)


# ---------------------------------------------------------------------------
# 1. Các phép biến đổi query
# ---------------------------------------------------------------------------
# Chỉ bỏ CỤM lời chào ở đầu dòng ("Hi team,", "Chào Mekong,"), không bỏ cả dòng — câu hỏi hay nằm ngay sau
GREETING = re.compile(r"^(chào|xin chào|dear|hi|hello|kính gửi)\b[^,.!\n]{0,25}[,.!]\s*", re.I | re.M)


def q_raw(e: dict) -> str:
    return f"{e['subject']}\n{e['body']}"                       # nguyên văn: cả quoted reply và chữ ký


def q_clean(e: dict) -> str:
    return GREETING.sub("", email_query(e)).strip()            # Module 04: bỏ quoted reply, chữ ký, lời chào


def q_condensed(e: dict, min_words: int = 30) -> str:
    """Ngưng tụ hội thoại kiểu thô (Module 06, mục 3): nếu email mới quá ngắn ("Vẫn không được"),
    nối thêm phần thư cũ được trích dẫn — nơi chứa ngữ cảnh. Bản LLM viết lại thành một câu hỏi độc lập."""
    q = q_clean(e)
    if len(clean_email(e["body"]).split()) >= min_words:     # đo trên thân email mới, không tính tiêu đề
        return q
    quoted = [l.lstrip("> ").strip() for l in e["body"].splitlines() if l.startswith(">")]
    return q + "\n" + " ".join(quoted)[:600]


def split_questions(text: str) -> list[str]:
    """Tách câu hỏi bằng quy tắc: câu kết thúc bằng '?', '？' hoặc chứa từ để hỏi.
    Phiên bản LLM (Module 06, mục 6) hiểu được câu hỏi ngầm; quy tắc chỉ là baseline."""
    sents = [s.strip() for s in re.split(r"(?<=[?？.!。])\s+|\n+", text) if len(s.strip()) > 8]
    qs = [s for s in sents if re.search(r"[?？]|\b(how|what|why|can|is there|làm sao|thế nào|có .* không|bao nhiêu)\b",
                                         s, re.I)]
    return qs if len(qs) >= 2 else [text]


def llm(prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI(base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"), api_key="local")
    return client.chat.completions.create(model=os.getenv("LLM_MODEL", "qwen3:4b-instruct-2507-q4_K_M"),
                                          messages=[{"role": "user", "content": prompt}],
                                          temperature=0).choices[0].message.content


def hyde_doc(q: str) -> str:
    return llm("Viết một đoạn ngắn (80–120 từ) theo văn phong bài Help Center của một phần mềm SaaS, "
               "trả lời câu hỏi sau. Không cần đúng chi tiết, chỉ cần đúng chủ đề và thuật ngữ.\n\n" + q)


def llm_split(q: str) -> list[str]:
    out = llm("Tách email sau thành danh sách các câu hỏi độc lập, mỗi dòng một câu, giữ nguyên ngôn ngữ. "
              "Nếu chỉ có một câu hỏi, trả về đúng một dòng.\n\n" + q)
    return [l.strip("-• ").strip() for l in out.splitlines() if len(l.strip()) > 5] or [q]


# ---------------------------------------------------------------------------
# 2. Retrieval với nhiều query
# ---------------------------------------------------------------------------
class Searcher:
    def __init__(self):
        arts = load_articles()
        self.ids = [a["id"] for a in arts]
        self.text = {a["id"]: article_text(a) for a in arts}
        self.bm25 = SimpleBM25(self.ids, [self.text[i] for i in self.ids])

    def run(self, q: str, k: int = 10) -> list[str]:
        return [d for d, _ in self.bm25.search(q, k)]

    def multi(self, qs: list[str], k: int = 10) -> list[str]:
        return rrf([self.run(q, k) for q in qs])[:k]          # Module 05, mục 4

    def prf(self, q: str, n_sent: int = 2, k: int = 10) -> list[str]:
        top = self.run(q, 1)[0]
        body = self.text[top].split("\n", 1)[-1]
        expansion = " ".join(re.split(r"(?<=[.。!?])\s", body)[:n_sent])
        return rrf([self.run(q, k), self.run(q + "\n" + expansion, k)])[:k]

    def hyde(self, q: str, k: int = 10) -> list[str]:
        return rrf([self.run(q, k), self.run(hyde_doc(q), k)])[:k]


# ---------------------------------------------------------------------------
# 3. Đánh giá
# ---------------------------------------------------------------------------
def combined_emails(emails: list[dict]) -> list[dict]:
    """Tập kiểm thử có kiểm soát: ghép hai email khác chủ đề thành một email hai câu hỏi.
    Tài liệu đúng = tài liệu đầu tiên (cùng ngôn ngữ) của mỗi email gốc. Bộ dữ liệu mẫu ít email
    nhiều câu hỏi thật, nên ta tạo chúng để đo đúng thứ kỹ thuật tách câu hỏi nhắm tới."""
    vi = [e for e in emails if e["lang"] == "vi" and e["relevant_doc_ids"] and len(e["relevant_doc_ids"]) <= 2]
    out = []
    for a, b in zip(vi[::2], vi[1::2]):
        if a["relevant_doc_ids"][0] == b["relevant_doc_ids"][0]:
            continue
        body = clean_email(a["body"]).strip() + "\nNgoài ra: " + clean_email(b["body"]).strip()
        out.append({"id": f"{a['id']}+{b['id']}", "lang": "vi", "subject": f"{a['subject']} / {b['subject']}",
                    "body": body, "relevant_doc_ids": [a["relevant_doc_ids"][0], b["relevant_doc_ids"][0]]})
    return out


def evaluate(use_llm: bool) -> None:
    S = Searcher()
    emails = [e for e in load_emails() if e["relevant_doc_ids"]]
    combo = combined_emails(emails)
    systems = {
        "nguyên văn (có quoted reply, chữ ký)": lambda e: S.run(q_raw(e)),
        "làm sạch (Module 04)": lambda e: S.run(q_clean(e)),
        "làm sạch + ngữ cảnh khi email quá ngắn": lambda e: S.run(q_condensed(e)),
        "  + tách câu hỏi (quy tắc) + RRF": lambda e: S.multi(split_questions(q_condensed(e))),
        "  + giữ query gốc + câu hỏi tách + RRF": lambda e: S.multi([q_condensed(e)] + split_questions(q_condensed(e))),
        "  + PRF (top-1, 2 câu)": lambda e: S.prf(q_condensed(e)),
    }
    if use_llm:
        systems["  + query gốc + câu hỏi tách bằng LLM + RRF"] = lambda e: S.multi([q_condensed(e)] + llm_split(q_condensed(e)))
        systems["  + HyDE"] = lambda e: S.hyde(q_condensed(e))
    print(f"{'Hệ thống':42s} {'R@3':>6s} {'MRR@5':>6s}   {'ghép 2 câu hỏi: R@3':>20s}")
    for name, fn in systems.items():
        r3 = np.mean([recall_at_k(fn(e), e["relevant_doc_ids"], 3) for e in emails])
        mrr = np.mean([mrr_at_k(fn(e), e["relevant_doc_ids"], 5) for e in emails])
        r3c = np.mean([recall_at_k(fn(e), e["relevant_doc_ids"], 3) for e in combo])
        print(f"{name:42s} {r3:6.3f} {mrr:6.3f}   {r3c:20.3f}")
    print(f"({len(emails)} email thật có tài liệu đúng; {len(combo)} email ghép hai câu hỏi)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show")
    ap.add_argument("--hyde", action="store_true", help="dùng LLM thật cho HyDE và tách câu hỏi")
    a = ap.parse_args()
    if a.show:
        pool = load_emails()
        if "+" in a.show:                                      # email ghép, ví dụ E-007+E-008
            pool = combined_emails([x for x in pool if x["relevant_doc_ids"]])
        e = next(x for x in pool if x["id"] == a.show)
        print("Query sau làm sạch + ngưng tụ:\n" + q_condensed(e) + "\n\nCâu hỏi tách được:")
        for q in split_questions(q_condensed(e)):
            print(" -", q)
        print("Tài liệu đúng:", e["relevant_doc_ids"])
        return
    evaluate(a.hyde or os.getenv("LLM_MODE") == "openai")


if __name__ == "__main__":
    main()
