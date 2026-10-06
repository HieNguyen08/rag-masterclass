"""
Lab 02 — So sánh chiến lược chunking + cài đặt tay Recall@k, MRR, nDCG@k.

Chạy (trong thư mục labs/):
    python lab02_chunking_metrics.py --selftest   # kiểm tra hàm metric bằng ví dụ tính tay
    python lab02_chunking_metrics.py              # so sánh chunking với BM25 (không cần model)
    python lab02_chunking_metrics.py --dense      # thêm dense (multilingual-e5-small)
"""
from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass

import numpy as np

from common import email_query, load_articles, load_emails
from lab01_bm25_dense_rrf import BM25Retriever


# ---------------------------------------------------------------------------
# 1. Metric (cài tay, không dùng thư viện)
# ---------------------------------------------------------------------------
def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """|relevant ∩ top-k| / |relevant|."""
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def mrr_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """1 / thứ hạng của tài liệu đúng ĐẦU TIÊN trong top-k (0 nếu không có)."""
    for i, d in enumerate(ranked[:k], start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def dcg_at_k(gains: list[float], k: int) -> float:
    """DCG@k = sum_{i=1..k} (2^g_i - 1) / log2(i + 1)."""
    return sum((2 ** g - 1) / math.log2(i + 1) for i, g in enumerate(gains[:k], start=1))


def ndcg_at_k(ranked: list[str], grades: dict[str, float], k: int) -> float:
    """nDCG@k = DCG@k / IDCG@k; grades: doc_id -> mức liên quan (0 = không liên quan)."""
    gains = [grades.get(d, 0.0) for d in ranked]
    ideal = sorted(grades.values(), reverse=True)
    idcg = dcg_at_k(ideal, k)
    return dcg_at_k(gains, k) / idcg if idcg > 0 else 0.0


def grades_from_labels(relevant_doc_ids: list[str]) -> dict[str, float]:
    """Nhãn trong dữ liệu: tài liệu đầu tiên là 'chính' (grade 2), còn lại grade 1."""
    return {d: (2.0 if i == 0 else 1.0) for i, d in enumerate(relevant_doc_ids)}


def selftest() -> None:
    """Ví dụ tính tay (giống bảng trong lab02 markdown)."""
    ranked = ["D3", "D1", "D7", "D2", "D9"]
    grades = {"D1": 2, "D2": 1, "D5": 1}            # D5 không được truy hồi
    rel = set(grades)
    assert recall_at_k(ranked, rel, 5) == 2 / 3
    assert recall_at_k(ranked, rel, 1) == 0.0
    assert mrr_at_k(ranked, rel, 5) == 0.5          # D1 ở hạng 2
    # DCG@5 = (2^2-1)/log2(3) + (2^1-1)/log2(5) = 3/1.585 + 1/2.322 = 1.8928 + 0.4307
    dcg = 3 / math.log2(3) + 1 / math.log2(5)
    # IDCG@5 = 3/log2(2) + 1/log2(3) + 1/log2(4) = 3 + 0.6309 + 0.5
    idcg = 3 + 1 / math.log2(3) + 0.5
    got = ndcg_at_k(ranked, grades, 5)
    assert abs(got - dcg / idcg) < 1e-12
    print(f"selftest OK: Recall@5=0.667  MRR@5=0.5  DCG@5={dcg:.4f}  IDCG@5={idcg:.4f}  nDCG@5={got:.4f}")


# ---------------------------------------------------------------------------
# 2. Chiến lược chunking
# ---------------------------------------------------------------------------
@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str


def _split_sentences(text: str) -> list[str]:
    # Tách câu thô: sau . ? ! 。 và xuống dòng. Đủ cho lab; production cần bộ tách câu tốt hơn.
    parts = re.split(r"(?<=[.?!。])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_whole(article: dict) -> list[Chunk]:
    """Baseline: cả bài là một chunk."""
    body = article["body"].replace("### ", "")
    return [Chunk(f"{article['id']}#0", article["id"], f"{article['title']}\n{body}")]


def chunk_fixed(article: dict, size: int = 40, overlap: int = 10) -> list[Chunk]:
    """Cửa sổ cố định theo 'đơn vị' (từ/âm tiết; với tiếng Nhật là ký tự) có overlap.
    Không thêm tiêu đề -> chunk giữa bài mất ngữ cảnh 'bài này nói về gì'."""
    body = article["body"].replace("### ", "")
    if article["lang"] == "ja":
        units, joiner, size, overlap = list(body), "", size * 3, overlap * 3
    else:
        units, joiner = body.split(), " "
    step = max(1, size - overlap)
    chunks = []
    for n, start in enumerate(range(0, max(1, len(units) - overlap), step)):
        piece = joiner.join(units[start:start + size])
        chunks.append(Chunk(f"{article['id']}#{n}", article["id"], piece))
    return chunks


def chunk_structure(article: dict, max_sentences: int = 6) -> list[Chunk]:
    """Theo cấu trúc: tách ở tiêu đề '### ' và gắn 'Tiêu đề bài > Tiêu đề mục' vào đầu chunk
    (ý tưởng gần với contextual chunk header). Mục quá dài thì tách tiếp theo câu."""
    sections: list[tuple[str, str]] = []
    current_head, buf = "", []
    for line in article["body"].split("\n"):
        if line.startswith("### "):
            if buf:
                sections.append((current_head, " ".join(buf)))
            current_head, buf = line[4:].strip(), []
        elif line.strip():
            buf.append(line.strip())
    if buf:
        sections.append((current_head, " ".join(buf)))

    chunks, n = [], 0
    for head, text in sections:
        header = article["title"] + (f" > {head}" if head else "")
        sents = _split_sentences(text)
        for i in range(0, len(sents), max_sentences):
            chunks.append(Chunk(f"{article['id']}#{n}", article["id"],
                                header + "\n" + " ".join(sents[i:i + max_sentences])))
            n += 1
    return chunks


STRATEGIES = {
    "whole_doc": chunk_whole,
    "fixed_40w_ov10": lambda a: chunk_fixed(a, 40, 10),
    "fixed_20w_ov5": lambda a: chunk_fixed(a, 20, 5),
    "structure+header": chunk_structure,
}


# ---------------------------------------------------------------------------
# 3. Đánh giá: truy hồi chunk -> gộp về tài liệu
# ---------------------------------------------------------------------------
def chunks_to_doc_ranking(chunk_hits: list[tuple[str, float]], chunk_doc: dict[str, str]) -> list[str]:
    """Gộp nhiều chunk cùng tài liệu: giữ thứ hạng của chunk tốt nhất (max-pooling theo thứ tự)."""
    seen, docs = set(), []
    for cid, _ in chunk_hits:
        d = chunk_doc[cid]
        if d not in seen:
            seen.add(d)
            docs.append(d)
    return docs


def evaluate(strategy: str, use_dense: bool, ks=(1, 3, 5)) -> dict[str, float]:
    articles = load_articles()
    chunks = [c for a in articles for c in STRATEGIES[strategy](a)]
    chunk_doc = {c.chunk_id: c.doc_id for c in chunks}
    ids, texts = [c.chunk_id for c in chunks], [c.text for c in chunks]

    if use_dense:
        from lab01_bm25_dense_rrf import DenseRetriever
        retriever = DenseRetriever(ids, texts)
    else:
        retriever = BM25Retriever(ids, texts)

    queries = [e for e in load_emails() if e["relevant_doc_ids"]]
    res = {f"R@{k}": [] for k in ks} | {"MRR@5": [], "nDCG@5": [], "ctx@3": []}
    text_of = dict(zip(ids, texts))
    for e in queries:
        hits = retriever.search(email_query(e), k=50)
        # Số từ của 3 chunk đầu = lượng context sẽ đưa vào LLM (proxy cho số token / chi phí)
        res["ctx@3"].append(sum(len(text_of[cid].split()) for cid, _ in hits[:3]))
        ranked = chunks_to_doc_ranking(hits, chunk_doc)
        rel = set(e["relevant_doc_ids"])
        for k in ks:
            res[f"R@{k}"].append(recall_at_k(ranked, rel, k))
        res["MRR@5"].append(mrr_at_k(ranked, rel, 5))
        res["nDCG@5"].append(ndcg_at_k(ranked, grades_from_labels(e["relevant_doc_ids"]), 5))

    out = {m: float(np.mean(v)) for m, v in res.items()}
    out["#chunks"] = len(chunks)
    out["avg_len"] = float(np.mean([len(t.split()) for t in texts]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dense", action="store_true", help="dùng dense thay BM25")
    ap.add_argument("--show-chunks", default=None, help="in các chunk của một bài, ví dụ KB-006")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if args.show_chunks:
        art = next(a for a in load_articles() if a["id"] == args.show_chunks)
        for name, fn in STRATEGIES.items():
            print(f"=== {name} ===")
            for c in fn(art):
                print(f"[{c.chunk_id}] {c.text[:160]!r}")
        return

    selftest()
    print(f"\nRetriever: {'dense' if args.dense else 'BM25'} | metric ở mức TÀI LIỆU (gộp chunk)")
    cols = ["#chunks", "avg_len", "ctx@3", "R@1", "R@3", "R@5", "MRR@5", "nDCG@5"]
    print(f"{'strategy':<18}" + "".join(f"{c:>9}" for c in cols))
    for name in STRATEGIES:
        r = evaluate(name, args.dense)
        print(f"{name:<18}" + "".join(
            f"{r[c]:>9d}" if c == "#chunks" else f"{r[c]:>9.1f}" if c in ("avg_len", "ctx@3") else f"{r[c]:>9.3f}"
            for c in cols))


if __name__ == "__main__":
    main()
