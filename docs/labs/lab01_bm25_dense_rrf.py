"""
Lab 01 — BM25 + dense retrieval + Reciprocal Rank Fusion (RRF).

Chạy (trong thư mục labs/):
    python lab01_bm25_dense_rrf.py                 # BM25 + dense + RRF
    python lab01_bm25_dense_rrf.py --no-dense      # chỉ BM25 (không cần tải model)
    python lab01_bm25_dense_rrf.py --show E-009    # xem chi tiết một email

Model dense mặc định: intfloat/multilingual-e5-small (~118M tham số, 384 chiều),
chạy được trên CPU; trên RTX 4050 6GB rất thoải mái.
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict

import numpy as np
from rank_bm25 import BM25Okapi

from common import article_text, email_query, load_articles, load_emails, tokenize

DENSE_MODEL = "intfloat/multilingual-e5-small"
# Họ E5 được huấn luyện với tiền tố: "query: " cho câu hỏi, "passage: " cho tài liệu.
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "


# ---------------------------------------------------------------------------
# 1. BM25
# ---------------------------------------------------------------------------
class BM25Retriever:
    def __init__(self, ids: list[str], texts: list[str], k1: float = 1.2, b: float = 0.75,
                 fold: str = "add", bigrams: bool = True):
        self.ids = ids
        self.fold, self.bigrams = fold, bigrams
        corpus_tokens = [tokenize(t, fold=fold, bigrams=bigrams) for t in texts]
        self.bm25 = BM25Okapi(corpus_tokens, k1=k1, b=b)

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        q = tokenize(query, fold=self.fold, bigrams=self.bigrams)
        scores = self.bm25.get_scores(q)               # mảng điểm cho mọi tài liệu
        top = np.argsort(-scores)[:k]
        return [(self.ids[i], float(scores[i])) for i in top]


# ---------------------------------------------------------------------------
# 2. Dense (bi-encoder)
# ---------------------------------------------------------------------------
class DenseRetriever:
    def __init__(self, ids: list[str], texts: list[str], model_name: str = DENSE_MODEL,
                 device: str | None = None, batch_size: int = 32):
        from sentence_transformers import SentenceTransformer  # import muộn: không bắt buộc cho --no-dense

        self.ids = ids
        self.model = SentenceTransformer(model_name, device=device)
        t0 = time.perf_counter()
        # normalize_embeddings=True -> tích vô hướng = cosine
        self.doc_emb = self.model.encode(texts, prompt=PASSAGE_PREFIX, batch_size=batch_size,
                                         normalize_embeddings=True, convert_to_numpy=True)
        self.index_seconds = time.perf_counter() - t0

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        return self.model.encode(queries, prompt=QUERY_PREFIX, normalize_embeddings=True,
                                 convert_to_numpy=True)

    def search(self, query: str, k: int = 10, q_emb: np.ndarray | None = None) -> list[tuple[str, float]]:
        if q_emb is None:
            q_emb = self.encode_queries([query])[0]
        scores = self.doc_emb @ q_emb                 # exact kNN: N phép nhân vô hướng
        top = np.argsort(-scores)[:k]
        return [(self.ids[i], float(scores[i])) for i in top]


# ---------------------------------------------------------------------------
# 3. Reciprocal Rank Fusion
# ---------------------------------------------------------------------------
def rrf(rankings: list[list[str]], k: int = 60, top_n: int = 10,
        weights: list[float] | None = None) -> list[tuple[str, float]]:
    """RRF(d) = sum_r w_r / (k + rank_r(d)), rank bắt đầu từ 1.

    Chỉ dùng THỨ HẠNG, không dùng điểm -> không cần chuẩn hóa thang điểm BM25 vs cosine.
    Tài liệu không xuất hiện trong một danh sách thì đóng góp 0 từ danh sách đó.
    """
    weights = weights or [1.0] * len(rankings)
    fused: dict[str, float] = defaultdict(float)
    for w, ranking in zip(weights, rankings):
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] += w / (k + rank)
    return sorted(fused.items(), key=lambda x: -x[1])[:top_n]


# ---------------------------------------------------------------------------
# 4. Đánh giá nhanh (bản đầy đủ Recall/MRR/nDCG ở lab02)
# ---------------------------------------------------------------------------
def hit_and_rr(ranked_ids: list[str], relevant: set[str], k: int) -> tuple[float, float]:
    """Hit@k (có ít nhất 1 tài liệu đúng trong top-k) và reciprocal rank của tài liệu đúng đầu tiên."""
    hit, rr = 0.0, 0.0
    for i, d in enumerate(ranked_ids[:k], start=1):
        if d in relevant:
            hit = 1.0
            rr = 1.0 / i
            break
    return hit, rr


def build_corpus() -> tuple[list[str], list[str], dict[str, dict]]:
    articles = load_articles()
    ids = [a["id"] for a in articles]
    texts = [article_text(a) for a in articles]
    return ids, texts, {a["id"]: a for a in articles}


def eval_queries() -> list[dict]:
    """Chỉ đánh giá retrieval trên email có nhãn relevant_doc_ids (bỏ email không có tài liệu)."""
    return [e for e in load_emails() if e["relevant_doc_ids"]]


def run_all(use_dense: bool = True, k: int = 5, rrf_k: int = 60, device: str | None = None):
    ids, texts, by_id = build_corpus()
    queries = eval_queries()
    q_texts = [email_query(e) for e in queries]

    systems: dict[str, list[list[str]]] = {}

    t0 = time.perf_counter()
    bm25 = BM25Retriever(ids, texts)
    systems["bm25"] = [[d for d, _ in bm25.search(q, k=20)] for q in q_texts]
    t_bm25 = time.perf_counter() - t0

    # Ablation: BM25 không bigram, không bỏ dấu -> thấy tác dụng của tách từ
    bm25_plain = BM25Retriever(ids, texts, fold="none", bigrams=False)
    systems["bm25_plain"] = [[d for d, _ in bm25_plain.search(q, k=20)] for q in q_texts]

    if use_dense:
        dense = DenseRetriever(ids, texts, device=device)
        t0 = time.perf_counter()
        q_emb = dense.encode_queries(q_texts)
        systems["dense"] = [[d for d, _ in dense.search(q, k=20, q_emb=qe)] for q, qe in zip(q_texts, q_emb)]
        t_dense = time.perf_counter() - t0
        systems["rrf(bm25,dense)"] = [
            [d for d, _ in rrf([b, de], k=rrf_k, top_n=20)]
            for b, de in zip(systems["bm25"], systems["dense"])
        ]
        print(f"[dense] index {len(ids)} bài: {dense.index_seconds:.2f}s | encode {len(q_texts)} query: {t_dense:.2f}s")
    print(f"[bm25 ] index + {len(q_texts)} query: {t_bm25:.3f}s")

    print(f"\nĐánh giá trên {len(queries)} email có nhãn | k={k}")
    print(f"{'hệ thống':<18}{'Hit@1':>8}{'Hit@' + str(k):>8}{'MRR@' + str(k):>8}")
    for name, runs in systems.items():
        h1 = np.mean([hit_and_rr(r, set(e["relevant_doc_ids"]), 1)[0] for r, e in zip(runs, queries)])
        hk, mrr = np.mean([hit_and_rr(r, set(e["relevant_doc_ids"]), k) for r, e in zip(runs, queries)], axis=0)
        print(f"{name:<18}{h1:>8.3f}{hk:>8.3f}{mrr:>8.3f}")

    # Theo ngôn ngữ: thấy rõ chỗ BM25 yếu (tiếng Nhật, không dấu) và dense bù vào
    print("\nHit@1 theo ngôn ngữ:")
    langs = sorted({e["lang"] for e in queries})
    print(f"{'hệ thống':<18}" + "".join(f"{lg + '(' + str(sum(e['lang'] == lg for e in queries)) + ')':>11}" for lg in langs))
    for name, runs in systems.items():
        row = []
        for lg in langs:
            vals = [hit_and_rr(r, set(e["relevant_doc_ids"]), 1)[0] for r, e in zip(runs, queries) if e["lang"] == lg]
            row.append(f"{np.mean(vals):>11.3f}")
        print(f"{name:<18}" + "".join(row))
    return systems, queries, by_id


def show_one(email_id: str, use_dense: bool, device: str | None):
    ids, texts, by_id = build_corpus()
    email = next(e for e in load_emails() if e["id"] == email_id)
    q = email_query(email)
    print("QUERY (đã làm sạch):\n" + q + "\n")
    print("Nhãn đúng:", email["relevant_doc_ids"])
    print("Token BM25:", tokenize(q)[:30], "...\n")
    bm = BM25Retriever(ids, texts).search(q, k=5)
    print("BM25 :", [(d, round(s, 2)) for d, s in bm])
    if use_dense:
        de = DenseRetriever(ids, texts, device=device).search(q, k=5)
        print("Dense:", [(d, round(s, 3)) for d, s in de])
        fused = rrf([[d for d, _ in bm], [d for d, _ in de]])
        print("RRF  :", [(d, round(s, 4)) for d, s in fused[:5]])
    for d, _ in bm[:1]:
        print(f"\nTop-1 BM25 = {d}: {by_id[d]['title']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-dense", action="store_true", help="bỏ dense retrieval (không cần tải model)")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--device", default=None, help="cpu / cuda (mặc định tự chọn)")
    ap.add_argument("--show", default=None, help="id email để xem chi tiết, ví dụ E-009")
    args = ap.parse_args()
    if args.show:
        show_one(args.show, not args.no_dense, args.device)
    else:
        run_all(use_dense=not args.no_dense, k=args.k, rrf_k=args.rrf_k, device=args.device)


if __name__ == "__main__":
    main()
