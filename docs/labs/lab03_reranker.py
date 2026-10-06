"""
Lab 03 — Rerank bằng cross-encoder đa ngữ và đo cải thiện so với lab01.

Chạy (trong thư mục labs/):
    python lab03_reranker.py                                   # bge-reranker-v2-m3 (GPU khuyến nghị)
    python lab03_reranker.py --model cross-encoder/mmarco-mMiniLMv2-L12-H384-v1   # nhẹ, chạy CPU được
    python lab03_reranker.py --first-stage bm25 --top-n 20      # không cần model embedding
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from common import article_text, email_query, load_articles, load_emails
from lab01_bm25_dense_rrf import BM25Retriever, DenseRetriever, rrf
from lab02_chunking_metrics import grades_from_labels, mrr_at_k, ndcg_at_k, recall_at_k

DEFAULT_RERANKER = "BAAI/bge-reranker-v2-m3"          # ~568M tham số, đa ngữ (vi/en/ja)
LIGHT_RERANKER = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"   # ~118M, đa ngữ, nhẹ


class Reranker:
    def __init__(self, model_name: str = DEFAULT_RERANKER, device: str | None = None,
                 max_length: int = 512, fp16: bool = True):
        import torch
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, device=device, max_length=max_length)
        # fp16 trên GPU: giảm một nửa VRAM (568M tham số * 2 byte ~ 1,1 GB) và nhanh hơn
        if fp16 and torch.cuda.is_available() and str(self.model.device).startswith("cuda"):
            self.model.half()

    def rerank(self, query: str, candidates: list[tuple[str, str]], top_k: int | None = None,
               batch_size: int = 16) -> list[tuple[str, float]]:
        """candidates: [(doc_id, text)]. Trả về [(doc_id, score)] đã sắp xếp giảm dần.

        Với model 1 nhãn, CrossEncoder mặc định áp sigmoid -> điểm trong (0, 1),
        dùng được như tín hiệu confidence (lab04/lab05), nhưng CHƯA được hiệu chuẩn.
        """
        if not candidates:
            return []
        hits = self.model.rank(query, [t for _, t in candidates], top_k=top_k, batch_size=batch_size)
        return [(candidates[h["corpus_id"]][0], float(h["score"])) for h in hits]


def first_stage(kind: str, ids, texts, q_texts, n: int, device=None) -> list[list[str]]:
    bm25 = BM25Retriever(ids, texts)
    runs_bm25 = [[d for d, _ in bm25.search(q, k=n)] for q in q_texts]
    if kind == "bm25":
        return runs_bm25
    dense = DenseRetriever(ids, texts, device=device)
    q_emb = dense.encode_queries(q_texts)
    runs_dense = [[d for d, _ in dense.search(q, k=n, q_emb=e)] for q, e in zip(q_texts, q_emb)]
    if kind == "dense":
        return runs_dense
    return [[d for d, _ in rrf([b, de], top_n=n)] for b, de in zip(runs_bm25, runs_dense)]


def metrics(runs: list[list[str]], queries: list[dict]) -> dict[str, float]:
    out = {"R@1": [], "R@3": [], "MRR@5": [], "nDCG@5": []}
    for r, e in zip(runs, queries):
        rel = set(e["relevant_doc_ids"])
        out["R@1"].append(recall_at_k(r, rel, 1))
        out["R@3"].append(recall_at_k(r, rel, 3))
        out["MRR@5"].append(mrr_at_k(r, rel, 5))
        out["nDCG@5"].append(ndcg_at_k(r, grades_from_labels(e["relevant_doc_ids"]), 5))
    return {k: float(np.mean(v)) for k, v in out.items()}


def paired_bootstrap(a: list[float], b: list[float], n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Khoảng tin cậy 95% cho chênh lệch trung bình (b - a) bằng bootstrap ghép cặp (xem Module 10)."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    diffs = b - a
    boots = [diffs[rng.integers(0, len(diffs), len(diffs))].mean() for _ in range(n_boot)]
    return float(diffs.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_RERANKER)
    ap.add_argument("--first-stage", choices=["bm25", "dense", "rrf"], default="rrf")
    ap.add_argument("--top-n", type=int, default=20, help="số ứng viên đưa vào reranker")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    articles = load_articles()
    ids = [a["id"] for a in articles]
    texts = [article_text(a) for a in articles]
    text_of = dict(zip(ids, texts))
    queries = [e for e in load_emails() if e["relevant_doc_ids"]]
    q_texts = [email_query(e) for e in queries]

    base_runs = first_stage(args.first_stage, ids, texts, q_texts, args.top_n, args.device)

    reranker = Reranker(args.model, device=args.device)
    t0 = time.perf_counter()
    rr_runs, top1_scores = [], []
    for q, cand in zip(q_texts, base_runs):
        ranked = reranker.rerank(q, [(d, text_of[d]) for d in cand])
        rr_runs.append([d for d, _ in ranked])
        top1_scores.append(ranked[0][1])
    dt = time.perf_counter() - t0
    n_pairs = len(queries) * args.top_n
    print(f"Rerank {n_pairs} cặp (query, doc) trong {dt:.1f}s -> {1000 * dt / len(queries):.0f} ms/email")

    m_base, m_rr = metrics(base_runs, queries), metrics(rr_runs, queries)
    print(f"\n{'':<22}" + "".join(f"{k:>9}" for k in m_base))
    print(f"{args.first_stage + ' (lab01)':<22}" + "".join(f"{v:>9.3f}" for v in m_base.values()))
    print(f"{'+ rerank':<22}" + "".join(f"{v:>9.3f}" for v in m_rr.values()))

    per_q_base = [mrr_at_k(r, set(e["relevant_doc_ids"]), 5) for r, e in zip(base_runs, queries)]
    per_q_rr = [mrr_at_k(r, set(e["relevant_doc_ids"]), 5) for r, e in zip(rr_runs, queries)]
    mean, lo, hi = paired_bootstrap(per_q_base, per_q_rr)
    print(f"\nΔMRR@5 = {mean:+.3f}  (95% CI bootstrap: [{lo:+.3f}, {hi:+.3f}])")

    better = [e["id"] for e, a, b in zip(queries, per_q_base, per_q_rr) if b > a]
    worse = [e["id"] for e, a, b in zip(queries, per_q_base, per_q_rr) if b < a]
    print(f"Email cải thiện: {better}\nEmail tệ đi   : {worse}")

    # Phân bố điểm top-1 của reranker: tách được 'có tài liệu đúng' và 'không'?
    all_emails = load_emails()
    no_doc = [e for e in all_emails if not e["relevant_doc_ids"]]
    bm25 = BM25Retriever(ids, texts)
    nd_scores = []
    for e in no_doc:
        q = email_query(e)
        cand = [d for d, _ in bm25.search(q, k=args.top_n)]
        nd_scores.append(reranker.rerank(q, [(d, text_of[d]) for d in cand])[0][1])
    print(f"\nĐiểm reranker top-1: email CÓ tài liệu  median={np.median(top1_scores):.3f}"
          f" | email KHÔNG có tài liệu median={np.median(nd_scores):.3f}"
          f" -> gợi ý ngưỡng abstention cho lab04")


if __name__ == "__main__":
    main()
