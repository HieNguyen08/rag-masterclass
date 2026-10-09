"""
Lab 07 — Fine-tune embedding cho retrieval: dữ liệu tổng hợp từ Help Center, hard negative,
InfoNCE với in-batch negatives, đánh giá trước/sau trên email thật bằng bootstrap ghép cặp.

Ba chế độ (trong thư mục labs/):
    python lab07_finetune_embedding.py                    # CPU, chỉ numpy: embedding băm + adapter tuyến tính
    python lab07_finetune_embedding.py --base st          # embedding thật (multilingual-e5-small) + adapter numpy
    python lab07_finetune_embedding.py --st-finetune      # fine-tune toàn bộ model bằng sentence-transformers (GPU 6 GB)

Ý tưởng chế độ numpy: giữ embedding tài liệu cố định, học ma trận W cho phía query sao cho
s(q, d) = <W q, d> / tau — đúng loss InfoNCE của Module 03 (mục 3.2) và Module 09 (mục 3), chỉ
thay "model" bằng một lớp tuyến tính để thấy rõ gradient và chạy được ở bất cứ đâu.
"""
from __future__ import annotations

import argparse
import math
import re
import zlib
from collections import Counter

import numpy as np

from common import (SimpleBM25, article_text, email_query, load_articles, load_emails, mrr_at_k,
                    recall_at_k, tokenize)

E5 = "intfloat/multilingual-e5-small"


# ---------------------------------------------------------------------------
# 1. Dữ liệu huấn luyện tổng hợp: (query giả, bài đúng, hard negative)
# ---------------------------------------------------------------------------
def synthetic_pairs(articles: list[dict]) -> list[tuple[str, str]]:
    """Query giả lấy từ cấu trúc bài: tiêu đề, tiêu đề mục, vài câu đầu.
    Thực tế nên sinh câu hỏi bằng LLM hoặc lấy từ ticket đã giải quyết (Module 09, mục 2)."""
    pairs = []
    for a in articles:
        pairs.append((a["title"], a["id"]))
        for h in re.findall(r"### (.+)", a["body"]):
            pairs.append((f"{a['title']} {h}", a["id"]))
        for s in re.split(r"(?<=[.。!?])\s", a["body"].replace("### ", ""))[:3]:
            if len(s) > 20:
                pairs.append((s, a["id"]))
    return pairs


def mine_hard_negatives(pairs, bm25: SimpleBM25, k: int = 5) -> list[str]:
    """Hard negative = tài liệu BM25 xếp cao nhất nhưng KHÔNG phải bài đúng.
    Cảnh báo false negative (Module 03, mục 3.5): bài 'sai' có thể cũng trả lời được câu hỏi."""
    negs = []
    for q, pos in pairs:
        cand = [d for d, _ in bm25.search(q, k + 1) if d != pos]
        negs.append(cand[0] if cand else pos)
    return negs


# ---------------------------------------------------------------------------
# 2. Embedding nền
# ---------------------------------------------------------------------------
class HashEmbedder:
    """Embedding băm (feature hashing) có trọng số IDF: không cần tải model, tất định.
    Nó *thuần từ vựng* — không hiểu 'mật khẩu' ~ 'password' — nên phần nào giống BM25."""

    def __init__(self, corpus: list[str], dim: int = 1024):
        self.dim = dim
        df = Counter()
        for t in corpus:
            df.update(set(tokenize(t)))
        self.idf = {t: math.log(1 + len(corpus) / c) for t, c in df.items()}

    def encode(self, texts: list[str], kind: str = "query") -> np.ndarray:
        X = np.zeros((len(texts), self.dim))
        for i, t in enumerate(texts):
            for tok in tokenize(t):
                h = zlib.crc32(tok.encode())
                X[i, h % self.dim] += (1 if (h >> 16) & 1 else -1) * self.idf.get(tok, 1.0)
        return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


class STEmbedder:
    def __init__(self, name: str = E5):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(name)

    def encode(self, texts: list[str], kind: str = "query") -> np.ndarray:
        prefix = "query: " if kind == "query" else "passage: "     # họ E5 cần tiền tố (Module 03, mục 9)
        return self.m.encode([prefix + t for t in texts], normalize_embeddings=True, convert_to_numpy=True)


# ---------------------------------------------------------------------------
# 3. Adapter tuyến tính + InfoNCE (numpy)
# ---------------------------------------------------------------------------
def train_adapter(Xq: np.ndarray, pos: np.ndarray, Dm: np.ndarray, tau: float = 0.05, lr: float = 0.5,
                  lam: float = 1e-3, epochs: int = 10, batch: int = 32, seed: int = 0) -> tuple[np.ndarray, list[float]]:
    """Học W tối thiểu hóa  L = -1/B Σ_i log softmax_j(<W q_i, d_j>/tau)[pos_i] + lam ||W - I||².

    Với kho nhỏ (40 bài) ta dùng *toàn bộ* tài liệu làm ứng viên âm (softmax đầy đủ) — tương đương
    in-batch negatives với batch rất lớn và mọi hard negative đều có mặt.
    Gradient: dL/d(Wq_i) = Σ_j (p_ij - y_ij) d_j / tau  ⇒  dL/dW = G^T Q / B + 2 lam (W - I).
    """
    rng = np.random.default_rng(seed)
    d = Xq.shape[1]
    W = np.eye(d)
    losses = []
    for _ in range(epochs):
        idx = rng.permutation(len(Xq))
        tot = 0.0
        for b in range(0, len(idx), batch):
            bi = idx[b:b + batch]
            q = Xq[bi]
            S = (q @ W.T) @ Dm.T / tau
            S -= S.max(axis=1, keepdims=True)
            P = np.exp(S)
            P /= P.sum(axis=1, keepdims=True)
            Y = np.zeros_like(P)
            Y[np.arange(len(bi)), pos[bi]] = 1.0
            tot += float(-np.log(P[np.arange(len(bi)), pos[bi]] + 1e-12).sum())
            G = (P - Y) @ Dm / tau
            W -= lr * (G.T @ q / len(bi) + 2 * lam * (W - np.eye(d)))
        losses.append(tot / len(Xq))
    return W, losses


# ---------------------------------------------------------------------------
# 4. Đánh giá trên email thật + bootstrap ghép cặp (Module 10, mục 6.4)
# ---------------------------------------------------------------------------
def evaluate(Q: np.ndarray, Dm: np.ndarray, ids: list[str], emails: list[dict], W: np.ndarray | None = None):
    S = (Q if W is None else Q @ W.T) @ Dm.T
    r3, mrr = [], []
    for i, e in enumerate(emails):
        ranked = [ids[j] for j in np.argsort(-S[i])]
        r3.append(recall_at_k(ranked, e["relevant_doc_ids"], 3))
        mrr.append(mrr_at_k(ranked, e["relevant_doc_ids"], 5))
    return np.array(r3), np.array(mrr)


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    d = b - a
    boots = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


# ---------------------------------------------------------------------------
# 5. Fine-tune toàn bộ model bằng sentence-transformers (GPU 6 GB; không chạy trong chế độ numpy)
# ---------------------------------------------------------------------------
def st_finetune(pairs, negs, text_of, out_dir: str = "out/e5-small-cs"):
    from datasets import Dataset
    from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments, losses)
    from sentence_transformers.training_args import BatchSamplers

    model = SentenceTransformer(E5)
    train = Dataset.from_dict({
        "anchor": ["query: " + q for q, _ in pairs],
        "positive": ["passage: " + text_of[p] for _, p in pairs],
        "negative": ["passage: " + text_of[n] for n in negs],
    })
    loss = losses.MultipleNegativesRankingLoss(model)           # InfoNCE + in-batch negatives + hard negative
    args = SentenceTransformerTrainingArguments(
        output_dir=out_dir, num_train_epochs=3, per_device_train_batch_size=32, learning_rate=2e-5,
        warmup_ratio=0.1, fp16=True, batch_sampler=BatchSamplers.NO_DUPLICATES,   # tránh trùng tài liệu trong batch
        save_strategy="no", logging_steps=10)
    SentenceTransformerTrainer(model=model, args=args, train_dataset=train, loss=loss).train()
    model.save(out_dir)
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", choices=["hash", "st"], default="hash")
    ap.add_argument("--st-finetune", action="store_true")
    ap.add_argument("--epochs", type=int, default=10)
    a = ap.parse_args()

    articles = load_articles()
    ids = [x["id"] for x in articles]
    text_of = {x["id"]: article_text(x) for x in articles}
    pairs = synthetic_pairs(articles)
    bm25 = SimpleBM25(ids, [text_of[i] for i in ids])
    negs = mine_hard_negatives(pairs, bm25)
    test = [e for e in load_emails() if e["relevant_doc_ids"]]           # email thật: KHÔNG dùng để huấn luyện
    print(f"Cặp huấn luyện tổng hợp: {len(pairs)} | email đánh giá: {len(test)}")

    if a.st_finetune:
        out = st_finetune(pairs, negs, text_of)
        before, after = STEmbedder(E5), STEmbedder(out)
        Db, Da = before.encode([text_of[i] for i in ids], "passage"), after.encode([text_of[i] for i in ids], "passage")
        qs = [email_query(e) for e in test]
        r0, m0 = evaluate(before.encode(qs), Db, ids, test)
        r1, m1 = evaluate(after.encode(qs), Da, ids, test)
    else:
        emb = HashEmbedder([text_of[i] for i in ids]) if a.base == "hash" else STEmbedder()
        Dm = emb.encode([text_of[i] for i in ids], "passage")
        Xq = emb.encode([q for q, _ in pairs], "query")
        pos = np.array([ids.index(p) for _, p in pairs])
        W, losses = train_adapter(Xq, pos, Dm, epochs=a.epochs)
        print("Loss huấn luyện theo epoch:", " ".join(f"{x:.3f}" for x in losses))
        Q = emb.encode([email_query(e) for e in test], "query")
        r0, m0 = evaluate(Q, Dm, ids, test)
        r1, m1 = evaluate(Q, Dm, ids, test, W)
        _, mtr0 = evaluate(Xq, Dm, ids, [{"relevant_doc_ids": [p]} for _, p in pairs])
        _, mtr1 = evaluate(Xq, Dm, ids, [{"relevant_doc_ids": [p]} for _, p in pairs], W)
        print(f"MRR@5 trên chính tập huấn luyện: {mtr0.mean():.3f} → {mtr1.mean():.3f}")

    print(f"Recall@3 (email): {r0.mean():.3f} → {r1.mean():.3f}")
    print(f"MRR@5    (email): {m0.mean():.3f} → {m1.mean():.3f}")
    d, lo, hi = paired_bootstrap(m0, m1)
    print(f"Chênh MRR@5: {d:+.3f}, CI 95% bootstrap ghép cặp [{lo:+.3f}, {hi:+.3f}]")
    print(f"Email tốt lên / xấu đi / không đổi: {(m1 > m0).sum()} / {(m1 < m0).sum()} / {(m1 == m0).sum()}")


if __name__ == "__main__":
    main()
