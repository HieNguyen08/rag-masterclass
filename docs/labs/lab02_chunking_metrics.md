# Lab 02 — Chiến lược chunking và metric retrieval cài tay

> Thời lượng: ~10 phút · Mức độ: Trung bình · Tiên quyết: Lab 01, Module 04, Module 10 · GPU: không

## Mục tiêu

- Cài đặt tay Recall@k, MRR@k, DCG/nDCG@k và kiểm tra bằng một ví dụ tính tay.
- Cài đặt 4 chiến lược chunking: cả bài, cửa sổ cố định 40 từ và 20 từ (có overlap), theo cấu trúc tiêu đề kèm "header ngữ cảnh".
- Đánh giá retrieval ở **mức tài liệu** khi index ở **mức chunk** (gộp chunk về tài liệu).
- Đọc đúng trade-off: chất lượng truy hồi so với lượng context phải đưa vào LLM.

## Liên hệ lý thuyết

| Khái niệm | Module |
|---|---|
| Fixed-size + overlap, chunk theo cấu trúc, contextual chunk header | Module 04 |
| Precision và recall theo kích thước chunk | Module 04 |
| Recall@k, MRR, nDCG, graded relevance | Module 10 |
| Lost in the middle, chi phí context | Module 02, Module 06 |

## 1. Chạy

```bash
python lab02_chunking_metrics.py --selftest          # kiểm tra metric
python lab02_chunking_metrics.py                     # BM25
python lab02_chunking_metrics.py --dense             # dense (cần tải model)
python lab02_chunking_metrics.py --show-chunks KB-006
```

## 2. Metric: định nghĩa và cài đặt

Ký hiệu: với một query, $\pi=(d_1,d_2,\dots)$ là danh sách đã xếp hạng, $R$ là tập tài liệu liên quan, $g(d)\ge 0$ là mức liên quan (graded).

**Recall@k** — phần tài liệu liên quan lấy được trong top-k:

$$\text{Recall@}k=\frac{|\{d_1,\dots,d_k\}\cap R|}{|R|}$$

**MRR@k** — nghịch đảo hạng của tài liệu đúng đầu tiên, trung bình qua các query:

$$\text{RR@}k=\begin{cases}1/i^{*} & i^{*}=\min\{i\le k: d_i\in R\}\\ 0 & \text{nếu không có}\end{cases}\qquad \text{MRR@}k=\frac{1}{|Q|}\sum_{q}\text{RR@}k(q)$$

**nDCG@k** — dùng mức liên quan và phạt theo log của vị trí:

$$\text{DCG@}k=\sum_{i=1}^{k}\frac{2^{g(d_i)}-1}{\log_2(i+1)},\qquad \text{nDCG@}k=\frac{\text{DCG@}k}{\text{IDCG@}k}$$

trong đó IDCG@k là DCG@k của thứ tự lý tưởng (sắp $g$ giảm dần).

Trong dữ liệu, bài đầu tiên của `relevant_doc_ids` (cùng ngôn ngữ với email) có $g=2$, các bài còn lại $g=1$.

```python
import math

def recall_at_k(ranked, relevant, k):
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)

def mrr_at_k(ranked, relevant, k):
    for i, d in enumerate(ranked[:k], start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0

def dcg_at_k(gains, k):
    return sum((2 ** g - 1) / math.log2(i + 1) for i, g in enumerate(gains[:k], start=1))

def ndcg_at_k(ranked, grades, k):
    gains = [grades.get(d, 0.0) for d in ranked]
    idcg = dcg_at_k(sorted(grades.values(), reverse=True), k)
    return dcg_at_k(gains, k) / idcg if idcg > 0 else 0.0

def grades_from_labels(relevant_doc_ids):
    return {d: (2.0 if i == 0 else 1.0) for i, d in enumerate(relevant_doc_ids)}
```

### Ví dụ tính tay (chính là `selftest()`)

Xếp hạng: `D3, D1, D7, D2, D9`. Liên quan: $g(D1)=2$, $g(D2)=1$, $g(D5)=1$ (D5 không được truy hồi).

| Hạng $i$ | Tài liệu | $g$ | $2^g-1$ | $\log_2(i+1)$ | Đóng góp |
|---|---|---|---|---|---|
| 1 | D3 | 0 | 0 | 1 | 0 |
| 2 | D1 | 2 | 3 | 1,585 | 1,8928 |
| 3 | D7 | 0 | 0 | 2 | 0 |
| 4 | D2 | 1 | 1 | 2,322 | 0,4307 |
| 5 | D9 | 0 | 0 | 2,585 | 0 |

- Recall@5 $=2/3=0{,}667$ (thiếu D5); Recall@1 $=0$.
- RR@5 $=1/2$ (tài liệu đúng đầu tiên ở hạng 2).
- DCG@5 $=1{,}8928+0{,}4307=2{,}3235$.
- IDCG@5 (thứ tự lý tưởng D1, D2, D5): $3/1+1/1{,}585+1/2=4{,}1309$.
- nDCG@5 $=2{,}3235/4{,}1309=0{,}5625$.

```text
$ python lab02_chunking_metrics.py --selftest
selftest OK: Recall@5=0.667  MRR@5=0.5  DCG@5=2.3235  IDCG@5=4.1309  nDCG@5=0.5625
```

Lưu ý: MRR chỉ quan tâm tài liệu đúng **đầu tiên**, nên phù hợp với bài toán CS khi một bài là đủ trả lời. Recall@k quan trọng hơn khi câu trả lời cần ghép nhiều bài (ví dụ "nâng cấp gói + SSO").

## 3. Bốn chiến lược chunking

```python
from dataclasses import dataclass

@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str

def chunk_whole(article):
    body = article["body"].replace("### ", "")
    return [Chunk(f"{article['id']}#0", article["id"], f"{article['title']}\n{body}")]

def chunk_fixed(article, size=40, overlap=10):
    """Cửa sổ trượt theo từ (tiếng Nhật: theo ký tự, size*3). Không có tiêu đề bài."""
    body = article["body"].replace("### ", "")
    if article["lang"] == "ja":
        units, joiner, size, overlap = list(body), "", size * 3, overlap * 3
    else:
        units, joiner = body.split(), " "
    step = max(1, size - overlap)
    return [Chunk(f"{article['id']}#{n}", article["id"], joiner.join(units[s:s + size]))
            for n, s in enumerate(range(0, max(1, len(units) - overlap), step))]

def chunk_structure(article, max_sentences=6):
    """Tách tại '### ', gắn 'Tiêu đề bài > Tiêu đề mục' vào đầu mỗi chunk."""
    sections, head, buf = [], "", []
    for line in article["body"].split("\n"):
        if line.startswith("### "):
            if buf:
                sections.append((head, " ".join(buf)))
            head, buf = line[4:].strip(), []
        elif line.strip():
            buf.append(line.strip())
    if buf:
        sections.append((head, " ".join(buf)))
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
```

Xem chunk của bài bảng giá `KB-006` để thấy vấn đề của cửa sổ cố định: chunk `#1` bắt đầu giữa câu "... và tích hợp Zendesk. Enterprise: báo giá theo hợp đồng ..." — chunk này không còn chữ "gói" hay "giá" nào ở đầu, nên query "gói Business giá bao nhiêu" khó khớp. Chunk theo cấu trúc giữ tiêu đề "Các gói dịch vụ ... > Các gói hiện có" ở đầu, nên mỗi chunk tự giải thích được.

```text
$ python lab02_chunking_metrics.py --show-chunks KB-006
=== fixed_40w_ov10 ===
[KB-006#0] 'Các gói hiện có Starter: 199.000 VND/người dùng/tháng, tối đa 5 người dùng, 1 kho. Business: ...'
[KB-006#1] 'và tích hợp Zendesk. Enterprise: báo giá theo hợp đồng, có SCIM, vai trò tùy chỉnh, SLA 1 giờ ...'
...
```

## 4. Gộp chunk về tài liệu và đánh giá

Index ở mức chunk nhưng nhãn ở mức tài liệu, nên ta lấy top-50 chunk rồi giữ thứ tự xuất hiện đầu tiên của mỗi tài liệu (tương đương max-pooling điểm theo tài liệu). Cột `ctx@3` là tổng số từ của 3 chunk đầu — xấp xỉ lượng context (và chi phí token) đưa vào LLM ở lab04.

```python
def chunks_to_doc_ranking(chunk_hits, chunk_doc):
    seen, docs = set(), []
    for cid, _ in chunk_hits:
        d = chunk_doc[cid]
        if d not in seen:
            seen.add(d)
            docs.append(d)
    return docs

def evaluate(strategy, use_dense, ks=(1, 3, 5)):
    chunks = [c for a in load_articles() for c in STRATEGIES[strategy](a)]
    chunk_doc = {c.chunk_id: c.doc_id for c in chunks}
    ids, texts = [c.chunk_id for c in chunks], [c.text for c in chunks]
    retriever = DenseRetriever(ids, texts) if use_dense else BM25Retriever(ids, texts)
    text_of = dict(zip(ids, texts))
    res = {f"R@{k}": [] for k in ks} | {"MRR@5": [], "nDCG@5": [], "ctx@3": []}
    for e in [e for e in load_emails() if e["relevant_doc_ids"]]:
        hits = retriever.search(email_query(e), k=50)
        res["ctx@3"].append(sum(len(text_of[cid].split()) for cid, _ in hits[:3]))
        ranked = chunks_to_doc_ranking(hits, chunk_doc)
        rel = set(e["relevant_doc_ids"])
        for k in ks:
            res[f"R@{k}"].append(recall_at_k(ranked, rel, k))
        res["MRR@5"].append(mrr_at_k(ranked, rel, 5))
        res["nDCG@5"].append(ndcg_at_k(ranked, grades_from_labels(e["relevant_doc_ids"]), 5))
    return {m: float(np.mean(v)) for m, v in res.items()}
```

## 5. Kết quả mong đợi

Đã chạy thật với BM25:

```text
Retriever: BM25 | metric ở mức TÀI LIỆU (gộp chunk)
strategy            #chunks  avg_len    ctx@3      R@1      R@3      R@5    MRR@5   nDCG@5
whole_doc                40     94.7    325.9    0.604    0.726    0.792    0.859    0.820
fixed_40w_ov10          130     33.9    105.2    0.613    0.708    0.774    0.866    0.820
fixed_20w_ov5           259     17.7     53.9    0.594    0.689    0.774    0.858    0.812
structure+header         74     54.9    176.9    0.604    0.726    0.764    0.858    0.809
```

Cách đọc:

- **R@1 khoảng 0,6 trong khi MRR@5 khoảng 0,86**: không mâu thuẫn. Nhiều email có 2 tài liệu liên quan (bản Việt và bản Anh), nên R@1 tối đa chỉ là 0,5 cho các email đó. MRR chỉ cần một bài đúng.
- **Chất lượng gần như bằng nhau** giữa 4 chiến lược (chênh lệch ≤ 2 điểm, tức ≤ 1 email). Bài Help Center ở đây ngắn (trung bình ~95 từ), nên "cả bài một chunk" là hợp lý. Đó là kết luận đúng cho bộ dữ liệu này, không phải lỗi của lab.
- **Khác biệt thật nằm ở `ctx@3`**: để có cùng MRR, `fixed_40w` chỉ đưa ~105 từ vào LLM so với ~326 từ của `whole_doc` (ít hơn 3 lần). Ở quy mô 1.500 ticket/ngày, đó là chênh lệch chi phí và độ trễ đáng kể (Module 11). Đổi lại, chunk nhỏ dễ cắt mất điều kiện quan trọng: chunk "Hạ cấp có hiệu lực từ kỳ sau, không hoàn lại phần chênh lệch" mà thiếu câu "Thuê bao theo năm..." có thể khiến LLM trả lời sai chính sách.
- Chunk theo cấu trúc tốn ~177 từ nhưng mỗi chunk tự mang tiêu đề: đây là cấu hình mình khuyên dùng cho Help Center thật (bài dài hàng nghìn từ, có heading rõ).

Với `--dense` (chưa chạy được trong sandbox), kỳ vọng chunk rất nhỏ (20 từ) bị thiệt nhiều hơn so với BM25, vì embedding của đoạn ngắn, thiếu ngữ cảnh kém ổn định hơn.

## 6. Bài tập mở rộng

1. **Bài dài**: ghép 5 bài cùng category thành một "bài dài" (mô phỏng tài liệu sản phẩm 500+ từ) rồi chạy lại. Ở điểm nào `whole_doc` bắt đầu thua? (CPU)
2. **Parent-child (small-to-big)**: truy hồi bằng chunk 20 từ nhưng trả về cả mục cha (section) cho LLM. Cài thêm cột `ctx@3` cho chiến lược này và so sánh. (CPU)
3. **Contextual chunk header cho fixed-size**: thêm tiêu đề bài vào đầu mỗi chunk của `chunk_fixed`. R@1 thay đổi thế nào với BM25? Với dense? (CPU/GPU)
4. **Đúng nghĩa thống kê**: viết hàm bootstrap ghép cặp (gợi ý: `paired_bootstrap` trong lab03) cho MRR@5 giữa `whole_doc` và `fixed_20w_ov5`. Khoảng tin cậy 95% có chứa 0 không? (CPU)
5. **MAP**: cài thêm Average Precision và MAP@5, so sánh thứ hạng 4 chiến lược theo MAP và theo nDCG. (CPU)

## Tài liệu tham khảo

- Järvelin, K., Kekäläinen, J. (2002). *Cumulated gain-based evaluation of IR techniques*. ACM TOIS 20(4).
- Voorhees, E. M. (1999). *The TREC-8 Question Answering Track Report* (MRR).
- Anthropic (2024). *Introducing Contextual Retrieval*. https://www.anthropic.com/news/contextual-retrieval
- Liu, N. F. et al. (2023). *Lost in the Middle: How Language Models Use Long Contexts*. arXiv:2307.03172.
