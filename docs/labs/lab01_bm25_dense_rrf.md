# Lab 01 — BM25, dense retrieval và Reciprocal Rank Fusion

> Thời lượng: ~15 phút · Mức độ: Cơ bản–Trung bình · Tiên quyết: Module 03, Module 05 · GPU: không bắt buộc (dense chạy CPU được)

## Mục tiêu

- Xây retriever BM25 cho corpus Việt/Anh/Nhật với một bộ tách từ đơn giản tự viết (âm tiết + bigram + bỏ dấu; bigram ký tự cho tiếng Nhật).
- Xây dense retriever với `intfloat/multilingual-e5-small` và dùng đúng tiền tố `query:` / `passage:`.
- Cài đặt Reciprocal Rank Fusion (RRF) và giải thích vì sao nó không cần chuẩn hóa điểm.
- Đo Hit@1, Hit@5, MRR@5 theo từng ngôn ngữ, rồi chỉ ra trường hợp BM25 thua và dense bù vào.

## Liên hệ lý thuyết

| Khái niệm trong lab | Module |
|---|---|
| BM25: TF bão hòa ($k_1$), chuẩn hóa độ dài ($b$), IDF | Module 05 |
| Tách từ tiếng Việt (âm tiết và từ ghép), tiếng Nhật (n-gram) | Module 05 |
| Bi-encoder, cosine trên vector đã chuẩn hóa, tiền tố instruction | Module 03 |
| Hybrid retrieval, RRF, $k=60$ | Module 05 |
| Làm sạch email (quoted reply, chữ ký) trước khi tạo query | Module 04 |

## 1. Chuẩn bị

```bash
cd labs
python data/generate_data.py
python lab01_bm25_dense_rrf.py --no-dense    # chỉ BM25, chạy ngay
python lab01_bm25_dense_rrf.py               # BM25 + dense + RRF (tải ~470 MB model lần đầu)
python lab01_bm25_dense_rrf.py --show E-009  # soi chi tiết một email
```

## 2. Từ email đến query: làm sạch trước khi truy hồi

Email thật có quoted reply và chữ ký. Nếu đưa nguyên email vào retriever, các từ trong chữ ký ("Kế toán", tên công ty) và trong đoạn trích dẫn của chính support sẽ nhiễu điểm. `common.py` cắt theo quy tắc đơn giản: dừng ở dòng header trích dẫn hoặc dòng chào cuối thư, bỏ mọi dòng bắt đầu bằng `>`.

```python
# common.py (rút gọn)
_QUOTE_HEADER = re.compile(
    r"^(On .+wrote:|Vào .+đã viết:|\d{4}年\d{1,2}月\d{1,2}日.*:|-----Original Message-----|From: .+)$", re.I)
_SIGNATURE_START = re.compile(
    r"^(Trân trọng|Thân mến|Best regards|Kind regards|Regards|Thanks,?|"
    r"よろしくお願いいたします|--\s*|---+)\s*[,.。]?\s*$", re.I)

def clean_email(body: str) -> str:
    kept = []
    for line in body.splitlines():
        s = line.strip()
        if _QUOTE_HEADER.match(s):      # từ đây trở xuống là thư cũ
            break
        if s.startswith(">"):
            continue
        if _SIGNATURE_START.match(s):   # chữ ký thường nằm sau lời chào
            break
        kept.append(line)
    return "\n".join(kept).strip() or body.strip()

def email_query(email: dict) -> str:
    subject = re.sub(r"^((re|fw|fwd|tl)\s*:\s*)+", "", email.get("subject", ""), flags=re.I)
    return f"{subject}\n{clean_email(email['body'])}".strip()
```

Quy tắc này cố tình đơn giản (Module 04 bàn bản đầy đủ: mô hình phân loại dòng, xử lý reply nằm trên/dưới, disclaimer nhiều ngôn ngữ).

## 3. BM25 và bộ tách từ

### 3.1 Nhắc lại công thức

Với query $q$ gồm các term $t$, tài liệu $d$ có độ dài $|d|$, độ dài trung bình $\text{avgdl}$, $f(t,d)$ là số lần $t$ xuất hiện trong $d$:

$$
\text{BM25}(q,d)=\sum_{t\in q}\text{IDF}(t)\cdot\frac{f(t,d)\,(k_1+1)}{f(t,d)+k_1\left(1-b+b\,\frac{|d|}{\text{avgdl}}\right)}
$$

`rank_bm25.BM25Okapi` dùng $\text{IDF}(t)=\ln\frac{N-n_t+0.5}{n_t+0.5}$ ($N$ số tài liệu, $n_t$ số tài liệu chứa $t$); term xuất hiện ở hơn nửa corpus cho IDF âm, nên thư viện thay bằng một giá trị sàn nhỏ ($\varepsilon \cdot$ IDF trung bình). Lab dùng $k_1=1.2$, $b=0.75$.

Ví dụ số: $k_1=1.2$, $b=0.75$, tài liệu dài đúng bằng trung bình ($|d|/\text{avgdl}=1$). Phần TF bằng $\frac{f\cdot 2.2}{f+1.2}$: $f=1 \to 1.0$; $f=2 \to 1.375$; $f=10 \to 1.96$. Lặp từ 10 lần chỉ được gần gấp đôi điểm so với 1 lần: đó là **bão hòa TF**. Nếu tài liệu dài gấp đôi trung bình, mẫu số thành $f+1.2\cdot 1.75=f+2.1$, nên $f=1$ chỉ còn $2.2/3.1 = 0.71$.

### 3.2 Vì sao cần tách từ riêng cho tiếng Việt và tiếng Nhật

- Tiếng Việt viết cách theo **âm tiết**, không theo từ: "mật khẩu" là 2 token "mật", "khẩu". Token "khẩu" đứng một mình gần như vô nghĩa. Thêm **bigram âm tiết** (`mật_khẩu`) là cách rẻ để xấp xỉ từ ghép mà không cần bộ tách từ như underthesea/pyvi.
- Khách gõ **không dấu** ("doi the thanh toan"). Token `doi` không khớp `đổi`. Ta thêm bản bỏ dấu (`fold="add"`) vào cả tài liệu lẫn query.
- Tiếng Nhật không có khoảng trắng: `\w+` sẽ nuốt cả câu thành một token. Ta cắt mỗi đoạn CJK thành **bigram ký tự** ("パスワード" → "パス", "スワ", "ワー", "ード").

```python
# common.py (rút gọn — bản đầy đủ trong file)
def strip_accents(s: str) -> str:
    s = s.replace("đ", "d").replace("Đ", "D")
    return "".join(ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) != "Mn")

def tokenize(text, fold="add", bigrams=True, remove_stopwords=True) -> list[str]:
    text = unicodedata.normalize("NFC", text).lower()          # NFC: tránh 2 cách mã hóa dấu
    text = _CJK_RUN.sub(lambda m: f" {m.group(0)} ", text)     # tách đoạn CJK khỏi chữ Latin
    tokens, syll, syll_folded = [], [], []

    def flush():                                               # sinh bigram cho chuỗi âm tiết liền nhau
        if bigrams:
            base = syll_folded if fold == "only" else syll
            bg = [f"{x}_{y}" for x, y in zip(base, base[1:])]
            tokens.extend(bg)
            if fold == "add":
                bg_f = [f"{x}_{y}" for x, y in zip(syll_folded, syll_folded[1:])]
                tokens.extend(f for f, o in zip(bg_f, bg) if f != o)
        syll.clear(); syll_folded.clear()

    for tok in _TOKEN.findall(text):
        if _CJK_RUN.fullmatch(tok):                            # tiếng Nhật: bigram ký tự
            flush()
            tokens.extend([tok] if len(tok) == 1 else [tok[i:i+2] for i in range(len(tok) - 1)])
            continue
        if remove_stopwords and tok in STOPWORDS:              # stopword làm đứt chuỗi bigram
            flush(); continue
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
```

Thử nhanh:

```text
>>> tokenize("Đặt lại mật khẩu đăng nhập của tôi")
['đặt','dat','lại','lai','mật','mat','khẩu','khau','đăng','dang','nhập','nhap',
 'đặt_lại','lại_mật','mật_khẩu','khẩu_đăng','đăng_nhập',
 'dat_lai','lai_mat','mat_khau','khau_dang','dang_nhap']
>>> tokenize("quen mat khau")
['quen','mat','khau','quen_mat','mat_khau']          # khớp được 'mat_khau' của tài liệu
>>> tokenize("パスワードの再設定 API 429")
['パス','スワ','ワー','ード','ドの','の再','再設','設定','api','429','api_429']
```

### 3.3 Retriever BM25

```python
import numpy as np
from rank_bm25 import BM25Okapi

class BM25Retriever:
    def __init__(self, ids, texts, k1=1.2, b=0.75, fold="add", bigrams=True):
        self.ids, self.fold, self.bigrams = ids, fold, bigrams
        corpus_tokens = [tokenize(t, fold=fold, bigrams=bigrams) for t in texts]
        self.bm25 = BM25Okapi(corpus_tokens, k1=k1, b=b)

    def search(self, query, k=10):
        scores = self.bm25.get_scores(tokenize(query, fold=self.fold, bigrams=self.bigrams))
        top = np.argsort(-scores)[:k]
        return [(self.ids[i], float(scores[i])) for i in top]
```

`get_scores` tính điểm cho **mọi** tài liệu, độ phức tạp $O(|q|\cdot N)$. Với 40 bài thì không sao; ở quy mô 800 bài + 200.000 ticket thì phải dùng inverted index (Elasticsearch/OpenSearch, Tantivy, hay `bm25s`) — xem Module 05.

## 4. Dense retrieval với multilingual-e5-small

`intfloat/multilingual-e5-small` (~118M tham số, vector 384 chiều, ~100 ngôn ngữ trong đó có vi/en/ja) đủ nhỏ để chạy CPU. Model card yêu cầu tiền tố **`query: `** cho câu hỏi và **`passage: `** cho tài liệu; quên tiền tố là lỗi phổ biến làm giảm chất lượng mà không báo lỗi.

```python
from sentence_transformers import SentenceTransformer

DENSE_MODEL = "intfloat/multilingual-e5-small"

class DenseRetriever:
    def __init__(self, ids, texts, model_name=DENSE_MODEL, device=None):
        self.ids = ids
        self.model = SentenceTransformer(model_name, device=device)
        # normalize_embeddings=True -> tích vô hướng chính là cosine
        self.doc_emb = self.model.encode(texts, prompt="passage: ",
                                         normalize_embeddings=True, convert_to_numpy=True)

    def encode_queries(self, queries):
        return self.model.encode(queries, prompt="query: ",
                                 normalize_embeddings=True, convert_to_numpy=True)

    def search(self, query, k=10, q_emb=None):
        if q_emb is None:
            q_emb = self.encode_queries([query])[0]
        scores = self.doc_emb @ q_emb          # exact kNN; ANN (HNSW) chỉ cần khi N lớn
        top = np.argsort(-scores)[:k]
        return [(self.ids[i], float(scores[i])) for i in top]
```

Tham số `prompt=` của `encode` (sentence-transformers ≥ 2.4, vẫn có ở 6.x) ghép tiền tố vào trước mỗi câu, tương đương với việc tự viết `"query: " + q`.

## 5. Reciprocal Rank Fusion

BM25 trả điểm trong khoảng 0–40, cosine trong khoảng 0,7–0,9: cộng thẳng hai thang này là vô nghĩa. RRF chỉ dùng **thứ hạng**:

$$
\text{RRF}(d)=\sum_{r\in R}\frac{w_r}{k+\text{rank}_r(d)}
$$

với $\text{rank}$ bắt đầu từ 1 và $k=60$ (giá trị Cormack và cộng sự, 2009 đề xuất). Hằng số $k$ làm phẳng chênh lệch giữa các hạng đầu: với $k=60$, hạng 1 được $1/61=0.0164$, hạng 2 được $1/62=0.0161$, gần như ngang nhau. Nhờ vậy một tài liệu đứng đầu ở **cả hai** danh sách luôn thắng một tài liệu chỉ đứng đầu ở một danh sách.

Ví dụ số: BM25 xếp `[KB-016, KB-017, KB-002]`, dense xếp `[KB-017, KB-040, KB-016]`.

| Tài liệu | Hạng BM25 | Hạng dense | RRF |
|---|---|---|---|
| KB-017 | 2 | 1 | $1/62+1/61=0.03252$ |
| KB-016 | 1 | 3 | $1/61+1/63=0.03227$ |
| KB-040 | — | 2 | $1/62=0.01613$ |
| KB-002 | 3 | — | $1/63=0.01587$ |

```python
from collections import defaultdict

def rrf(rankings, k=60, top_n=10, weights=None):
    weights = weights or [1.0] * len(rankings)
    fused = defaultdict(float)
    for w, ranking in zip(weights, rankings):
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] += w / (k + rank)
    return sorted(fused.items(), key=lambda x: -x[1])[:top_n]
```

## 6. Đánh giá

Chỉ 53 email có `relevant_doc_ids` được dùng để đo retrieval (7 email không có tài liệu nào để dành cho abstention ở lab04). Hit@k = có ít nhất một tài liệu đúng trong top-k; MRR@k = trung bình của $1/\text{hạng}$ của tài liệu đúng đầu tiên (định nghĩa đầy đủ ở lab02).

```python
def hit_and_rr(ranked_ids, relevant, k):
    for i, d in enumerate(ranked_ids[:k], start=1):
        if d in relevant:
            return 1.0, 1.0 / i
    return 0.0, 0.0

queries = [e for e in load_emails() if e["relevant_doc_ids"]]
q_texts = [email_query(e) for e in queries]
bm25 = BM25Retriever(ids, texts)
runs_bm25 = [[d for d, _ in bm25.search(q, k=20)] for q in q_texts]
dense = DenseRetriever(ids, texts)
q_emb = dense.encode_queries(q_texts)
runs_dense = [[d for d, _ in dense.search(q, 20, e)] for q, e in zip(q_texts, q_emb)]
runs_rrf = [[d for d, _ in rrf([b, de], top_n=20)] for b, de in zip(runs_bm25, runs_dense)]
```

Script in thêm bảng Hit@1 theo ngôn ngữ và một cấu hình ablation `bm25_plain` (giữ dấu, không bigram) để bạn thấy tác dụng của bộ tách từ.

## 7. Kết quả mong đợi

Phần BM25 (đã chạy thật trong sandbox, dữ liệu mặc định):

```text
Đánh giá trên 53 email có nhãn | k=5
hệ thống             Hit@1   Hit@5   MRR@5
bm25                 0.830   0.906   0.859
bm25_plain           0.849   0.906   0.874

Hit@1 theo ngôn ngữ:
hệ thống               en(14)      ja(4)   mixed(2)     vi(33)
bm25                    0.786      0.750      1.000      0.848
bm25_plain              0.786      0.750      1.000      0.879
```

Đọc kết quả cho đúng:

- Trên bộ này, bigram + bỏ dấu **không** thắng bản thô (khác nhau 1/53 email = 1,9 điểm). Lý do: tài liệu ngắn, email thường dùng đúng từ khóa của bài. Bigram cũng làm IDF của các cụm phổ biến ("thanh_toán", "người_dùng") lấn át. Đây là bài học thật: một kỹ thuật "hợp lý trên giấy" phải được đo trên dữ liệu của mình, và 53 mẫu không đủ để phân biệt chênh lệch 2 điểm (xem bootstrap ở lab03).
- Bỏ dấu cứu được email gõ không dấu: `E-009` "doi the visa ... gia han tu dong" → top-1 `KB-009` với điểm 34,96 so với 20,03 của hạng 2.
- Các lỗi còn lại của BM25 rơi vào: email ngắn mà ý chính không nằm ở từ khóa (`E-039` muốn gặp người, `E-041` "SSO outage"), email tiếng Nhật về hoàn tiền (`E-052`, không có bài tiếng Nhật về hoàn tiền nên cần **cross-lingual**), câu hỏi "giảm giá cho nonprofit" (`E-048`).

Phần dense và RRF **chưa chạy được trong sandbox** (không tải được model). Khi bạn chạy trên máy, kỳ vọng định tính: dense tốt hơn ở email tiếng Nhật hỏi về tài liệu chỉ có bản Việt/Anh (cross-lingual) và ở email diễn đạt khác từ khóa; BM25 tốt hơn ở mã lỗi/tên riêng ("429", "InvalidAudience", "X-Mekong-Signature"). RRF thường xấp xỉ hoặc nhỉnh hơn cái tốt nhất trong hai — nếu RRF kém hơn hẳn một nhánh, nhánh kia đang kéo xuống và bạn nên thử trọng số `weights=[0.5, 1.0]`.

## 8. Bài tập mở rộng

1. **Tách từ thật**: cài `underthesea` và thay tách âm tiết bằng `underthesea.word_tokenize(text, format="text")` (nối từ ghép bằng `_`). So sánh Hit@1 và thời gian index với bigram âm tiết. (CPU)
2. **Quét tham số BM25**: thử $k_1\in\{0.5, 1.2, 2.0\}$, $b\in\{0, 0.5, 0.75, 1\}$. Với corpus mà bài dài ngắn rất khác nhau (KB-003 dài gấp 4 lần KB-015), $b$ ảnh hưởng thế nào? (CPU)
3. **Trọng số RRF và $k$**: thử $k\in\{1, 10, 60, 200\}$ và `weights`. Giải thích bằng công thức vì sao $k$ nhỏ làm RRF gần với "lấy top-1 của từng hệ". (CPU)
4. **Cross-lingual**: đánh giá riêng 6 email tiếng Nhật, chỉ đếm là đúng nếu top-1 là bài **khác** ngôn ngữ (vi/en). Dense có tìm được bài tiếng Việt/Anh từ câu hỏi tiếng Nhật không? (CPU/GPU)
5. **Embedding lớn hơn**: thay bằng `BAAI/bge-m3` (568M tham số, 1024 chiều, không cần tiền tố). Đo tăng chất lượng so với tăng thời gian encode và bộ nhớ. (GPU 6GB vừa)

## Tài liệu tham khảo

- Robertson, S., Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond*. Foundations and Trends in IR.
- Cormack, G. V., Clarke, C. L. A., Büttcher, S. (2009). *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods*. SIGIR 2009.
- Wang, L. et al. (2024). *Multilingual E5 Text Embeddings: A Technical Report*. arXiv:2402.05672.
- Model card: https://huggingface.co/intfloat/multilingual-e5-small
- rank_bm25: https://github.com/dorianbrown/rank_bm25
