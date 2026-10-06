# Lab 03 — Rerank bằng cross-encoder đa ngữ

> Thời lượng: ~10 phút (chưa tính tải model) · Mức độ: Trung bình · Tiên quyết: Lab 01, Lab 02, Module 06 · GPU: khuyến nghị cho `bge-reranker-v2-m3`; bản mMiniLM chạy CPU được

## Mục tiêu

- Dùng `CrossEncoder` của sentence-transformers để rerank top-N ứng viên từ lab01.
- So sánh hai reranker đa ngữ: `BAAI/bge-reranker-v2-m3` (lớn, mạnh) và `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (nhỏ, nhanh).
- Đo cải thiện (R@1, R@3, MRR@5, nDCG@5) và khoảng tin cậy bootstrap ghép cặp, không chỉ nhìn một con số.
- Đo độ trễ rerank theo ms/email và dùng điểm reranker làm tín hiệu abstention cho lab04.

## Liên hệ lý thuyết

| Khái niệm | Module |
|---|---|
| Bi-encoder và cross-encoder: $f(q,d)$ chung so với $\langle E(q),E(d)\rangle$ | Module 03, 06 |
| Retrieve-then-rerank, chi phí $O(N)$ lượt forward | Module 06 |
| Bootstrap, kiểm định ghép cặp | Module 10 |
| Retrieval score làm tín hiệu confidence | Module 10 |

## 1. Vì sao rerank

Bi-encoder mã hóa query và tài liệu **riêng rẽ**: $s(q,d)=\cos(E(q),E(d))$. Nhờ vậy embedding tài liệu tính trước được và tìm kiếm nhanh, nhưng mô hình không bao giờ "nhìn thấy" query và tài liệu cùng lúc. Cross-encoder ghép cặp `[CLS] q [SEP] d [SEP]` qua toàn bộ transformer, nên attention đi qua lại giữa từng token của query và tài liệu:

$$s(q,d)=\sigma\big(w^\top h_{[CLS]}(q\oplus d)+b\big)$$

Cái giá: mỗi cặp là một lượt forward, không cache được theo tài liệu. Rerank $N$ ứng viên tốn $N$ lượt forward cho mỗi query, nên chỉ dùng cho top-N nhỏ (20–100) sau bước truy hồi rẻ.

Ước lượng chi phí: bge-reranker-v2-m3 có ~568M tham số. Một lượt forward cho chuỗi $L$ token tốn khoảng $2\cdot P\cdot L$ FLOP $=2\cdot 5{,}68\times10^8\cdot 512\approx 5{,}8\times10^{11}$ FLOP. Với 20 ứng viên là ~$1{,}2\times10^{13}$ FLOP mỗi email. RTX 4050 laptop đạt khoảng vài chục TFLOPS fp16 trên lý thuyết, hiệu suất thực tế thường chỉ một phần nhỏ, nên kỳ vọng **cỡ vài trăm ms/email trên GPU, vài giây trên CPU** (ước lượng, hãy đo thực tế). Với email (không cần real-time như chat), mức này chấp nhận được.

## 2. Chọn model (kiểm tra trên Hugging Face, 10/2026)

| Model | Tham số | Ngôn ngữ | Khi nào dùng |
|---|---|---|---|
| `BAAI/bge-reranker-v2-m3` | ~568M (nền bge-m3 / XLM-R) | Đa ngữ, gồm vi/en/ja | Mặc định khi có GPU; fp16 ~1,1 GB trọng số |
| `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | ~118M | Đa ngữ (huấn luyện trên mMARCO dịch máy) | Chạy CPU, hoặc khi GPU đã bị LLM chiếm |

## 3. Code chính

```python
import torch
from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name="BAAI/bge-reranker-v2-m3", device=None, max_length=512, fp16=True):
        self.model = CrossEncoder(model_name, device=device, max_length=max_length)
        if fp16 and torch.cuda.is_available() and str(self.model.device).startswith("cuda"):
            self.model.half()                # giảm một nửa VRAM

    def rerank(self, query, candidates, top_k=None, batch_size=16):
        """candidates: [(doc_id, text)] -> [(doc_id, score)] giảm dần."""
        if not candidates:
            return []
        hits = self.model.rank(query, [t for _, t in candidates], top_k=top_k, batch_size=batch_size)
        return [(candidates[h["corpus_id"]][0], float(h["score"])) for h in hits]
```

`CrossEncoder.rank(query, documents, top_k=...)` trả về danh sách dict `{"corpus_id", "score"}` đã sắp xếp. Với model 1 nhãn đầu ra, thư viện mặc định áp **sigmoid**, nên `score` nằm trong (0, 1). Đừng nhầm đây là xác suất đã hiệu chuẩn: nó chỉ đơn điệu với mức liên quan (lab05 sẽ hiệu chuẩn).

Luồng đánh giá: lấy top-20 từ RRF (lab01), rerank, rồi tính metric bằng các hàm của lab02.

```python
base_runs = first_stage("rrf", ids, texts, q_texts, n=20)       # BM25 + dense + RRF
reranker = Reranker(args.model)
rr_runs, top1_scores = [], []
for q, cand in zip(q_texts, base_runs):
    ranked = reranker.rerank(q, [(d, text_of[d]) for d in cand])
    rr_runs.append([d for d, _ in ranked])
    top1_scores.append(ranked[0][1])
```

### Bootstrap ghép cặp

Hai hệ chạy trên **cùng** 53 email, nên ta bootstrap trên hiệu số từng email $\delta_i=\text{RR}^{\text{rerank}}_i-\text{RR}^{\text{base}}_i$, không bootstrap riêng từng hệ (cách đó bỏ qua tương quan và cho khoảng tin cậy rộng giả tạo):

$$\hat\Delta=\frac1n\sum_i\delta_i,\qquad \hat\Delta^{*(b)}=\frac1n\sum_i\delta_{j_i^{(b)}},\ j^{(b)}_i\sim\text{Uniform}\{1..n\}$$

CI 95% là phân vị 2,5% và 97,5% của $\{\hat\Delta^{*(b)}\}$.

```python
def paired_bootstrap(a, b, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    diffs = np.asarray(b) - np.asarray(a)
    boots = [diffs[rng.integers(0, len(diffs), len(diffs))].mean() for _ in range(n_boot)]
    return float(diffs.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
```

### Điểm reranker làm tín hiệu abstention

Cuối script so sánh điểm top-1 của reranker giữa email **có** tài liệu đúng và 7 email **không** có tài liệu nào (bug máy in, roadmap, chỉ muốn gặp người...). Nếu hai phân bố tách nhau, điểm top-1 là một tín hiệu tốt để quyết định "không đủ căn cứ → không trả lời" ở lab04.

## 4. Chạy

```bash
python lab03_reranker.py                                                     # rrf + bge-reranker-v2-m3
python lab03_reranker.py --model cross-encoder/mmarco-mMiniLMv2-L12-H384-v1   # nhẹ
python lab03_reranker.py --first-stage bm25 --top-n 10 --device cpu
```

## 5. Kết quả mong đợi

Lab này **chưa chạy được trong sandbox** (không tải được model từ Hugging Face); code đã qua `py_compile` và API `CrossEncoder.rank` đã được đối chiếu với mã nguồn sentence-transformers 6.1.0. Định dạng output:

```text
Rerank 1060 cặp (query, doc) trong ...s -> ... ms/email

                            R@1      R@3    MRR@5   nDCG@5
rrf (lab01)               0.xxx    0.xxx    0.xxx    0.xxx
+ rerank                  0.xxx    0.xxx    0.xxx    0.xxx

ΔMRR@5 = +0.0xx  (95% CI bootstrap: [..., ...])
Email cải thiện: [...]
Email tệ đi   : [...]

Điểm reranker top-1: email CÓ tài liệu median=0.9x | email KHÔNG có tài liệu median=0.xx
```

Kỳ vọng định tính:

- MRR@5 của BM25 đã là 0,86 (lab01), nên trần cải thiện nhỏ. Reranker thường sửa các ca mà từ khóa đánh lừa BM25, ví dụ `E-041` ("SSO outage", BM25 đưa bài webhook lên đầu) và email tiếng Nhật về hoàn tiền `E-052` (cross-lingual).
- Với ~53 email, CI 95% của ΔMRR thường rộng cỡ ±0,05. Nếu CI chứa 0, kết luận đúng là "chưa đủ bằng chứng", không phải "reranker vô dụng".
- Có thể có vài email **tệ đi**. Hãy đọc từng ca: thường là email có 2 tài liệu liên quan và reranker đẩy bản khác ngôn ngữ lên (vẫn "đúng" với MRR nhưng làm giảm nDCG vì bài chính có $g=2$).
- Điểm top-1 của email không có tài liệu thường thấp hơn rõ rệt: dùng để đặt `ABSTAIN_THRESHOLD` cho lab04 (mặc định 0,30 khi có reranker).

## 6. Bài tập mở rộng

1. **Quét top-N**: N ∈ {5, 10, 20, 40}. Vẽ MRR@5 và ms/email theo N. Ở đâu lợi ích bão hòa? (GPU)
2. **So sánh hai reranker** trên cùng first-stage: chênh lệch chất lượng có đáng chi phí gấp ~5 lần tham số không? Đo VRAM bằng `torch.cuda.max_memory_allocated()`. (GPU)
3. **Rerank theo chunk**: dùng chunk theo cấu trúc của lab02 và rerank chunk thay vì cả bài. Max-length 512 token có cắt mất nội dung bài dài nào không? (GPU)
4. **Ensemble điểm**: kết hợp điểm reranker với hạng RRF (ví dụ $0{,}8\cdot s_{\text{rr}}+0{,}2\cdot s_{\text{rrf}}$ đã min-max). Có tốt hơn rerank thuần không? Vì sao phải chuẩn hóa ở đây mà RRF thì không cần? (GPU/CPU)
5. **ONNX/OpenVINO**: thử `CrossEncoder(..., backend="onnx")` trên CPU, đo tốc độ so với torch. (CPU)

## Tài liệu tham khảo

- Nogueira, R., Cho, K. (2019). *Passage Re-ranking with BERT*. arXiv:1901.04085.
- Chen, J. et al. (2024). *BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation*. arXiv:2402.03216.
- Bonifacio, L. et al. (2021). *mMARCO: A Multilingual Version of the MS MARCO Passage Ranking Dataset*. arXiv:2108.13897.
- Efron, B., Tibshirani, R. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
- https://huggingface.co/BAAI/bge-reranker-v2-m3 · https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 · https://sbert.net/docs/package_reference/cross_encoder/cross_encoder.html
