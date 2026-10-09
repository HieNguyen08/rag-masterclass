# Lab 07 — Fine-tune embedding cho retrieval: InfoNCE, hard negative và đánh giá trung thực

> Thời lượng: ~15 phút (chế độ numpy) · Mức độ: Nâng cao · Tiên quyết: Lab 01, Module 03 (mục 3), Module 09 (mục 2–3), Module 10 (mục 6) · GPU: không cần cho chế độ numpy; chế độ `--st-finetune` cần GPU 6 GB

## Mục tiêu

- Tạo dữ liệu huấn luyện tổng hợp (query giả, bài đúng) từ cấu trúc Help Center và đào hard negative bằng BM25.
- Cài InfoNCE bằng numpy, tự viết gradient, huấn luyện một **adapter tuyến tính** phía query trên embedding cố định.
- Đánh giá trên **email thật** (không dùng để huấn luyện) với Recall@3, MRR@5 và khoảng tin cậy bootstrap ghép cặp.
- Biết cách chuyển sang fine-tune toàn bộ model bằng `sentence-transformers` khi có GPU.

## Liên hệ lý thuyết

| Khái niệm | Module |
|---|---|
| InfoNCE, nhiệt độ $\tau$, in-batch negatives, hard negative và false negative | Module 03, mục 3 |
| Dữ liệu fine-tune từ Zendesk, sinh câu hỏi tổng hợp | Module 09, mục 2 |
| `MultipleNegativesRankingLoss`, đánh giá trước/sau | Module 09, mục 3 |
| Bootstrap ghép cặp | Module 10, mục 6.4 |

## 1. Chạy

```bash
python lab07_finetune_embedding.py                 # CPU, ~1 giây, chỉ numpy
python lab07_finetune_embedding.py --base st       # adapter trên multilingual-e5-small (cần sentence-transformers)
pip install datasets                               # cho chế độ dưới
python lab07_finetune_embedding.py --st-finetune   # fine-tune toàn bộ e5-small, fp16, batch 32 (GPU 6 GB)
```

## 2. Toán: adapter tuyến tính với InfoNCE

Giữ embedding tài liệu $\mathbf{d}_j$ cố định (không phải index lại kho — đúng chi tiết thực tế mà RAG gốc của Lewis et al. cũng chọn, Module 02 mục 5.2), học ma trận $W$ cho query:

$$
s(q, d_j) = \frac{(W\mathbf{q})^\top \mathbf{d}_j}{\tau}, \qquad
\mathcal{L} = -\frac{1}{B}\sum_{i=1}^{B}\log\frac{e^{s(q_i, d_{+(i)})}}{\sum_{j} e^{s(q_i, d_j)}} + \lambda\,\|W - I\|_F^2 .
$$

Đặt $p_{ij}$ là xác suất softmax và $y_{ij}$ là one-hot của bài đúng. Như gradient cross-entropy ở Module 01 (mục 5.1), đạo hàm theo vector query đã biến đổi là "dự đoán trừ sự thật", lan về $W$ bằng quy tắc chuỗi:

$$
\frac{\partial \mathcal{L}}{\partial (W\mathbf{q}_i)} = \frac{1}{\tau}\sum_j (p_{ij} - y_{ij})\,\mathbf{d}_j, \qquad
\frac{\partial \mathcal{L}}{\partial W} = \frac{1}{B}\sum_i \frac{\partial \mathcal{L}}{\partial (W\mathbf{q}_i)}\,\mathbf{q}_i^\top + 2\lambda (W - I).
$$

Khởi tạo $W = I$ nghĩa là bắt đầu đúng bằng model gốc; số hạng $\lambda\|W - I\|^2$ giữ adapter không đi quá xa — cùng tinh thần với ràng buộc KL của RLHF/DPO (Module 01, mục 7) và với việc LoRA chỉ học một phần dư nhỏ (Module 09, mục 7).

Với kho 40 bài, softmax chạy trên **toàn bộ** tài liệu: mọi hard negative đều có mặt. Với kho thật hàng trăm nghìn chunk, ta quay về in-batch negatives cộng một hard negative đào sẵn cho mỗi cặp — đúng thứ `MultipleNegativesRankingLoss` làm trong chế độ `--st-finetune`.

## 3. Kết quả mong đợi (đã chạy thật, chế độ numpy mặc định)

```text
Cặp huấn luyện tổng hợp: 198 | email đánh giá: 53
Loss huấn luyện theo epoch: 0.047 0.015 0.011 0.009 0.008 0.007 0.006 0.006 0.005 0.005
MRR@5 trên chính tập huấn luyện: 1.000 → 1.000
Recall@3 (email): 0.726 → 0.736
MRR@5    (email): 0.851 → 0.884
Chênh MRR@5: +0.033, CI 95% bootstrap ghép cặp [+0.005, +0.071]
Email tốt lên / xấu đi / không đổi: 5 / 1 / 47
```

Đọc kết quả — đây là phần quan trọng nhất của lab:

- **Tập huấn luyện quá dễ.** MRR trên query tổng hợp đã là 1,000 *trước* khi huấn luyện: câu lấy từ chính bài viết khớp từ vựng hoàn hảo với bài đó. Loss vẫn giảm (adapter làm biên rộng hơn) nhưng tín hiệu học rất ít. Đây đúng là cảnh báo của Module 09: dữ liệu tổng hợp phải *giống câu hỏi thật* (ngôn ngữ đời thường, không dấu, lẫn ngôn ngữ, có lỗi chính tả), nếu không model học cách giải một bài khác.
- **Cải thiện nhỏ nhưng thật.** MRR@5 tăng 0,033 với khoảng tin cậy không chứa 0, nhưng chỉ 6/53 email thay đổi thứ hạng (5 tốt lên, 1 xấu đi). Báo cáo "MRR tăng 4%" mà không kèm hai con số này là thiếu trung thực.
- **Embedding băm là thuần từ vựng.** Adapter tuyến tính trên nó không thể học quan hệ xuyên ngôn ngữ ("mật khẩu" ↔ "password") vì hai từ rơi vào hai chiều băm không liên quan. Chạy lại với `--base st` để thấy khác biệt khi nền là embedding ngữ nghĩa đa ngữ.

## 4. Chế độ `--st-finetune`

```python
loss = losses.MultipleNegativesRankingLoss(model)        # InfoNCE + in-batch negatives + hard negative
args = SentenceTransformerTrainingArguments(
    output_dir="out/e5-small-cs", num_train_epochs=3, per_device_train_batch_size=32,
    learning_rate=2e-5, warmup_ratio=0.1, fp16=True,
    batch_sampler=BatchSamplers.NO_DUPLICATES)             # tránh cùng một bài xuất hiện hai lần trong batch
```

`NO_DUPLICATES` quan trọng với in-batch negatives: nếu hai query trong batch có cùng bài đúng, bài đó vừa là positive của query này vừa bị coi là negative của query kia — một dạng false negative do chính cách lấy batch (Module 03, mục 3.5). Với e5-small (~118M tham số), fp16, batch 32, chuỗi ≤ 256 token, bộ nhớ nằm thoải mái trong 6 GB (ước lượng; đo bằng `nvidia-smi`).

## 5. Bài tập mở rộng

1. **Query giống thật hơn.** Sinh 3 câu hỏi cho mỗi bài bằng LLM local (vi/en/ja, có câu không dấu); thêm các cặp (email đã giải quyết, bài agent dán link) nếu có. Huấn luyện lại và so khoảng tin cậy với dữ liệu hiện tại.
2. **False negative.** Với mỗi hard negative BM25, kiểm tra thủ công 20 cặp: bao nhiêu "negative" thật ra cũng trả lời được câu hỏi? Thử lọc negative có điểm quá gần positive (ví dụ trên 95% điểm positive) và đo ảnh hưởng.
3. **Nhiệt độ.** Chạy với $\tau \in \{0{,}01; 0{,}05; 0{,}2\}$; giải thích kết quả bằng phân tích trọng số hard negative ở Module 03 (mục 3.4).
4. **Không làm hỏng chỗ khác.** Thêm một tập kiểm tra "câu hỏi ngoài miền" (10 câu không liên quan tới kho) và kiểm tra điểm cao nhất của chúng có tăng sau fine-tune không — nếu tăng, ngưỡng abstention ở Lab 04/05 phải hiệu chuẩn lại.

## Tài liệu tham khảo

- Oord, A. van den, Li, Y., Vinyals, O. (2018). *Representation Learning with Contrastive Predictive Coding* (InfoNCE). arXiv:1807.03748.
- Henderson, M. et al. (2017). *Efficient Natural Language Response Suggestion for Smart Reply* (in-batch negatives). arXiv:1705.00652.
- Wang, L. et al. (2024). *Multilingual E5 Text Embeddings: A Technical Report.* arXiv:2402.05672.
- Sentence Transformers — Training Overview: https://sbert.net/docs/sentence_transformer/training_overview.html
