# Lab 09 — Xử lý query cho email: làm sạch, ngữ cảnh hội thoại, tách câu hỏi, PRF và HyDE

> Thời lượng: ~15 phút · Mức độ: Trung bình · Tiên quyết: Lab 01, Module 04 (mục 4), Module 06 (mục 2–6) · GPU: không cần; phần HyDE và tách câu hỏi bằng LLM cần LLM local hoặc API

## Mục tiêu

- Đo từng phép biến đổi query của Module 06 trên cùng một retriever (BM25) và cùng tập email có nhãn, thay vì tin rằng "kỹ thuật nào cũng giúp".
- Thấy cụ thể một trường hợp làm sạch email làm *mất* ngữ cảnh, và cách ngưng tụ hội thoại khôi phục nó.
- Dựng tập kiểm thử có kiểm soát (email ghép hai câu hỏi) để đo đúng thứ mà kỹ thuật tách câu hỏi nhắm tới.
- Chạy HyDE và tách câu hỏi bằng LLM thật khi có LLM.

## Liên hệ lý thuyết

| Khái niệm | Module |
|---|---|
| Làm sạch quoted reply, chữ ký | Module 04, mục 4 |
| Ngưng tụ hội thoại (query condensation) | Module 06, mục 3 |
| Multi-query, RRF | Module 06, mục 4; Module 05, mục 4 |
| HyDE, pseudo-relevance feedback | Module 06, mục 5 |
| Tách câu hỏi (decomposition) | Module 06, mục 6 |

## 1. Chạy

```bash
python lab09_query_rewriting.py                    # không cần LLM, < 2 giây
python lab09_query_rewriting.py --show E-037       # xem query sau làm sạch và các câu hỏi tách được
python lab09_query_rewriting.py --show E-007+E-008 # một email ghép hai câu hỏi
LLM_MODE=openai LLM_BASE_URL=http://localhost:11434/v1 LLM_MODEL=qwen3:4b-instruct-2507-q4_K_M \
    python lab09_query_rewriting.py --hyde         # thêm HyDE và tách câu hỏi bằng LLM
```

## 2. Các phép biến đổi

| Hệ thống | Làm gì |
|---|---|
| Nguyên văn | Tiêu đề + toàn bộ thân email, kể cả quoted reply và chữ ký |
| Làm sạch | `email_query` của Lab 01 + bỏ cụm lời chào đầu dòng |
| Ngữ cảnh khi email quá ngắn | Nếu thân email mới dưới 30 từ ("Vẫn không được"), nối thêm thư cũ được trích dẫn — phiên bản thô của ngưng tụ hội thoại; bản LLM viết lại thành một câu hỏi độc lập |
| Tách câu hỏi (quy tắc) + RRF | Giữ các câu có dấu hỏi hoặc từ để hỏi, truy xuất từng câu, gộp bằng RRF |
| Giữ query gốc + câu hỏi tách + RRF | Như trên nhưng thêm chính query đầy đủ vào danh sách gộp |
| PRF | Nối 2 câu đầu của tài liệu top-1 vào query rồi gộp với kết quả gốc — tổ tiên không cần LLM của HyDE |
| HyDE (cần LLM) | LLM viết một "bài Help Center giả" trả lời câu hỏi, truy xuất bằng bài giả đó, gộp với query gốc |

**Tập kiểm thử ghép.** Bộ dữ liệu mẫu gần như không có email hỏi nhiều chủ đề khác nhau (đa số email có hai "tài liệu đúng" là hai bản ngôn ngữ của *cùng* một bài). Để đo kỹ thuật tách câu hỏi, lab ghép từng cặp email tiếng Việt khác chủ đề thành một email ("... Ngoài ra: ...") với hai tài liệu đúng — 15 email ghép.

## 3. Kết quả mong đợi (đã chạy thật, không LLM)

```text
Hệ thống                                      R@3  MRR@5    ghép 2 câu hỏi: R@3
nguyên văn (có quoted reply, chữ ký)        0.726  0.869                  0.833
làm sạch (Module 04)                        0.726  0.859                  0.833
làm sạch + ngữ cảnh khi email quá ngắn      0.726  0.869                  0.833
  + tách câu hỏi (quy tắc) + RRF            0.698  0.828                  0.467
  + giữ query gốc + câu hỏi tách + RRF      0.717  0.847                  0.533
  + PRF (top-1, 2 câu)                      0.726  0.862                  0.800
(53 email thật có tài liệu đúng; 15 email ghép hai câu hỏi)
```

Đọc kết quả — phần lớn là kết quả *âm*, và đó là bài học:

- **Làm sạch giảm MRR nhẹ vì một email**: `E-040` chỉ viết "Vẫn không được. Mình muốn gặp người thật...", còn chủ đề (đồng bộ Lazada) nằm trong thư cũ được trích dẫn. Cắt quoted reply là đúng cho chunking và cho prompt, nhưng với *query* thì làm mất ngữ cảnh. Bước ngưng tụ đưa MRR về lại 0,869. Đây chính là lý do Module 06 đặt ngưng tụ hội thoại trước retrieval.
- **Tách câu hỏi bằng quy tắc làm hại**, kể cả trên tập ghép được dựng riêng cho nó (0,833 → 0,467). Lý do thấy ngay khi chạy `--show E-007+E-008`: câu hỏi tách ra mất từ khóa chủ đề nằm ở tiêu đề và ở các câu không có dấu hỏi ("Bao lâu thì có ạ?" không còn biết là hỏi về cái gì). Với BM25, một query dài chứa cả hai chủ đề vốn đã cộng điểm cho cả hai nhóm tài liệu, nên không có gì để sửa. Giữ query gốc trong danh sách gộp đỡ được một phần (0,533) nhưng vẫn thua.
- **Khi nào tách câu hỏi mới có ích:** khi câu hỏi con được viết lại *đầy đủ ngữ cảnh* (việc của LLM, không phải regex), và với retriever **một vector** (dense) — vì embedding của một email hai chủ đề là trung bình của hai hướng, gần cả hai mà không thật gần hướng nào (Module 06, mục 6). Bài tập 1 và 2 kiểm chứng điều này.
- **PRF gần như trung tính**: với kho 40 bài, tài liệu top-1 thường đã đúng nên mở rộng không thêm gì; khi top-1 sai, mở rộng kéo query lệch thêm (query drift).

## 4. Bài tập mở rộng

1. **LLM thật.** Chạy `--hyde` với một model 3–4B local. So hàng "query gốc + câu hỏi tách bằng LLM" với hàng quy tắc trên tập ghép. Đọc 5 câu hỏi LLM tách ra: chúng có tự đủ nghĩa không?
2. **Dense.** Thay `Searcher.run` bằng retriever dense của Lab 01 (multilingual-e5-small). Chạy lại bảng: tách câu hỏi có còn hại trên tập ghép không? Giải thích bằng lập luận "trung bình hai hướng".
3. **Ngưỡng ngưng tụ.** Thay ngưỡng 30 từ bằng bộ phát hiện tin nhắn tiếp nối ("vẫn", "still", "như trên", "まだ") và đo lại; liệt kê các email bị thay đổi.
4. **Chi phí.** HyDE thêm một lượt gọi LLM trước retrieval. Đo độ trễ thêm mỗi email và đặt nó vào ngân sách latency của Module 06 (mục 11) — có đáng không với mức cải thiện bạn đo được?

## Tài liệu tham khảo

- Gao, L., Ma, X., Lin, J., Callan, J. (2022). *Precise Zero-Shot Dense Retrieval without Relevance Labels* (HyDE). arXiv:2212.10496.
- Rackauckas, Z. (2024). *RAG-Fusion: a New Take on Retrieval-Augmented Generation.* arXiv:2402.03367.
- Cormack, G. V., Clarke, C. L. A., Büttcher, S. (2009). *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods.* SIGIR 2009.
