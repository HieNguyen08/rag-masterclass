# Lab 08 — Ingestion: làm sạch email, khử trùng lặp bằng MinHash/LSH và che PII

> Thời lượng: ~15 phút · Mức độ: Trung bình · Tiên quyết: Module 04 (mục 4–6) · GPU: không cần; chỉ thư viện chuẩn và numpy

## Mục tiêu

- Đo tác dụng của bộ làm sạch email (`common.clean_email`) lên số token đưa vào retrieval.
- Tự cài MinHash bằng họ băm phổ quát và LSH banding; kiểm chứng sai số ước lượng Jaccard và đường cong S trên dữ liệu thật.
- Cài bộ che PII nhất quán theo tài liệu (email, số điện thoại Việt/Nhật, mã số thuế, số thẻ có kiểm tra Luhn) và đo precision/recall trên một bộ kiểm thử có nhãn.

## Liên hệ lý thuyết

| Khái niệm | Module |
|---|---|
| Quoted reply, chữ ký, disclaimer | Module 04, mục 4 |
| Shingle, Jaccard, MinHash, $\Pr[h_{\min}(A) = h_{\min}(B)] = J(A,B)$ | Module 04, mục 5.2–5.3 |
| LSH banding, $P = 1 - (1 - J^r)^b$ | Module 04, mục 5.4 |
| Che PII nhất quán, ánh xạ lưu riêng | Module 04, mục 6 |

## 1. Chạy

```bash
python lab08_ingestion_dedup_pii.py                   # cả ba phần, < 1 giây
python lab08_ingestion_dedup_pii.py --part pii --show E-001
```

## 2. Phần A — làm sạch email

Số token BM25 trung bình mỗi email trước và sau `clean_email` (cắt quoted reply từ dòng header trích dẫn, bỏ dòng bắt đầu bằng `>`, cắt từ lời chào cuối thư). Kèm một kiểm tra an toàn rẻ: email nào có dấu `?` trong phần thân gốc mà mất hết `?` sau làm sạch là nghi cắt nhầm câu hỏi.

## 3. Phần B — MinHash và LSH

**Hàm băm phổ quát.** Mỗi shingle (5 ký tự, sau NFC và lowercase) được băm thành số nguyên $x$, rồi $h_i(x) = (a_i x + b_i) \bmod p$ với $p = 2^{61} - 1$ và $a_i, b_i$ ngẫu nhiên. Chữ ký của tài liệu là $\big(\min_x h_1(x), \dots, \min_x h_k(x)\big)$, và $\hat J$ là tỉ lệ vị trí trùng nhau của hai chữ ký.

**Dữ liệu kiểm thử.** 40 bài Help Center cộng 20 bản "gần trùng" sinh có kiểm soát: bỏ một câu, đổi một con số, thêm tiền tố "Cập nhật:" — mô phỏng phiên bản cũ/mới của cùng một bài hoặc macro bị sao chép rồi sửa nhẹ.

**LSH.** Chia chữ ký $k = 128$ thành $b = 16$ dải × $r = 8$ hàng; hai tài liệu là ứng viên nếu trùng ít nhất một dải.

## 4. Phần C — che PII

Mỗi giá trị PII được thay bằng một nhãn nhất quán trong tài liệu (`<PHONE_1>`, `<EMAIL_1>`...), cùng giá trị viết khác nhau ("0900 000 101" và "0900.000.101") quy về cùng nhãn nhờ chuẩn hóa khóa. Ánh xạ nhãn → giá trị thật được trả về riêng để lưu ở kho có kiểm soát truy cập, phục vụ thay ngược khi gửi email (Module 04, mục 6.3).

Số thẻ dùng thêm **kiểm tra Luhn** để giảm báo nhầm với mã đơn hàng dài.

## 5. Kết quả mong đợi (đã chạy thật)

```text
[A] Token BM25 trung bình mỗi email: 92 → 56 sau làm sạch (giảm 39%)
[A] Email mất dấu '?' sau làm sạch (nghi cắt nhầm câu hỏi): không có
[B] 60 tài liệu, 1770 cặp; sai số ước lượng Jaccard: TB -0.0010, độ lệch chuẩn 0.0136 (lý thuyết ≤ 1/(2√k) = 0.0442)
[B] Cặp có Jaccard thật ≥ 0.5: 20
[B] Cặp có Jaccard thật ≥ 0.7: 17
[B] LSH b=16, r=8: 18 cặp ứng viên (so với 1770 cặp nếu so tất cả); bắt được 17/17 cặp J ≥ 0,7
      P(thành ứng viên | J = 0.5) = 1 - (1 - J^r)^b = 0.061
      P(thành ứng viên | J = 0.7) = 1 - (1 - J^r)^b = 0.613
      P(thành ứng viên | J = 0.8) = 1 - (1 - J^r)^b = 0.947
      P(thành ứng viên | J = 0.9) = 1 - (1 - J^r)^b = 1.000
[B] Cặp gần trùng khó nhất: KB-012~KB-012-v2 J=0.70, KB-011~KB-011-v2 J=0.71, KB-019~KB-019-v2 J=0.74
[C]   lệch: 'Đơn hàng số 2026100512345 bị lỗi' → 'Đơn hàng số <CARD_1> bị lỗi'
[C] Bộ kiểm thử 14 câu: precision 0.92, recall 1.00 (TP 11, FP 1, FN 0)
[C] 60 email: còn sót email người gửi hoặc SĐT chữ ký trong 0 email sau khi che
```

Đọc kết quả:

- **Làm sạch bỏ ~39% token** — phần lớn là quoted reply và chữ ký, đúng những thứ làm nhiễu BM25 và tốn token prompt.
- **Sai số MinHash nhỏ hơn cận lý thuyết** vì phương sai là $J(1-J)/k$, lớn nhất ở $J = 0{,}5$; đa số 1.770 cặp có $J \approx 0$ nên phương sai gần 0. Muốn thấy sai số "thật", chỉ nhìn các cặp có $J$ trong khoảng 0,3–0,7.
- **LSH chỉ tạo 18 ứng viên thay vì 1.770 cặp** mà vẫn bắt đủ 17 cặp gần trùng. Theo đường cong S, cặp có $J = 0{,}7$ chỉ có 61% khả năng thành ứng viên; ở đây bắt đủ vì phần lớn cặp gần trùng có $J$ cao hơn 0,7. Cặp khó nhất ($J = 0{,}70$) là cặp bỏ một câu dài — với kho thật, đó chính là loại "phiên bản mới của bài" dễ bị lọt nhất.
- **Một báo nhầm số thẻ:** mã đơn hàng 13 chữ số tình cờ qua kiểm tra Luhn (khoảng 1/10 chuỗi số ngẫu nhiên qua được). Luhn giảm báo nhầm chứ không loại bỏ được; cách sửa thực tế là yêu cầu thêm ngữ cảnh ("thẻ", "card", "visa") gần chuỗi số.
- **Tên người chưa được che.** Ở ví dụ `--show E-001`, "Trần Thị Mai" vẫn còn: regex không nhận được tên. Kiểm tra "0 email còn sót" chỉ đo email người gửi và số điện thoại chữ ký — đừng đọc nó thành "đã che hết PII".

## 6. Bài tập mở rộng

1. **Tên người.** Thêm bước che tên dựa trên vị trí (dòng ngay sau lời chào cuối thư) và một NER (ví dụ `underthesea` cho tiếng Việt); đo trên 20 email có gán nhãn tay.
2. **Chọn $(b, r)$.** Với $k = 128$, thử $(32, 4)$, $(16, 8)$, $(8, 16)$; với mỗi cấu hình đếm số ứng viên và số cặp $J \ge 0{,}7$ bị bỏ sót. Vẽ đường cong S lý thuyết chồng lên tỉ lệ bắt được thực nghiệm.
3. **Thứ tự che PII và khử trùng lặp.** Hai email giống hệt nhưng khác số điện thoại: chạy MinHash trước và sau khi che PII, giải thích vì sao Module 04 đặt che PII *trước* khử trùng lặp trong pipeline.
4. **Số thẻ có ngữ cảnh.** Sửa mẫu CARD để chỉ nhận chuỗi số khi trong 30 ký tự xung quanh có từ khóa thẻ; chạy lại bộ kiểm thử và thêm 5 câu khó của riêng bạn.

## Tài liệu tham khảo

- Broder, A. Z. (1997). *On the resemblance and containment of documents.* Compression and Complexity of Sequences.
- Leskovec, J., Rajaraman, A., Ullman, J. D. *Mining of Massive Datasets*, chương 3 (Finding Similar Items). http://www.mmds.org/
- Microsoft Presidio (phát hiện và che PII): https://microsoft.github.io/presidio/
