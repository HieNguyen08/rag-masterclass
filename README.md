# RAG Masterclass

**LLM & RAG từ nền tảng đến hệ thống production — xoay quanh bài toán AI tư vấn khách hàng qua Zendesk.**

Website: https://hienguyen08.github.io/rag-masterclass/

Giáo trình tiếng Việt ~8–9 giờ học: 13 module lý thuyết (toán + trực giác + ví dụ số + liên hệ thực tế) và 5 lab thực hành.

| Phần | Module |
|---|---|
| I · Nền tảng | 00 Tổng quan & bài toán · 01 Nền tảng LLM · 02 Giới hạn LLM & RAG |
| II · Pipeline RAG | 03 Embedding · 04 Ingestion & chunking · 05 Retrieval · 06 Query & reranking · 07 Generation & guardrails |
| III · Nâng cao | 08 Kiến trúc nâng cao & Agentic RAG · 09 Fine-tuning cho RAG · 10 Đánh giá, confidence & escalation |
| IV · Hệ thống | 11 Production & quy mô · 12 Capstone: hệ thống AI CS Zendesk |
| Thực hành | Lab 01–05 (`docs/labs/`) |

## Chạy website trên máy

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-docs.txt
mkdocs serve            # mở http://127.0.0.1:8000
```

## Cấu trúc repo

```
docs/                 nội dung website (Markdown)
  00-…md … 12-…md     các module lý thuyết
  labs/               lab thực hành (.md + .py) và dữ liệu mẫu
  javascripts/        cấu hình MathJax
  stylesheets/        CSS bổ sung
notes/                style guide và syllabus dùng khi viết nội dung
mkdocs.yml            cấu hình MkDocs Material
.github/workflows/    tự build và deploy lên GitHub Pages khi push vào main
```

## Đóng góp

Phát hiện sai sót về công thức, trích dẫn hay thông tin lỗi thời? Hãy mở issue hoặc pull request. Mỗi trang có nút chỉnh sửa dẫn thẳng tới file nguồn.

## Lưu ý

- Số liệu quy mô công ty trong bài là giả định để học.
- Thông tin model, thư viện, giá API và pháp lý được cập nhật tính đến 10/2026 và thay đổi nhanh.
- Dữ liệu trong `docs/labs/data/` là dữ liệu hư cấu, được sinh bằng `generate_data.py`.
