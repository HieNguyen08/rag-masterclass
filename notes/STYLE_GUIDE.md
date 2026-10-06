# STYLE GUIDE — RAG Masterclass (dành cho người viết module)

Khóa học: **"LLM & RAG từ nền tảng đến hệ thống production — xoay quanh bài toán AI tư vấn khách hàng qua Zendesk"**.
Tổng thời lượng mục tiêu: ~8–9 giờ học (đọc + nghĩ + làm bài). Mỗi module là một "buổi giảng" 35–55 phút.

## 1. Người học
- Hiếu, sinh viên năm cuối Khoa học Máy tính (HCMUT), đã làm đồ án tốt nghiệp NLU/NER/RAG tự host (không dùng API LLM bên thứ ba), từng làm Rails full-stack.
- Sắp bước vào dự án nền tảng LLM với mentor: vLLM, FastAPI (healthcheck, log, history, observability), SSE streaming, gRPC, MCP server, LangChain/LangGraph harness (agent harness + evaluation harness), system design.
- Máy cá nhân: Windows + WSL2 + Docker, GPU RTX 4050 **6 GB VRAM** (dùng cho lab: chỉ model nhỏ / quantized).
- Đã biết lập trình, đại số tuyến tính và xác suất ở mức đại học → **được phép đi sâu toán**, nhưng mỗi công thức phải có trực giác + ví dụ số.

## 2. Ngôn ngữ & giọng văn
- Viết **tiếng Việt**, giữ nguyên thuật ngữ kỹ thuật tiếng Anh (embedding, retriever, reranker, chunk, KV cache…); lần đầu xuất hiện thì giải thích ngắn.
- Giọng giảng viên/chuyên gia: rõ ràng, có chính kiến kỹ thuật ("trong thực tế mình khuyên…"), không màu mè.
- Không dùng emoji. Không viết kiểu marketing.

## 3. Định dạng
- Markdown chuẩn GitHub. Toán: `$...$` inline và `$$...$$` block (LaTeX, render được trên GitHub/Obsidian/VS Code).
- Sơ đồ: dùng khối ```mermaid khi giúp hiểu luồng/kiến trúc.
- Code: Python là chính, ngắn, chạy được, có comment tiếng Việt; ghi rõ thư viện/phiên bản nếu quan trọng.
- Bảng để so sánh lựa chọn (trade-off).

## 4. Cấu trúc bắt buộc của mỗi module
```
# Module NN — Tên module
> Thời lượng: ~X phút · Mức độ: Cơ bản/Trung bình/Nâng cao · Tiên quyết: Module ...

## Mục tiêu học tập        (4–6 gạch đầu dòng, đo được)
## 1. ... 2. ... (các phần nội dung)
   - Mỗi kỹ thuật: Vấn đề nó giải quyết → Ý tưởng/trực giác → Toán (định nghĩa, công thức, dẫn xuất ngắn) → Ví dụ số tính tay → Code minh họa (nếu hợp) → Trade-off / khi nào KHÔNG dùng.
   - Ít nhất 1 "Liên hệ Zendesk" cho mỗi phần lớn (xem mục 5).
## Lỗi thường gặp & cách xử lý      (bảng: Triệu chứng | Nguyên nhân gốc | Cách xử lý)
## Tóm tắt (cheat-sheet)
## Câu hỏi tự kiểm tra / phỏng vấn  (8–12 câu, có đáp án gợi ý gọn trong <details>)
## Bài tập thực hành               (2–4 bài, ghi rõ chạy được trên GPU 6GB hay cần API)
## Tài liệu tham khảo              (paper có arXiv ID + năm; docs chính thức; chỉ dùng link đã kiểm tra)
```
Độ dài mục tiêu: **mỗi module 7.000–11.000 từ** (~45–70 KB markdown). Sâu, đầy đủ, không độn chữ.

## 5. Case study xuyên suốt (giả định thống nhất — dùng đúng các con số này)
Ticket nội bộ: *"[CS] AI tư vấn (Zendesk) sản phẩm cho khách hàng"* — Priority High, Group CS. Yêu cầu:
1. **AI trả lời email cho khách hàng thông qua Zendesk.**
2. **AI thông báo cho team CS vào xử lý khi khách hàng có yêu cầu (muốn gặp người) hoặc khi AI tự đánh giá cần người can thiệp.**

Giả định quy mô (ghi rõ là giả định để học, không phải số liệu thật của công ty):
- Doanh nghiệp phần mềm B2B (SaaS), khách hàng Việt Nam + Nhật + quốc tế → email **tiếng Việt, tiếng Anh, tiếng Nhật**, thường lẫn ngôn ngữ, có chữ ký, quoted reply, đính kèm ảnh chụp màn hình.
- ~**1.500 ticket mới/ngày**, peak gấp ~3 lần giờ cao điểm; mỗi ticket trung bình 3–4 lượt trao đổi.
- Nguồn tri thức: ~**800 bài Help Center**, ~**300 macro** (mẫu trả lời) của CS, **~200.000 ticket đã giải quyết** (lịch sử), tài liệu sản phẩm/API, release notes, chính sách giá/hoàn tiền/SLA. Một phần dữ liệu thay đổi hằng tuần.
- Ràng buộc: không bịa chính sách (giá, hoàn tiền, cam kết SLA), không lộ dữ liệu khách hàng khác (multi-tenant), có PII trong email, email có thể chứa **prompt injection**.
- Mục tiêu kinh doanh: giảm thời gian phản hồi đầu tiên (FRT), tăng tỷ lệ tự giải quyết (deflection/automation rate), giữ CSAT; mọi trường hợp rủi ro → chuyển người (escalate) qua internal note + đổi group/assignee/tag + thông báo (Slack/email).
- Lộ trình triển khai an toàn: giai đoạn 1 AI chỉ soạn **draft (internal note)** cho agent duyệt → giai đoạn 2 tự gửi với nhóm intent rủi ro thấp → mở rộng dần dựa trên số liệu đánh giá.
- Stack tham chiếu (phù hợp người học): Python, FastAPI, LangGraph, vLLM (self-host) hoặc API LLM thương mại, Postgres + pgvector hoặc Qdrant, Redis, Zendesk API (Tickets, Comments, Help Center, Webhooks/Triggers), OpenTelemetry.

## 6. Nguồn & độ chính xác
- **Dùng WebSearch/WebFetch** để kiểm tra: các paper (đúng tên, tác giả, năm, arXiv ID), con số benchmark, tính năng/phiên bản thư viện, API Zendesk. Tri thức về model/tool thay đổi nhanh — ghi "tính đến 10/2026" khi nêu hiện trạng.
- Không bịa con số. Nếu là ước lượng thì ghi "ước lượng" và cách tính.
- Bản quyền: **diễn đạt lại bằng lời của mình**, không chép đoạn văn từ nguồn; trích dẫn nguyên văn tối đa 1 câu ngắn (<15 từ)/nguồn và rất hạn chế.
- Toán phải đúng ký hiệu và nhất quán trong module; định nghĩa mọi ký hiệu trước khi dùng.

## 7. Tên file
`/home/claude/rag-masterclass/NN-ten-module-khong-dau.md` (đúng tên được giao). Viết theo từng phần (Write rồi Edit/append) để tránh file quá dài một lần.
