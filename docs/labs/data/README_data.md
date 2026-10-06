# Dữ liệu mẫu (hư cấu) — Mekong Cloud

Sinh bởi `generate_data.py`. Mọi tên người/công ty/email/số điện thoại/giá/chính sách đều là BỊA.

## help_center.jsonl
| field | ý nghĩa |
|---|---|
| id | `KB-xxx` |
| lang | `vi` / `en` / `ja` |
| category | account, billing, api, ... |
| title, body | nội dung; `### ` đánh dấu tiêu đề mục (dùng cho chunking theo cấu trúc) |
| url, updated_at, visibility, product | metadata |

## emails.jsonl
| field | ý nghĩa |
|---|---|
| id | `E-xxx` |
| lang | `vi` / `en` / `ja` / `mixed` |
| subject, body | email thô: có chữ ký, quoted reply, disclaimer, có email không dấu |
| intent | nhãn intent |
| relevant_doc_ids | các bài trả lời được câu hỏi (cùng ngôn ngữ đứng trước) — rỗng nếu không có |
| needs_human | nhãn vàng: có cần chuyển người không |
| wants_human | khách chủ động muốn gặp người |
| sensitive_topic | refund / pricing / cancellation / legal / security_incident / outage / null |
| has_injection | email có prompt injection |

## judge_set.jsonl
Draft trả lời mẫu + nhãn người duyệt (`human_label` 1 = chấp nhận gửi) cho lab05.
