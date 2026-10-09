MD = "docs/07-generation-grounding-guardrails.md"
MODULE = "07"
PREFIX = "7"

FIGS = [
    ("Nói gọn: generation tốt là **tối đa hóa chất lượng", "valid-output-set",
     "Đầu ra hợp lệ là giao của các ràng buộc; khi giao rỗng, hành động đúng là không sinh mà escalate (sơ đồ minh họa)."),
    ("**Nhắc lại nhiệm vụ ở cuối.**", "prompt-anatomy",
     "Giải phẫu prompt RAG theo bảng mục 2.1: màu là mức độ tin cậy của từng khối."),
    ("Lợi khoảng 3 điểm phần trăm trong ví dụ này", "context-order-u",
     "Ví dụ mục 2.3: hạng chunk đặt tại mỗi vị trí theo hai cách sắp xếp, và xác suất trả lời đúng tương ứng."),
    ("**Ví dụ số (ước lượng).** System prompt + schema", "token-budget",
     "Ngân sách token một lượt sinh theo ví dụ mục 2.4."),
    ("$s_3$ cho thấy citation đúng ≠ thông tin đúng", "alce-citation",
     "Ví dụ ALCE của mục 3.2: câu nào được nguồn hỗ trợ, citation nào thừa."),
    ("3. **Tầng ngữ nghĩa (đắt hơn).**", "citation-check-tiers",
     "Ba tầng kiểm tra citation, tầng rẻ chạy trước."),
    ("**Ví dụ số.** Câu hỏi hướng dẫn sử dụng (how-to)", "abstention-threshold",
     "Chi phí kỳ vọng của «gửi» và «escalate» theo p̂; cùng p̂ = 0.9, câu how-to được gửi còn câu hoàn tiền bị escalate."),
    ("**Ví dụ số.** S1 (policy, 30 ngày tuổi)", "source-priority",
     "Trái: hàm độ mới với λ = 180 ngày. Phải: điểm ưu tiên của S1 và S3 tách theo thành phần."),
    ("**Ví dụ số.** Schema yêu cầu `\"escalate\": true|false`", "constrained-decoding",
     "Masking logits: token ngoài A_t nhận −∞, phần xác suất còn lại được chuẩn hóa lại tỷ lệ thuận."),
    ("Hai cái khác nhau vì mẫu số ở mỗi bước chỉ \"nhìn một bước\"", "constrained-distortion",
     "Mô hình đồ chơi (số tự chọn để minh họa): masking từng bước cho phân phối khác hẳn phân phối có điều kiện đúng trên tập chuỗi hợp lệ."),
    ("Sau sinh, chạy lại language detection trên `draft.body`", {"id": "language-flow", "mermaid": """
flowchart LR
    M[Tin nhắn mới nhất<br/>bỏ chữ ký, quoted reply] --> D[Classifier ngôn ngữ<br/>tiền xử lý]
    D --> P[Truyền ngôn ngữ vào prompt<br/>+ khung chào/kết theo ngôn ngữ]
    P --> G[Sinh draft.body]
    G --> C{Detect lại ngôn ngữ<br/>khớp không?}
    C -->|khớp| OK[Tiếp tục verify]
    C -->|lệch| R[Retry hoặc escalate]
"""}, "Ngôn ngữ đầu ra do ứng dụng quyết định và kiểm lại sau sinh, không để model tự chọn."),
    ("**Ví dụ số.** Draft ở 7.3 tách thành 4 claim", "groundedness-claims",
     "Điểm NLI của các claim trong ví dụ mục 8.1 so với ngưỡng δ."),
    ("Từ rẻ đến đắt: **rule**", "verifier-cascade",
     "Cascade verifier: chỉ gọi LLM-judge khi điểm của model nhỏ rơi vào vùng không chắc."),
    ("**Mục tiêu của kẻ tấn công**", "attack-surface",
     "Bề mặt tấn công prompt injection trong hệ thống Zendesk."),
    ("Gọi ASR (attack success rate) của mỗi lớp", "injection-layers",
     "Tỷ lệ tấn công vượt qua từng lớp phòng thủ prompt-level dưới giả định độc lập (trục log)."),
    ("Tóm lại, các lớp prompt-level (1–3) **giảm xác suất**", "defense-layers",
     "Bảy lớp phòng thủ của mục 10.4, chia theo vai trò."),
    ("Hallucination được giảm ở mọi tầng", "hallucination-stack",
     "Các tầng giảm hallucination xuyên suốt khóa học."),
]
