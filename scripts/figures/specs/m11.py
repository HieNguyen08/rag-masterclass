MD = "docs/11-production-va-quy-mo.md"
MODULE = "11"
PREFIX = "11"

FIGS = [
    ("Một tháng 30 ngày: ~1,17 tỷ token vào", "tokens-per-step",
     "Token vào/ra của từng bước trong một lượt chạy theo bảng giả định của mục 1.3."),
    ("Tức là khi lưu lượng vượt ~2,5 lần hiện tại", "cost-options",
     "Trái: chi phí/tháng của các phương án ở mục 1.4–1.5. Phải: điểm hòa vốn giữa API (phương án C) và self-host hai GPU với giá thuê giả định."),
    ("| Tỷ lệ ticket bị lọc trước LLM", "cost-sensitivity",
     "Phân tích độ nhạy của mục 1.6: số lượt chạy mỗi ticket là thừa số ảnh hưởng mạnh nhất."),
    ("**Vai trò của GQA**", "kv-cache-length",
     "Bộ nhớ KV cache theo độ dài sequence cho hai cấu hình Qwen3 (đọc từ config.json) và một cấu hình giả định không dùng GQA."),
    ("Bài học: **với model 30B+ trên một GPU 80 GB", "gpu-memory-budget",
     "Bảng mục 2.3 dưới dạng ngân sách VRAM: phần còn lại sau trọng số và overhead quyết định số sequence 8k chạy đồng thời."),
    ("Hệ quả phụ quan trọng: block có thể **chia sẻ**", "paged-attention",
     "Ví dụ mục 2.4: cấp phát liền mạch theo max_seq_len lãng phí gần 2/3 slot; PagedAttention chỉ lãng phí nửa block cuối mỗi sequence."),
    ("Với $\\alpha = 0{,}7$, $k = 4$", "speculative-decoding",
     "Số token kỳ vọng mỗi lượt verify của speculative decoding theo xác suất chấp nhận α và số token nháp k."),
    ("SLO đề xuất: **95% ticket đủ điều kiện", "latency-budget",
     "Latency budget của mục 3.2 (giả định): thanh là p50, vạch là p95 của từng bước."),
    ("Mỗi worker là một coroutine async", "utilization-wait",
     "Minh họa định tính bằng hàng đợi M/M/1: thời gian lưu trú tăng rất nhanh khi mức sử dụng tiến tới 1 — lý do giữ ρ quanh 50% ở giờ đỉnh."),
    ("Ví dụ $t_0 = 1$ s:", "backoff-jitter",
     "Exponential backoff với full jitter (t₀ = 1 s): mỗi lần thử chờ ngẫu nhiên trong [0, t₀·2ⁿ], nên các client tản ra thay vì thử lại cùng lúc."),
    ("Kết luận thực dụng của mình: **không dùng semantic cache", "semantic-cache",
     "Lợi ích ròng của semantic cache theo tỉ lệ trả lời sai khi trúng cache, với tỉ lệ trúng h = 0,3 và ba mức chi phí của một câu trả lời sai."),
    ("Với int8 scalar quantization: vector còn ~205 MB", "index-memory",
     "Ước lượng bộ nhớ index của mục 4.4 cho ba kiểu lưu vector."),
    ("Đánh đổi: latency tăng cho phần bị leo thang", "cascade-cost",
     "Chi phí kỳ vọng của cascade theo tỉ lệ q lượt model nhỏ đạt chuẩn (số liệu mục 8.2)."),
    ("Cụm \"đổi mật khẩu\" có nhiều ticket nhất", "gap-impact",
     "Xếp ưu tiên lỗ hổng kho tri thức theo tác động (ví dụ giả định của mục 9.4)."),
]
