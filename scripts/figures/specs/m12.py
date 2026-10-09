MD = "docs/12-capstone-thiet-ke-he-thong-zendesk.md"
MODULE = "12"
PREFIX = "12"

FIGS = [
    ("Tức tương đương ~11 agent-ngày công mỗi ngày", "business-value",
     "Ước lượng lạc quan của mục 2 về thời gian agent tiết kiệm được ở giai đoạn 2 (giả định 8 phút/ticket, 25% tự trả lời, 50% có draft giúp tiết kiệm 3 phút)."),
    ("Ngưỡng này chỉ có nghĩa khi $\\hat p$ được hiệu chuẩn tốt", "cost-threshold",
     "Ngưỡng gửi theo chi phí của mục 8.2: khi chi phí của một câu trả lời sai tăng, ngưỡng tiến về 1; hai đường ứng với hai mức chi phí chuyển người (giả định)."),
    ("| **3. Agentic có tool** (tùy chọn)", "rollout-gates",
     "Các giai đoạn rollout và tiêu chí chuyển giai đoạn theo bảng mục 10."),
    ("- **Kích thước mẫu**: để ước lượng risk 2%", "review-sample-size",
     "Số câu trả lời đã gửi cần review để ước lượng tỉ lệ sai với sai số mong muốn (xấp xỉ chuẩn, CI 95%)."),
    ("| Chấp nhận của team CS thấp |", "risk-matrix",
     "Bảng rủi ro của mục 11 sắp theo khả năng và tác động."),
]
