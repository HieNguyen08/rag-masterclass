MD = "docs/02-gioi-han-llm-va-rag.md"
MODULE = "02"
PREFIX = "2"

FIGS = [
    ("- **Extrinsic hallucination:** đầu ra chứa thông tin", "intrinsic-extrinsic",
     "Trục 1 phân loại hallucination theo quan hệ với nguồn được cung cấp, minh họa bằng ví dụ chính sách hoàn tiền của mục 1.1."),
    ("- Nếu phạt sai gấp 9 lần ($c = 9$)", "abstain-threshold",
     "Mô hình chấm điểm của mục 1.2: trả lời có lợi khi đường nằm trên 0, tức p > c/(1 + c). Với c = 0 thì luôn nên đoán."),
    ("Ở một số cấu hình, đặt đáp án ở giữa", "lost-in-middle",
     "Sơ đồ định tính của hiệu ứng lost in the middle (Liu et al., 2024): đường cong chỉ thể hiện hình dạng, không phải số liệu trong bài báo."),
    ("Độ dài tăng 80 lần nhưng chi phí prefill", "prefill-flops",
     "Ước lượng FLOPs prefill của mục 3.3 cho cấu hình kiểu 7B: phần attention bậc hai chiếm ~14% ở 8K token và ~93% ở 640K token."),
    ("Prompt caching (ví dụ API của Anthropic", "context-cost",
     "Chi phí input mỗi ngày theo bảng của mục 3.3 (giá và lưu lượng đều là giả định để học)."),
    ("| Rủi ro lỗi do truy xuất sai |", "strategy-matrix",
     "Ma trận quyết định của mục 4.3 dưới dạng bản đồ màu (xanh: phù hợp, đỏ: kém); cột RAG được viền đậm."),
    ("- Tích: $0.696 \\times 0.568 = 0.395$.", "rag-seq-vs-token",
     "Ví dụ tính tay của mục 5.3: phân phối retriever, xác suất generator theo từng tài liệu, và xác suất câu trả lời dưới hai cách lấy biên."),
    ("Gradient theo điểm: $(0.966 - 0.629", "realm-posterior",
     "Ví dụ số của mục 5.4: gradient theo điểm retriever bằng posterior trừ prior, đẩy bảng giá hiện hành lên và bài blog cũ xuống."),
    ("**Vì sao hiệu quả.** Self-attention trong encoder", "fid-cost",
     "Fusion-in-Decoder encode từng cặp (câu hỏi, tài liệu) riêng nên chỉ tính các khối trên đường chéo của ma trận attention."),
    ("và giả định số hạng cuối xấp xỉ 0", "pipeline-chain",
     "Xác suất trả lời đúng của pipeline naive là tích xác suất các khâu (giá trị minh họa của mục 6.1)."),
    ("| FP7 | Incomplete |", "failure-points",
     "Bảy điểm hỏng của Barnett et al. (2024) gắn với các khâu của pipeline naive, cùng hai điểm hỏng bổ sung cho case Zendesk."),
    ("| Module khóa học | 03–05 |", "rag-paradigms",
     "Naive, Advanced và Modular RAG theo Gao et al.: Advanced thêm bước trước và sau truy xuất; Modular có router, nhiều nguồn và vòng lặp."),
]
