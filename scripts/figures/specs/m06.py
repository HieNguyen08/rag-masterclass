MD = "docs/06-query-va-reranking.md"
MODULE = "06"
PREFIX = "6"

FIGS = [
    ("Toán của vấn đề \"trung bình hai chủ đề\"", "topic-averaging",
     "Trái: email gộp hai chủ đề trực giao nằm giữa hai tài liệu đúng, nên một bài «tổng quan» chung chung lại gần nó nhất. Phải: cosine tới tài liệu đúng giảm theo 1/√m."),
    ("Đầu ra này phục vụ **ba** mục đích cùng lúc", "email-analysis",
     "Một lời gọi LLM biến email ở mục 2.1 thành truy vấn con, định danh cho BM25 và các tín hiệu routing/escalate."),
    ("**Condensation:** cho LLM lịch sử hội thoại", "condensation",
     "Viết lại tin nhắn không tự đứng được thành truy vấn độc lập, đủ thực thể."),
    ("với $r'$ là recall trên phần \"không khó\"", "multiquery-recall",
     "Recall của multi-query theo số diễn đạt lại: mô hình độc lập so với mô hình có phần «khó» ρ = 0.25 (bão hòa ở 0.75)."),
    ("Cosine: $\\mathrm{sim}(\\mathbf{q}, \\mathbf{d}_1) = 0{,}51 \\times 0{,}6", "hyde",
     "Trái: phép chiếu minh họa của ví dụ 4 chiều — truy vấn khớp «phong cách câu hỏi», tài liệu giả định khớp «phong cách tài liệu». Phải: các cosine tính trong mục 5.2."),
    ("Trực giác bằng toán: câu cụ thể có embedding", "decomposition-stepback",
     "Decomposition tách một câu ghép thành các câu con độc lập; step-back thêm một câu tổng quát để khớp bài chính sách nền."),
    ("**Routing mềm vs cứng.**", "routing-hard-soft",
     "Recall kỳ vọng của routing cứng a·r_in so với tìm trên mọi nguồn; với r_in = 0.9, router phải đúng trên ~94% mới hòa vốn."),
    ("**Ví dụ số (ước lượng).** `bge-reranker-v2-m3`", "rerank-cascade",
     "Trái: kiến trúc nhiều tầng của mục 1. Phải: thời gian rerank ước lượng theo công thức FLOPs của mục 8.2 (bỏ qua hạng tử L²)."),
    ("| Listwise sliding window |", "llm-rerank-calls",
     "Số lời gọi LLM theo số ứng viên cho từng cách dùng LLM làm reranker (trục log)."),
    ("**$\\lambda = 0{,}5$:** d1 → d5", "mmr-example",
     "Ví dụ MMR mục 9.3: ma trận tương tự giữa năm ứng viên và ba tài liệu được chọn đầu tiên với từng λ."),
    ("Token dễ đoán (xác suất cao, $I$ thấp)", "selective-context",
     "Selective Context giữ token có self-information cao; số liệu giả định chỉ để minh họa cơ chế và rủi ro «xé» văn bản."),
    ("Với context ngắn (≤ 8 đoạn, < 4k token)", "context-ordering",
     "Hai cách sắp xếp năm đoạn đã xếp hạng (màu đậm = điểm cao)."),
    ("| **Tổng tiền xử lý trước generation** |", "latency-budget",
     "Bảng ngân sách latency mục 11.2 trên trục log; hai bước xám chỉ bật có điều kiện."),
]
