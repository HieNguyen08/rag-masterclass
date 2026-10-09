MD = "docs/03-embedding-va-bieu-dien.md"
MODULE = "03"
PREFIX = "3"

FIGS = [
    ("Đây là *vocabulary mismatch*", "onehot-vs-dense",
     "Trái: với vector đếm từ, d₁ thắng chỉ nhờ hai từ chức năng, còn bài đúng d₂ có cosine 0. "
     "Phải: sơ đồ minh họa điều ta muốn — văn bản cùng nghĩa nằm gần nhau dù khác chữ, khác ngôn ngữ."),
    ("Số hạng đầu kéo cặp thật lại gần", "word2vec-skipgram",
     "Skip-gram với negative sampling: từ trung tâm được kéo gần các từ trong cửa sổ ngữ cảnh và đẩy xa K từ nhiễu."),
    ("Ba kiến trúc là ba điểm khác nhau trên đường cong trade-off", "scoring-architectures",
     "Ba cách chấm điểm f(q, d). Bi-encoder và ColBERT tính trước được phía tài liệu; cross-encoder phải chạy lại cho từng cặp."),
    ("khớp mạnh *từng* khái niệm", "colbert-maxsim",
     "MaxSim trên ví dụ của mục 2.4: d⁽¹⁾ có token khớp mạnh cho cả hai khái niệm nên điểm cao hơn d⁽²⁾."),
    ("Đây là trực giác cốt lõi: **$\\tau$ điều khiển", "infonce-temperature",
     "Trái: phân phối softmax của bảng ở mục 3.3. Phải: khi τ giảm, p(d⁺) tăng và gần như toàn bộ lực đẩy dồn vào hard negative d₂."),
    ("Gradient descent kéo $\\mathbf{z}_q$ về phía", "infonce-gradient",
     "Hình học của gradient (τ = 0.05, góc dựng từ các cosine 0.82 / 0.75 / 0.40 / 0.30): z_q bị kéo về d⁺ và đẩy khỏi trọng tâm có trọng số, mà trọng tâm đó gần như chỉ gồm d⁺ và d₂."),
    ("Embedding chỉ giữ những gì *giúp phân biệt*", "infonce-mi-bound",
     "Cận dưới thông tin tương hỗ từ InfoNCE không thể vượt log B — một lý do định lượng cho batch lớn."),
    ("Hai đại lượng này tiện để chẩn đoán model", "alignment-uniformity",
     "Alignment và uniformity trên đường tròn đơn vị (dữ liệu mô phỏng). Chỉ tối ưu alignment dẫn tới sụp đổ; InfoNCE cần cả hai."),
    ("**Vì sao last-token cho decoder-only?**", "pooling",
     "Ba kiểu pooling trên cùng ma trận H. Mean pooling phải nhân attention mask để loại padding."),
    ("**Quy tắc vàng:** dùng đúng độ đo", "similarity-metrics",
     "Trái: trên mặt cầu đơn vị, L2² là hàm giảm tuyến tính của cosine nên ba độ đo cho cùng thứ hạng (điểm cam là ví dụ mục 5.2). "
     "Phải: khi chưa chuẩn hóa, dot product thưởng cho vector dài."),
    ("Chạy thật cho trung bình $0.978$", "anisotropy",
     "Anisotropy: vector dồn vào một hình nón hẹp nên cosine giữa hai văn bản bất kỳ ≈ 0.978; trừ trung bình (centering) đưa phân phối về quanh 0. Số liệu bên phải là mô phỏng đúng như mục 6.1."),
    ("→ $d_B$ thắng, đúng như mong muốn", "hubness",
     "Trái: mô phỏng hubness — ở 100 chiều, phân phối N₁₀ lệch phải mạnh, vài điểm lọt top-10 của hơn 150 điểm khác; centering làm giảm rõ rệt. "
     "Phải: ví dụ CSLS của mục 6.2 đảo lại thứ hạng giữa hub và tài liệu đúng."),
    ("(Cách tính: $300{,}000", "index-memory",
     "Bộ nhớ vector thô cho ~300K chunk theo bảng mục 7.1 (trục log)."),
    ("**Ví dụ số: vì sao không thể tùy tiện cắt chiều", "matryoshka",
     "Trái: loss Matryoshka cộng InfoNCE trên từng tiền tố lồng nhau. Phải: ví dụ 4 chiều — với model không có MRL, cắt chiều biến hai vector khác nghĩa thành gần như trùng nhau."),
    ("Trong dải, sai số tối đa chỉ", "int8-quantization",
     "Hàm lượng tử int8 với dải hiệu chuẩn [−0.30, 0.30] và bốn tọa độ của ví dụ; 0.34 nằm ngoài dải nên bị cắt về 127."),
    ("Sai lệch lớn: với $d=8$", "binary-hamming",
     "Trái: vì sao xác suất một siêu phẳng ngẫu nhiên tách x và y bằng θ/π. Phải: mô phỏng ước lượng cos(π·H/d) — 8 bit rất nhiễu, 1024 bit bám sát đường chéo."),
    ("Blog Hugging Face (3/2024) báo cáo binary + rescoring", {"id": "binary-rescore", "mermaid": """
flowchart LR
    Q[Email / query] --> E[Embed float32 q]
    E --> B[Lấy dấu → b_q nhị phân]
    B --> P1["Pha 1: Hamming trên index binary<br/>top-k·ρ (ρ ≈ 4)"]
    P1 --> P2["Pha 2: tính lại qᵀd̃<br/>q float32, d̃ int8/float32 từ đĩa"]
    E --> P2
    P2 --> K[Top-k cuối]
"""}, "Quy trình binary + rescoring hai pha: pha thô rẻ trên bit, pha tinh chính xác trên ít ứng viên."),
    ("Tóm lại: MRL 1024→256 và int8", "compression-tradeoff",
     "Các mức nén cho ~300K chunk 1024 chiều và điều kiện để dùng an toàn."),
    ("Phần ngữ nghĩa của $d_{\\text{en}}$ cao hơn", "language-bias",
     "Ví dụ giả định của mục 8.2: «thưởng» cùng ngôn ngữ đủ để bài sai chủ đề vượt bài đúng."),
    ("Instruction phía query cho phép **một index phục vụ nhiều tác vụ**", "instruction-prefix",
     "Cùng một câu, ba instruction khác nhau cho ba vector phục vụ ba tác vụ; phía tài liệu giữ nguyên."),
    ("**Điểm hay:** vector của", "splade-expansion",
     "Minh họa vector SPLADE (trọng số giả định): ngoài từ có trong văn bản, model gán trọng số cho từ liên quan như «login», «sign»."),
    ("với $\\text{rank}_i$ là vị trí của tài liệu liên quan đầu tiên", "recall-mrr",
     "Tính Recall@k và MRR trên một danh sách xếp hạng."),
]
