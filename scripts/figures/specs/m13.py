MD = "docs/13-rag-da-phuong-thuc.md"
MODULE = "13"
PREFIX = "13"

FIGS = [
    ("Nhân tiếp với xác suất retrieval và generation đúng", "parse-loss",
     "Mô hình xác suất nối tiếp của mục 1.2 với các giá trị giả định: bước nhận dạng bảng yếu nhất kéo cả chuỗi xuống."),
    ("Có thể giảm số token bằng cách hạ độ phân giải", "visual-tokens",
     "Trái: mỗi visual token của Qwen3-VL ứng với ô khoảng 32×32 pixel. Phải: số token của ảnh so với văn bản OCR của cùng một trang (văn bản là ước lượng)."),
    ("Tổng $0.243$, chia $B = 2$", "siglip-example",
     "Ví dụ SigLIP của mục 2.2: mỗi cặp ảnh–văn bản là một bài phân loại nhị phân độc lập; cặp đúng có điểm 0,7 đóng góp loss lớn nhất."),
    ("Trừ vector trung bình của từng modality (ước lượng từ kho)", "modality-gap",
     "Trái: sơ đồ minh họa (dữ liệu sinh ngẫu nhiên) hai modality nằm ở hai vùng tách biệt. Phải: ví dụ số của mục 2.3, thứ hạng đúng sau khi khử thưởng modality."),
    ("Gộp trung bình xóa mất *vị trí* của thông tin", "maxsim-example",
     "Ví dụ tính tay của mục 4.2: mỗi token query chọn patch khớp nhất (viền cam). MaxSim phân biệt hai trang, còn gộp trung bình thì không."),
    ("Với kho vài nghìn trang, 1,2 GB là chấp nhận được.", "storage-cost",
     "Dung lượng chỉ mục cho 4.600 trang (giả định của case study) theo từng cách biểu diễn, thang log."),
    ("Bài học: \"thị giác hay văn bản\"", "vidore-findings",
     "Trái: ColPali so với pipeline văn bản tốt nhất trên ViDoRe V1. Phải: định vị vùng chứng cứ trên ViDoRe V3, model tốt nhất còn xa mức đồng thuận của người."),
    ("**Ví dụ 2 — dấu tiếng Việt.**", "cer-example",
     "Ví dụ 2 của mục 7.1: bốn lỗi dấu cho CER khoảng 23,5% nhưng WER 100%."),
    ("**Phân tầng là bắt buộc.**", "stratified-recall",
     "Ví dụ giả định của mục 7.2: Recall@5 tổng 0,84 che mất phân tầng bảng/hình chỉ đạt 0,60, với khoảng tin cậy rộng do ít mẫu."),
]
