# Hình minh họa (Phần II–III)

Mọi hình trong `docs/assets/figures/<module>/` được sinh bằng code trong thư mục này — sửa code rồi chạy lại, không sửa tay file SVG.

| File | Nội dung |
|---|---|
| `figkit.py` | Bảng màu, style chung, xuất mỗi hình thành hai bản `.light.svg` / `.dark.svg` |
| `ch03.py` … `ch10.py` | Hình của từng module (mỗi hàm `@figure("tên")` là một hình) |
| `insert.py` + `specs/m0X.py` | Chèn khối `<figure>` vào Markdown sau dòng neo, đánh số «Hình X.Y» theo thứ tự xuất hiện |

```bash
pip install matplotlib numpy scipy
cd scripts/figures
python ch05.py                 # sinh lại toàn bộ hình Module 05
python ch05.py ivf hnsw        # chỉ sinh lại vài hình
python insert.py specs/m05.py  # chèn/cập nhật hình trong docs (chạy lại an toàn)
FIG_PREVIEW=/tmp/prev python ch05.py   # thêm bản PNG để xem nhanh
```

Trang MkDocs Material chọn đúng bản theo theme qua hậu tố `#only-light` / `#only-dark`. Số liệu trong hình lấy đúng từ ví dụ trong bài; hình nào là mô phỏng hoặc số giả định thì chú thích ghi rõ.
