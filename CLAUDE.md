# Quy tắc cho AI/agent làm việc trong repo này

## Tác giả (bắt buộc, áp dụng mọi lúc)

- Tác giả duy nhất của repo là **Nguyen Minh Hieu** (`HieNguyen08`).
- Mọi commit phải có **author và committer** là:
  `Nguyen Minh Hieu <nguyenminhhieu080203@gmail.com>`
  Trước khi commit, chạy:
  ```bash
  git config user.name "Nguyen Minh Hieu"
  git config user.email "nguyenminhhieu080203@gmail.com"
  ```
- **Không** thêm bất kỳ dòng ghi công nào cho AI trong commit message, mô tả pull request, nội dung trang hoặc file: không `Co-Authored-By`, không `Claude-Session`, không "Generated with …".
- Commit do GitHub Actions tạo ra (deploy `gh-pages`) cũng phải dùng danh tính trên (đã cấu hình trong `.github/workflows/deploy.yml`).
- Không đổi `site_author`, `copyright` trong `mkdocs.yml` sang tên khác.

## Nội dung

- Viết nội dung theo `notes/STYLE_GUIDE.md` và ranh giới module trong `notes/SYLLABUS.md`.
- Toán dùng `$...$` / `$$...$$` (dòng trống trước và sau khối `$$`); khối đáp án dùng `<details markdown="1">`.
- Kiểm tra `mkdocs build --strict` trước khi push.
