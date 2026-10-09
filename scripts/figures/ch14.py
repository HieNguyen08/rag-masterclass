"""Hình minh họa cho Module 14 — RAG trên dữ liệu có cấu trúc."""
from math import comb

import numpy as np
from matplotlib.patches import Rectangle

from figkit import (AQUA, BLUE, GREEN, MAGENTA, ORANGE, RED, VIOLET, YELLOW, arrow, box,
                    canvas, figure, run, tint, xgrid, ygrid)


def ink(t, c):
    return t["c"][c] if t["name"] == "light" else t["fg"]


# ---------------------------------------------------------------- 1.1 top-k không đủ
@figure("topk-aggregation", size=(8.8, 2.9))
def _(fig, t):
    a1, a2 = fig.subplots(1, 2, gridspec_kw=dict(wspace=0.3, width_ratios=[1.35, 1]))
    rng = np.random.default_rng(7)
    amt = np.round(rng.uniform(0.5, 6.0, 37), 1)          # triệu đồng, minh họa
    seen = np.argsort(-rng.random(37))[:5]
    for i in range(37):
        r, c = divmod(i, 10)
        on = i in seen
        col = t["c"][BLUE] if on else t["line"]
        a1.add_patch(Rectangle((c, -r), 0.9, 0.9, fc=tint(t["c"][BLUE], t, .55) if on else t["panel"],
                                                               ec=col, lw=1.4 if on else 0.8))
    a1.set_xlim(-0.2, 10.1); a1.set_ylim(-3.3, 1.1); a1.set_aspect("equal"); a1.axis("off")
    a1.set_title("37 hóa đơn chưa trả; LLM chỉ thấy 5 dòng top-5", fontsize=9)
    vals = [amt.sum(), amt[seen].sum()]
    a2.bar(range(2), vals, color=[t["c"][GREEN], t["c"][RED]], width=0.55)
    for k, v in enumerate(vals):
        a2.text(k, v + 2, f"{v:.1f}".replace(".", ","), ha="center", fontsize=8.6, color=t["fg"], weight="bold")
    a2.set_xticks(range(2), ["tổng đúng\n(SQL trên cả tập)", "tổng LLM thấy\n(top-5)"], fontsize=8)
    a2.set_ylim(0, vals[0] * 1.2)
    a2.set_ylabel("triệu đồng (minh họa)")
    a2.set_title("Con số sai nhưng «trông hợp lý»", fontsize=9)
    ygrid(a2, t)


# ---------------------------------------------------------------- 2.2 phổ lựa chọn
@figure("freedom-spectrum", size=(8.8, 2.9))
def _(fig, t):
    ax = canvas(fig, (0, 124), (0, 40))
    ax.annotate("", xy=(120, 8), xytext=(4, 8), arrowprops=dict(arrowstyle="-|>", color=t["fg2"], lw=1.4))
    ax.text(4, 3, "ít tự do, bề mặt tấn công nhỏ", fontsize=7.8, color=t["fg2"])
    ax.text(120, 3, "nhiều tự do, rủi ro sai âm thầm lớn", fontsize=7.8, color=t["fg2"], ha="right")
    items = [("(a) Tool tham số hóa", "list_invoices(status: Literal[...])\ntenant gắn phía server", "Khách hàng (luồng email)", GREEN),
             ("(b) Lớp metric", "{metric, group_by, filters}\n→ SQL đã viết sẵn", "Báo cáo nội bộ lặp lại", YELLOW),
             ("(c) Text-to-SQL", "SQL tùy ý trên view cho phép\nchỉ đọc, có người kiểm tra", "Phân tích nội bộ ad-hoc", RED)]
    for k, (a, b, who, c) in enumerate(items):
        x = 4 + k * 40
        box(ax, x, 13, 36, 22, "", t, color=t["c"][c], fill=tint(t["c"][c], t, .14))
        ax.text(x + 2, 31, a, fontsize=9, weight="bold", color=t["fg"])
        ax.text(x + 2, 24, b, fontsize=7.6, color=t["fg"], va="center", linespacing=1.4)
        ax.text(x + 2, 16, "Dùng cho: " + who, fontsize=7.6, color=ink(t, c) if c != YELLOW else t["fg"])


# ---------------------------------------------------------------- 3.3 bỏ phiếu
@figure("vote-probability", size=(7.8, 2.9))
def _(fig, t):
    ax = fig.subplots()
    ns = np.arange(1, 16, 2)
    for p, c in [(0.4, RED), (0.6, ORANGE), (0.8, GREEN)]:
        ys = [sum(comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(n // 2 + 1, n + 1)) for n in ns]
        ax.plot(ns, ys, "o-", color=t["c"][c], lw=1.6, ms=4, label=f"p = {p:.1f} mỗi ứng viên".replace(".", ","))
    ax.plot(5, 0.68256, "o", ms=10, mfc="none", mec=t["fg"], mew=1.4)
    ax.text(5.4, 0.62, "n = 5, p = 0,6 → 0,683", fontsize=8, color=t["fg"])
    ax.set_xlabel("số ứng viên SQL n")
    ax.set_ylabel("P(đa số tuyệt đối đúng)")
    ax.set_ylim(0, 1.05); ax.set_xticks(ns)
    ax.legend(fontsize=7.8, loc="center right")
    ax.set_title("Cận dưới của bỏ phiếu theo kết quả (giả định ứng viên sai không trùng nhau)", fontsize=9)
    ygrid(ax, t)


# ---------------------------------------------------------------- 5.2 phòng thủ nhiều lớp
@figure("sql-defense-layers", size=(8.8, 3.3))
def _(fig, t):
    ax = canvas(fig, (0, 124), (0, 46))
    layers = [("1", "Không SQL tự do ở luồng khách — chỉ tool", GREEN),
              ("2", "Kết nối chỉ đọc, read replica", BLUE),
              ("3", "Cách ly tenant ở database: RLS / view theo tenant", BLUE),
              ("4", "Danh sách trắng view, không cột PII", BLUE),
              ("5", "Kiểm tra truy vấn: một câu lệnh, chỉ SELECT, ép LIMIT", VIOLET),
              ("6", "Giới hạn tài nguyên: timeout, số dòng", VIOLET),
              ("7", "Nhật ký + cảnh báo bất thường", AQUA)]
    for k, (n, s, c) in enumerate(layers):
        y = 38 - k * 5.4
        box(ax, 22 + k * 1.2, y, 70 - k * 2.4, 4.6, "", t, color=t["c"][c], fill=tint(t["c"][c], t, .16), radius=0.6)
        ax.text(24 + k * 1.2, y + 2.3, f"{n}. {s}", va="center", fontsize=7.8, color=t["fg"])
    box(ax, 2, 18, 16, 10, "Câu hỏi /\nchỉ dẫn độc\n(P2SQL)", t, color=t["c"][RED], fill=tint(t["c"][RED], t, .15), fs=7.8)
    arrow(ax, 18.3, 23, 21.7, 23, t, color=t["c"][RED], lw=1.6)
    box(ax, 100, 18, 22, 10, "Dữ liệu\nđúng tenant,\nkhông PII", t, color=t["c"][GREEN], fill=tint(t["c"][GREEN], t, .15), fs=7.8)
    arrow(ax, 92.5, 23, 99.7, 23, t, color=t["c"][GREEN], lw=1.6)
    ax.text(2, 3, "Mỗi lớp vẫn đứng vững nếu LLM bị thao túng hoàn toàn; lớp 1 loại bỏ phần lớn bề mặt tấn công ở luồng email.",
            fontsize=7.8, color=t["fg2"])


if __name__ == "__main__":
    run("14")
