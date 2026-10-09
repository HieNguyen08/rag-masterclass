"""Hình minh họa cho Module 00 — Tổng quan khóa học và bài toán xuyên suốt."""
from figkit import (AQUA, BLUE, GREEN, ORANGE, RED, VIOLET, YELLOW, arrow, box, canvas, figure, run, tint)


# ---------------------------------------------------------------- 1 ba bài toán con
@figure("three-subproblems", size=(8.8, 3.2))
def _(fig, t):
    ax = canvas(fig, (0, 124), (0, 44))
    box(ax, 2, 16, 16, 12, "Email\nkhách hàng", t, fs=9, weight="bold")
    parts = [("Phân loại", "intent · ngôn ngữ · rủi ro\nmuốn gặp người?", BLUE),
             ("RAG", "tìm tri thức + soạn\ntrả lời có căn cứ", AQUA),
             ("Quyết định", "gửi / để draft /\nchuyển người", VIOLET)]
    for i, (a, b, c) in enumerate(parts):
        x = 24 + i * 26
        box(ax, x, 14, 22, 16, "", t, color=t["c"][c], fill=tint(t["c"][c], t, .15), lw=2 if i == 2 else 1.4)
        ax.text(x + 11, 25.5, a, ha="center", fontsize=9.6, weight="bold", color=t["fg"])
        ax.text(x + 11, 19.5, b, ha="center", va="center", fontsize=7.8, color=t["fg2"])
        arrow(ax, x - 5.7 if i else 18.3, 22, x - 0.3, 22, t, lw=1.6)
    outs = [("SEND", "public reply", GREEN), ("DRAFT", "internal note", YELLOW), ("ESCALATE", "báo team CS", RED)]
    for i, (a, b, c) in enumerate(outs):
        y = 31 - i * 10
        box(ax, 104, y, 18, 7.5, f"{a}\n{b}", t, color=t["c"][c], fill=tint(t["c"][c], t, .15), fs=7.8)
        arrow(ax, 98.3, 22, 103.7, y + 3.75, t, lw=1)
    ax.text(2, 40, "Hệ thống = Phân loại + RAG + Quyết định", fontsize=9.6, weight="bold", color=t["fg"])
    ax.text(87, 8.5, "phần hay bị xem nhẹ nhất —\ntrọng tâm của Module 10", ha="center", va="top", fontsize=7.8,
            color=t["c"][VIOLET] if t["name"] == "light" else t["fg"])


# ---------------------------------------------------------------- 2 quy mô
@figure("scale-at-a-glance", size=(8.8, 2.7))
def _(fig, t):
    ax = canvas(fig, (0, 124), (0, 36))
    steps = [("1.500", "ticket mới / ngày", BLUE), ("× 3,5 lượt", "", None), ("5.250", "lần sinh trả lời / ngày", VIOLET)]
    box(ax, 2, 13, 26, 14, "1.500\nticket mới / ngày", t, color=t["c"][BLUE], fill=tint(t["c"][BLUE], t, .15), fs=9.4, weight="bold")
    arrow(ax, 28.3, 20, 38.7, 20, t, lw=1.6)
    ax.text(33.5, 22, "× 3,5 lượt", ha="center", fontsize=8.2, color=t["fg2"])
    box(ax, 39, 13, 26, 14, "5.250\nlần sinh / ngày", t, color=t["c"][VIOLET], fill=tint(t["c"][VIOLET], t, .15), fs=9.4, weight="bold")
    for i, (a, b, mult, c) in enumerate([("31,5 triệu", "token vào / ngày", "× ~6.000 token context", ORANGE),
                                         ("2,1 triệu", "token ra / ngày", "× ~400 token đầu ra", AQUA)]):
        y = 21 - i * 14
        arrow(ax, 65.3, 20, 87.7, y + 5, t, lw=1.3)
        ax.text(71, y + 7.5 if i == 0 else y + 1, mult, fontsize=7.8, color=t["fg2"])
        box(ax, 88, y, 34, 10, f"{a}\n{b}", t, color=t["c"][c], fill=tint(t["c"][c], t, .15), fs=9, weight="bold")
    ax.text(2, 33, "Quy mô giả định dùng xuyên suốt khóa học", fontsize=9.4, weight="bold", color=t["fg"])
    ax.text(2, 6, "giờ cao điểm gấp ~3 lần;\nchi phí chi tiết ở Module 11", fontsize=7.8, color=t["fg2"])


# ---------------------------------------------------------------- 2 lộ trình rollout
@figure("rollout-stages", size=(8.8, 2.4))
def _(fig, t):
    ax = canvas(fig, (0, 124), (0, 30))
    stages = [("Giai đoạn 1", "AI chỉ viết draft (internal note)\nagent duyệt mọi câu trả lời", BLUE),
              ("Giai đoạn 2", "tự gửi với nhóm intent rủi ro thấp\nphần còn lại vẫn là draft", AQUA),
              ("Mở rộng dần", "thêm intent khi số liệu đánh giá\nchứng minh rủi ro trong giới hạn", GREEN)]
    for i, (a, b, c) in enumerate(stages):
        x = 2 + i * 41
        box(ax, x, 6, 36, 16, "", t, color=t["c"][c], fill=tint(t["c"][c], t, .14))
        ax.text(x + 2, 18.5, a, fontsize=9.4, weight="bold", color=t["fg"])
        ax.text(x + 2, 11, b, fontsize=7.8, color=t["fg"], va="center")
        if i < 2:
            arrow(ax, x + 36.3, 14, x + 40.7, 14, t, lw=1.8)
    ax.text(2, 26.5, "Rollout an toàn: mỗi bước mở thêm quyền tự động chỉ khi có số liệu (Module 10)", fontsize=9, weight="bold", color=t["fg"])
    ax.text(2, 1, "Giai đoạn 1 đồng thời là «máy sinh nhãn»: mỗi lần agent sửa draft là một nhãn cho đánh giá và fine-tune.",
            fontsize=7.8, color=t["fg2"])


if __name__ == "__main__":
    run("00")
