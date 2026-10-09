"""Hình minh họa cho Module 06 — Xử lý query, reranking, nén context."""
import numpy as np
from matplotlib.patches import Patch, Rectangle

from figkit import (AQUA, BLUE, GREEN, MAGENTA, ORANGE, RED, VIOLET, YELLOW, arrow, box,
                    canvas, figure, note, run, tint, xgrid, ygrid)


def _vec(ax, v, c, lab, t, off=(0.04, 0.04), lw=2.2, ha="left"):
    ax.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=c, lw=lw))
    ax.text(v[0] + off[0], v[1] + off[1], lab, fontsize=8.8, color=t["fg"], ha=ha)


# ---------------------------------------------------------------- 2.1 trung bình chủ đề
@figure("topic-averaging", size=(8.8, 3.3))
def _(fig, t):
    a1, a2 = fig.subplots(1, 2, gridspec_kw=dict(wspace=0.3, width_ratios=[1, 1.1]))
    a1.set_aspect("equal"); a1.axis("off")
    th = np.linspace(0, np.pi / 2, 100)
    a1.plot(np.cos(th), np.sin(th), color=t["line"], lw=1)
    _vec(a1, (1, 0), t["c"][BLUE], "d_a: bài hóa đơn", t, off=(-0.05, -0.12))
    _vec(a1, (0, 1), t["c"][AQUA], "d_b: bài lỗi đồng bộ", t, off=(0.04, 0.0))
    q = np.array([1, 1]) / np.sqrt(2)
    _vec(a1, q * 0.98, t["c"][ORANGE], "q = email gộp hai chủ đề\n≈ hướng bài «tổng quan»\n(cos với q ≈ 1, thắng cả hai)", t, off=(0.04, -0.02))
    a1.text(0.38, 0.12, "cos = 0.71", fontsize=8.6, color=t["fg2"])
    a1.set_xlim(-0.15, 1.75); a1.set_ylim(-0.25, 1.2)
    a1.set_title("Email hai ý: không gần tài liệu nào thật sự", fontsize=9.6)
    m = np.arange(1, 6)
    b = a2.bar(m, 1 / np.sqrt(m), color=t["c"][ORANGE], width=0.55)
    for x, v in zip(m, 1 / np.sqrt(m)):
        a2.text(x, v + 0.02, f"{v:.2f}", ha="center", fontsize=8.6, color=t["fg"])
    a2.set_xlabel("số chủ đề độc lập gộp trong một truy vấn"); a2.set_ylabel("cos tới tài liệu đúng")
    a2.set_ylim(0, 1.12); a2.set_title("cos = 1/√m (chủ đề trực giao)", fontsize=9.6)
    ygrid(a2, t)


# ---------------------------------------------------------------- 2.2 email → truy vấn
@figure("email-analysis", size=(8.8, 3.8))
def _(fig, t):
    ax = canvas(fig, (0, 124), (0, 52))
    box(ax, 2, 6, 34, 42, "", t)
    lines = ["Chào team,", "Gói Business, tuần trước nâng", "lên 50 user. Hóa đơn tháng này",
             "vẫn tính 30 user?? Ngoài ra đồng", "bộ MISA báo lỗi ERR_SYNC_409,", "đã thử reconnect. Nếu không xử",
             "lý được hôm nay thì cho mình nói", "chuyện với ai đó nhé…", "Thanks, Lan — Kế toán trưởng"]
    for i, s in enumerate(lines):
        ax.text(4, 44 - i * 4.3, s, fontsize=8, color=t["fg2"] if i in (0, 8) else t["fg"])
    box(ax, 41, 22, 14, 10, "LLM nhỏ\nJSON schema\nT = 0", t, color=t["c"][VIOLET], fill=tint(t["c"][VIOLET], t, .14), fs=8.4)
    arrow(ax, 36.5, 27, 40.5, 27, t, lw=1.8)
    outs = [("câu hỏi con 1 (ngầm): proration khi nâng user giữa kỳ", "→ retrieval", BLUE),
            ("câu hỏi con 2: hóa đơn vẫn tính 30 user", "→ retrieval", BLUE),
            ("câu hỏi con 3: ERR_SYNC_409 với MISA", "→ retrieval", BLUE),
            ("identifiers: ERR_SYNC_409, MISA, Business", "→ BM25 nguyên văn", ORANGE),
            ("sensitive_topics: pricing", "→ routing, guardrail", RED),
            ("wants_human = true · urgency = high", "→ escalate (Module 10)", GREEN)]
    for i, (a, b, c) in enumerate(outs):
        y = 44 - i * 7.3
        box(ax, 60, y, 44, 5.6, a, t, color=t["c"][c], fill=tint(t["c"][c], t, .12), fs=7.9, ha="left", radius=0.7)
        ax.text(105, y + 2.8, b, fontsize=7.9, color=t["fg2"], va="center")
        arrow(ax, 55.3, 27, 59.6, y + 2.8, t, lw=0.9)


# ---------------------------------------------------------------- 3 condensation
@figure("condensation", size=(8.8, 2.9))
def _(fig, t):
    ax = canvas(fig, (0, 124), (0, 38))
    msgs = [("Khách", "Đồng bộ MISA báo ERR_SYNC_409."), ("AI", "Thử bước 1 … bước 2: tạo lại API token …"),
            ("Khách", "Vẫn không được, làm bước 2 rồi mà nó báo lỗi khác.")]
    for i, (who, s) in enumerate(msgs):
        y = 28 - i * 9
        c = t["c"][BLUE] if who == "Khách" else t["c"][VIOLET]
        box(ax, 2, y, 52, 7, f"{who}: {s}", t, color=c, fill=tint(c, t, .12), fs=7.9, ha="left", radius=0.8,
            lw=2.2 if i == 2 else 1.2)
    ax.text(2, 36, "Lịch sử ticket + tin nhắn mới (không tự đứng được)", fontsize=8.8, color=t["fg2"])
    box(ax, 60, 16, 14, 9, "LLM\nviết lại", t, color=t["c"][VIOLET], fill=tint(t["c"][VIOLET], t, .14), fs=8.6)
    arrow(ax, 54.5, 20.5, 59.5, 20.5, t, lw=1.8)
    arrow(ax, 74.5, 20.5, 79.5, 20.5, t, lw=1.8)
    box(ax, 80, 13, 42, 15, "Lỗi mới sau khi làm bước 2 (tạo lại\nAPI token) trong hướng dẫn khắc phục\nERR_SYNC_409 khi đồng bộ MISA", t,
        color=t["c"][GREEN], fill=tint(t["c"][GREEN], t, .14), fs=8.2, ha="left")
    ax.text(80, 30, "Truy vấn độc lập cho retriever", fontsize=8.8, color=t["fg2"])
    ax.text(2, 1.5, "Bẫy: khách đổi sang chủ đề khác mà rewriter vẫn chèn «ERR_SYNC_409» → kéo retrieval về bài cũ.",
            fontsize=8.3, color=t["fg2"])


# ---------------------------------------------------------------- 4.2 multi-query
@figure("multiquery-recall", size=(6.8, 3.2))
def _(fig, t):
    ax = fig.subplots()
    n = np.arange(1, 9)
    ax.plot(n, 1 - 0.4 ** n, color=t["c"][BLUE], marker="o", ms=5, label="độc lập, rᵢ = 0.6")
    rho, rp = 0.25, 0.8
    y = (1 - rho) * (1 - (1 - rp) ** n)
    ax.plot(n, y, color=t["c"][ORANGE], marker="s", ms=5, label="có tương quan: ρ = 0.25, r′ = 0.8")
    ax.axhline(1 - rho, color=t["c"][ORANGE], ls=":", lw=1.2)
    ax.text(8.1, 1 - rho - 0.04, "trần 1 − ρ = 0.75", fontsize=8.3, color=t["fg2"], ha="right")
    for k in (1, 3):
        ax.text(k + 0.1, y[k - 1] - 0.06, f"{y[k - 1]:.3f}", fontsize=8.3, color=t["fg"])
    ax.text(3.1, 1 - 0.4 ** 3 + 0.02, f"{1 - 0.4 ** 3:.3f}", fontsize=8.3, color=t["fg"])
    ax.set_xlabel("số truy vấn diễn đạt lại n"); ax.set_ylabel("P(tài liệu đúng lọt ít nhất một danh sách)")
    ax.set_ylim(0.5, 1.03)
    ax.set_title("Multi-query: phần lớn lợi ích đến từ 2–3 truy vấn đầu")
    ax.legend(fontsize=8.3, loc="lower right")
    ygrid(ax, t)


# ---------------------------------------------------------------- 5.2 HyDE
@figure("hyde", size=(8.8, 3.3))
def _(fig, t):
    a1, a2 = fig.subplots(1, 2, gridspec_kw=dict(wspace=0.3, width_ratios=[1, 1.1]))
    a1.set_xlim(-0.95, 1.25); a1.set_ylim(-1.15, 1.15); a1.axis("off")
    a1.axhline(0, color=t["line"], lw=0.8); a1.axvline(0, color=t["line"], lw=0.8)
    a1.text(1.22, 0.04, "chủ đề\n«hoàn tiền»", fontsize=7.8, color=t["fg2"], ha="right")
    a1.text(-0.9, 1.08, "phong cách câu hỏi ↑", fontsize=7.8, color=t["fg2"])
    a1.text(-0.9, -1.12, "phong cách tài liệu ↓", fontsize=7.8, color=t["fg2"])
    _vec(a1, (0.51, 0.86), t["c"][ORANGE], "q (câu hỏi khách)", t)
    _vec(a1, (0.6, -0.8), t["c"][GREEN], "d₁ chính sách\nhoàn tiền (đúng)", t, off=(0.03, -0.15))
    _vec(a1, (0.0, 0.87), t["c"][RED], "d₂ FAQ hóa đơn\n(sai, cùng giọng hỏi)", t, off=(-0.06, -0.2), ha="right")
    _vec(a1, (0.55, -0.83), t["c"][VIOLET], "g (tài liệu\ngiả định)", t, off=(-0.2, 0.05), lw=1.6, ha="right")
    a1.set_title("Phép chiếu minh họa: phong cách vs chủ đề", fontsize=9.6)
    pairs = ["q·d₁", "q·d₂", "g·d₁", "g·d₂", "mix·d₁", "mix·d₂"]
    vals = [0.30, 0.75, 0.997, 0.03, 0.81, 0.48]
    cols = [t["c"][GREEN] if "d₁" in p else t["c"][RED] for p in pairs]
    x = np.array([0, 0.8, 2, 2.8, 4, 4.8])
    b = a2.bar(x, vals, color=cols, width=0.7)
    for xi, v in zip(x, vals):
        a2.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=8.3, color=t["fg"])
    a2.set_xticks(x, pairs, fontsize=8.2)
    for xc, lab in [(0.4, "truy vấn gốc\n→ chọn SAI"), (2.4, "HyDE\n→ chọn đúng"), (4.4, "normalize(q + g)\n→ đúng, giữ neo")]:
        a2.text(xc, -0.32, lab, ha="center", fontsize=8, color=t["fg2"])
    a2.set_ylim(0, 1.1); a2.set_ylabel("cosine")
    a2.set_title("Ví dụ 4 chiều của mục 5.2", fontsize=9.6)
    a2.legend(handles=[Patch(color=t["c"][GREEN], label="với tài liệu đúng d₁"), Patch(color=t["c"][RED], label="với tài liệu sai d₂")],
              fontsize=8, loc="upper right")
    ygrid(a2, t)


# ---------------------------------------------------------------- 6 decomposition / step-back
@figure("decomposition-stepback", size=(8.8, 3.3))
def _(fig, t):
    ax = canvas(fig, (0, 124), (0, 46))
    ax.text(2, 43, "Decomposition song song", fontsize=9.4, weight="bold", color=t["fg"])
    box(ax, 2, 26, 30, 12, "Business có SSO Azure AD\nkhông, và nâng Enterprise\ngiữa kỳ thì SSO có ngay?", t, fs=7.9)
    for i, s in enumerate(["Gói nào hỗ trợ\nSSO / Azure AD?", "Nâng gói giữa kỳ: tính năng\nmới có hiệu lực khi nào?"]):
        y = 33 - i * 10
        box(ax, 38, y, 24, 8, s, t, color=t["c"][BLUE], fill=tint(t["c"][BLUE], t, .12), fs=7.8)
        arrow(ax, 32.5, 32, 37.5, y + 4, t)
    ax.text(2, 18, "→ retrieval song song rồi hợp nhất", fontsize=8.2, color=t["fg2"])
    ax.plot([66, 66], [3, 45], color=t["grid"], lw=1)
    ax.text(70, 43, "Step-back", fontsize=9.4, weight="bold", color=t["fg"])
    box(ax, 70, 28, 52, 9, "Cụ thể: «Tại sao tài khoản của tôi bị khóa\nsau khi đổi email?» → khớp ticket tương tự", t,
        color=t["c"][ORANGE], fill=tint(t["c"][ORANGE], t, .12), fs=8)
    box(ax, 70, 12, 52, 9, "Tổng quát: «Chính sách bảo mật khi thay đổi\nemail đăng nhập là gì?» → khớp bài chính sách", t,
        color=t["c"][GREEN], fill=tint(t["c"][GREEN], t, .12), fs=8)
    arrow(ax, 96, 27.5, 96, 21.5, t, lw=1.6)
    ax.text(98, 24, "lùi một bước", fontsize=8, color=t["fg2"])
    ax.text(70, 5, "q = μ_khái niệm + δ_chi tiết; câu step-back ≈ μ_khái niệm", fontsize=8.3, color=t["fg2"])


# ---------------------------------------------------------------- 7.2 routing cứng vs mềm
@figure("routing-hard-soft", size=(6.6, 3.1))
def _(fig, t):
    ax = fig.subplots()
    a = np.linspace(0.6, 1.0, 200)
    for rin, c in [(0.9, BLUE), (0.95, AQUA)]:
        ax.plot(a, a * rin, color=t["c"][c], label=f"routing cứng, r_in = {rin}")
    ax.axhline(0.85, color=t["c"][ORANGE], ls="--", lw=1.6, label="tìm trên mọi nguồn, r_all = 0.85")
    ax.scatter([0.9], [0.81], color=t["c"][BLUE], s=36, zorder=4)
    ax.text(0.905, 0.8, "a = 0.9 → 0.81 < 0.85", fontsize=8.4, color=t["fg"], ha="left", va="top")
    ax.axvline(0.85 / 0.9, color=t["muted"], ls=":", lw=1)
    ax.text(0.85 / 0.9 + 0.005, 0.6, f"hòa vốn a ≈ {0.85 / 0.9:.3f}", fontsize=8.2, color=t["fg2"])
    ax.set_xlabel("độ chính xác của router a"); ax.set_ylabel("recall kỳ vọng")
    ax.set_title("Routing cứng chỉ thắng khi router rất chính xác")
    ax.legend(fontsize=8, loc="upper left")
    ygrid(ax, t)


# ---------------------------------------------------------------- 8 reranking cascade + chi phí
@figure("rerank-cascade", size=(8.8, 3.3))
def _(fig, t):
    a1 = fig.add_axes([0.0, 0.0, 0.5, 1.0]); a2 = fig.add_axes([0.6, 0.15, 0.39, 0.7])
    a1.set_xlim(0, 60); a1.set_ylim(0, 44); a1.axis("off")
    stages = [("Kho ~500K chunk", "", 56, t["muted"]),
              ("Hybrid retrieval", "top-100 · ms", 44, t["c"][BLUE]),
              ("Cross-encoder rerank", "top-50 → 20 · ~0.2–0.8 s", 32, t["c"][ORANGE]),
              ("MMR + nén", "top 5–8 · ms", 20, t["c"][AQUA])]
    for i, (name, sub, w, c) in enumerate(stages):
        y = 34 - i * 10
        box(a1, 30 - w / 2, y, w, 7.5, "", t, color=c, fill=tint(c, t, .14))
        a1.text(30, y + 4.9, name, ha="center", fontsize=8.8, weight="bold", color=t["fg"])
        if sub:
            a1.text(30, y + 1.8, sub, ha="center", fontsize=7.9, color=t["fg2"])
    a1.text(30, 1.5, "mỗi tầng đắt hơn xử lý ít ứng viên hơn", ha="center", fontsize=8.3, color=t["fg2"])
    k = np.arange(10, 101)
    for L, c in [(200, AQUA), (400, ORANGE), (512, RED)]:
        sec = k * 2 * 0.3e9 * L / 30e12
        a2.plot(k, sec, color=t["c"][c], label=f"L = {L} token")
    a2.scatter([50], [50 * 2 * 0.3e9 * 400 / 30e12], color=t["c"][ORANGE], s=30, zorder=4)
    a2.text(52, 0.33, "k = 50, L = 400\n≈ 0.4 s", fontsize=8, color=t["fg"])
    a2.set_xlabel("số ứng viên rerank k"); a2.set_ylabel("giây (ước lượng)")
    a2.set_title("~0.3B tham số không-embedding,\nGPU hiệu dụng 30 TFLOPS", fontsize=9)
    a2.legend(fontsize=7.8, loc="upper left")
    ygrid(a2, t)


# ---------------------------------------------------------------- 8.5 LLM reranker: số lời gọi
@figure("llm-rerank-calls", size=(8.4, 3.2))
def _(fig, t):
    ax = fig.subplots()
    k = np.arange(10, 101)
    ax.plot(k, k, color=t["c"][BLUE], label="pointwise: k (song song)")
    ax.plot(k, k * (k - 1), color=t["c"][RED], label="pairwise all-pairs: k(k−1)")
    ax.plot(k, k * np.log2(k), color=t["c"][ORANGE], label="pairwise sorting: ~k·log₂k")
    w, s = 20, 10
    ax.plot(k, np.ceil(np.maximum(k - w, 0) / s) + 1, color=t["c"][GREEN], label="listwise trượt (w=20, s=10)")
    ax.scatter([50], [50 * 49], color=t["c"][RED], s=26, zorder=4); ax.text(52, 50 * 49 * 0.6, "2.450", fontsize=8.2, color=t["fg"])
    ax.scatter([100], [9], color=t["c"][GREEN], s=26, zorder=4); ax.text(97, 13, "9 lời gọi", fontsize=8.2, color=t["fg"], ha="right")
    ax.set_yscale("log")
    ax.set_xlabel("số ứng viên k"); ax.set_ylabel("số lời gọi LLM (log)")
    ax.set_title("Chi phí ba cách dùng LLM làm reranker")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ygrid(ax, t)


# ---------------------------------------------------------------- 9 MMR
@figure("mmr-example", size=(8.8, 3.4))
def _(fig, t):
    a1 = fig.add_axes([0.02, 0.12, 0.34, 0.74]); a2 = fig.add_axes([0.44, 0.0, 0.56, 1.0])
    from matplotlib.colors import LinearSegmentedColormap
    S = np.array([[1, .95, .92, .40, .15], [.95, 1, .93, .45, .12], [.92, .93, 1, .42, .14],
                  [.40, .45, .42, 1, .20], [.15, .12, .14, .20, 1]])
    cmap = LinearSegmentedColormap.from_list("s", [t["bg"], t["c"][VIOLET]])
    a1.imshow(S, cmap=cmap, vmin=0, vmax=1.1)
    names = ["d1", "d2", "d3", "d4", "d5"]
    a1.set_xticks(range(5), names); a1.set_yticks(range(5), names)
    for s in a1.spines.values():
        s.set_visible(False)
    a1.tick_params(length=0)
    for i in range(5):
        for j in range(5):
            a1.text(j, i, f"{S[i, j]:.2f}", ha="center", va="center", fontsize=7.6,
                    color="#ffffff" if S[i, j] > 0.6 else t["fg"])
    a1.set_title("sim(dᵢ, dⱼ)", fontsize=9.6)
    a2.set_xlim(0, 70); a2.set_ylim(0, 46); a2.axis("off")
    rel = dict(d1=.90, d2=.88, d3=.85, d4=.70, d5=.55)
    kind = dict(d1=("ticket", BLUE), d2=("ticket", BLUE), d3=("ticket", BLUE), d4=("HC chính sách", GREEN), d5=("macro", ORANGE))
    rows = [("λ = 1", ["d1", "d2", "d3"], "ba bản gần trùng"),
            ("λ = 0.7", ["d1", "d4", "d2"], "cách mở khóa + chính sách nền"),
            ("λ = 0.5", ["d1", "d5", "d4"], "đa dạng nhất, macro yếu vào sớm")]
    for i, (lab, picks, cm) in enumerate(rows):
        y = 34 - i * 13
        a2.text(1, y + 4, lab, fontsize=9.4, weight="bold", color=t["fg"], va="center")
        for j, d in enumerate(picks):
            name, c = kind[d]
            box(a2, 12 + j * 15, y, 13.5, 8, f"{d}  ({rel[d]:.2f})\n{name}", t, color=t["c"][c], fill=tint(t["c"][c], t, .14), fs=7.8)
        a2.text(12, y - 2.5, cm, fontsize=8, color=t["fg2"])
    a2.text(1, 45, "Ba tài liệu được chọn đầu tiên (rel trong ngoặc)", fontsize=9, color=t["fg"], weight="bold")


# ---------------------------------------------------------------- 10.3 selective context
@figure("selective-context", size=(8.6, 3.0))
def _(fig, t):
    ax = fig.subplots()
    toks = ["Theo", "như", "chính", "sách", ",", "hoàn", "tiền", "trong", "14", "ngày", ",", "không", "áp", "dụng", "gói", "năm", "."]
    I = np.array([2.1, 1.0, 3.2, 1.4, 0.3, 4.6, 1.2, 1.5, 6.1, 2.4, 0.3, 4.8, 2.2, 0.9, 4.1, 3.9, 0.2])
    thr = np.percentile(I, 40)
    cols = [t["c"][BLUE] if v >= thr else t["muted"] for v in I]
    x = np.arange(len(toks))
    ax.bar(x, I, color=cols, width=0.7)
    ax.axhline(thr, color=t["c"][ORANGE], ls="--", lw=1.3)
    ax.text(len(toks) - 0.5, thr + 0.15, "ngưỡng (phân vị 40)", fontsize=8, color=t["fg"], ha="right")
    ax.set_xticks(x, toks, fontsize=8.4)
    ax.set_ylabel("I(xₜ) = −log p(xₜ | x<ₜ)")
    ax.set_title("Self-information theo token (số giả định để minh họa)")
    i_k = toks.index("tiền")
    ax.annotate("«tiền», «sách» bị bỏ: văn bản bị «xé»; ở câu khác,\ncả «không» cũng có thể rơi → không dùng cho chính sách",
                xy=(i_k, I[i_k] + 0.1), xytext=(0.6, 6.4), fontsize=8.2, color=t["fg"],
                arrowprops=dict(arrowstyle="-", color=t["muted"], lw=0.8))
    ax.set_ylim(0, 7.6)
    ax.legend(handles=[Patch(color=t["c"][BLUE], label="giữ"), Patch(color=t["muted"], label="bỏ")], fontsize=8.2, loc="upper right")
    ygrid(ax, t)


# ---------------------------------------------------------------- 10.4 sắp xếp context
@figure("context-ordering", size=(8.0, 2.4))
def _(fig, t):
    ax = canvas(fig, (0, 116), (0, 30))
    for r, (lab, order) in enumerate([("Giảm dần theo điểm", [1, 2, 3, 4, 5]), ("«Hai đầu»", [1, 3, 5, 4, 2])]):
        y = 16 - r * 12
        ax.text(2, y + 3.5, lab, fontsize=9, color=t["fg"], weight="bold", va="center")
        for j, k in enumerate(order):
            strength = (6 - k) / 5
            c = t["c"][BLUE]
            box(ax, 36 + j * 15, y, 13, 7, f"#{k}", t, color=c, fill=tint(c, t, 0.12 + 0.55 * strength), fs=10,
                tc="#ffffff" if strength > 0.7 else t["fg"])
    ax.text(36, 27.5, "đầu prompt", fontsize=8.2, color=t["fg2"])
    ax.text(110, 27.5, "cuối prompt", fontsize=8.2, color=t["fg2"], ha="right")
    ax.text(2, 0.5, "Đoạn mạnh đặt ở hai đầu, đoạn yếu ở giữa — khai thác hình chữ U của «lost in the middle».",
            fontsize=8.3, color=t["fg2"])


# ---------------------------------------------------------------- 11.2 ngân sách latency
@figure("latency-budget", size=(8.6, 3.6))
def _(fig, t):
    ax = fig.subplots()
    rows = [("Phân tích email (LLM)", 0.5, 2.0, VIOLET), ("Embed truy vấn", 0.02, 0.06, AQUA),
            ("Hybrid retrieval", 0.02, 0.10, BLUE), ("RRF + gom", 0.0, 0.005, BLUE), ("Rerank ~60 (0.6B)", 0.2, 0.8, ORANGE),
            ("MMR + top-k", 0.0, 0.005, AQUA), ("Nén extractive", 0.1, 0.4, GREEN),
            ("HyDE (có điều kiện)", 1.0, 3.0, t["muted"]), ("LLM listwise (nếu bật)", 2.0, 10.0, t["muted"])]
    y = np.arange(len(rows))
    for i, (name, lo, hi, c) in enumerate(rows):
        col = t["c"][c] if isinstance(c, int) else c
        ax.barh(i, max(hi - lo, 0.004), left=max(lo, 0.001), color=col, height=0.55)
        s = f"{lo * 1000:.0f}–{hi * 1000:.0f} ms" if hi < 1 else f"{lo:g}–{hi:g} s"
        if hi < 0.01:
            s = "< 5 ms"
        ax.text(max(hi, 0.001) * 1.15, i, s, va="center", fontsize=8, color=t["fg2"])
    ax.set_yticks(y, [r[0] for r in rows]); ax.invert_yaxis()
    ax.set_xscale("log"); ax.set_xlim(0.001, 40)
    ax.set_xlabel("thời gian (giây, log) — dải ước lượng để lập kế hoạch")
    ax.set_title("Ngân sách latency trước generation: ~1–4 s nếu không bật bước xám")
    xgrid(ax, t)


if __name__ == "__main__":
    run("06")
