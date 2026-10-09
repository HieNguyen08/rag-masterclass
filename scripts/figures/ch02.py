"""Hình minh họa cho Module 02 — Giới hạn LLM & hình thức hóa RAG."""
import numpy as np
from matplotlib.patches import Rectangle

from figkit import (AQUA, BLUE, GREEN, MAGENTA, ORANGE, RED, VIOLET, YELLOW, arrow, box,
                    canvas, figure, run, tint, xgrid, ygrid)


def ink(t, c):
    """Màu chữ nhấn: dùng màu ở theme sáng, chữ thường ở theme tối."""
    return t["c"][c] if t["name"] == "light" else t["fg"]


# ---------------------------------------------------------------- 1.1 intrinsic / extrinsic
@figure("intrinsic-extrinsic", size=(8.8, 2.9))
def _(fig, t):
    ax = canvas(fig, (0, 124), (0, 40))
    box(ax, 2, 12, 30, 16, "", t, color=t["muted"], fill=t["panel"])
    ax.text(4, 24.5, "Nguồn (tài liệu chính sách)", fontsize=8.4, weight="bold", color=t["fg"])
    ax.text(4, 17, "«Hoàn tiền trong 14 ngày\nkể từ ngày thanh toán.»", fontsize=8.2, color=t["fg"], va="center")
    outs = [("Trung thành", "«…hoàn tiền trong 14 ngày»", "khớp nguồn", GREEN),
            ("Intrinsic", "«…hoàn tiền trong 30 ngày»", "mâu thuẫn với nguồn", RED),
            ("Extrinsic", "«…và không mất phí xử lý»", "nguồn không nhắc tới → không kiểm chứng được", ORANGE)]
    for k, (a, b, c, col) in enumerate(outs):
        y = 29 - k * 12
        box(ax, 44, y, 78, 9.5, "", t, color=t["c"][col], fill=tint(t["c"][col], t, .13))
        ax.text(46, y + 4.75, a, va="center", fontsize=8.8, weight="bold", color=ink(t, col))
        ax.text(64, y + 6.4, b, va="center", fontsize=8.2, color=t["fg"])
        ax.text(64, y + 2.6, c, va="center", fontsize=7.8, color=t["fg2"])
        arrow(ax, 32.3, 20, 43.7, y + 4.75, t, lw=1.1, color=t["line"])
    ax.text(2, 37, "Câu AI viết, so với nguồn được cung cấp", fontsize=9, weight="bold", color=t["fg"])
    ax.text(2, 4, "Extrinsic có thể tình cờ đúng,\nnhưng vẫn là lỗi quy trình.", fontsize=7.8, color=t["fg2"])


# ---------------------------------------------------------------- 1.2 ngưỡng trả lời
@figure("abstain-threshold", size=(8.0, 3.0))
def _(fig, t):
    ax = fig.subplots()
    p = np.linspace(0, 1, 300)
    for c, col in [(0, BLUE), (1, ORANGE), (9, RED)]:
        th = c / (1 + c)
        ax.plot(p, p - c * (1 - p), color=t["c"][col], lw=1.8,
                label=f"c = {c}: trả lời khi p > {th:g}".replace(".", ","))
        ax.plot(th, 0, "o", color=t["c"][col], ms=6)
    ax.axhline(0, color=t["fg2"], lw=1)
    ax.text(0.62, -0.08, "từ chối = 0 điểm", fontsize=7.8, color=t["fg2"], va="top")
    ax.fill_between(p, 0, 1.15, color=t["c"][GREEN], alpha=0.06, lw=0)
    ax.set_xlim(0, 1); ax.set_ylim(-3.2, 1.15)
    ax.set_xlabel("p = xác suất câu trả lời đúng")
    ax.set_ylabel("điểm kỳ vọng khi trả lời")
    ax.legend(fontsize=7.8, loc="lower right")
    ax.set_title("Điểm kỳ vọng p − c(1 − p): phạt sai càng nặng, ngưỡng tự tin để trả lời càng cao", fontsize=9)
    ygrid(ax, t)


# ---------------------------------------------------------------- 3.1 lost in the middle (sơ đồ)
@figure("lost-in-middle", size=(7.6, 2.8))
def _(fig, t):
    ax = fig.subplots()
    x = np.linspace(0, 1, 200)
    y = 0.52 + 0.9 * (x - 0.5) ** 2 + 0.05 * x
    ax.plot(x, y, color=t["c"][VIOLET], lw=2)
    ax.axhline(0.58, color=t["c"][RED], ls="--", lw=1.1)
    ax.text(0.5, 0.505, "đường đứt: mức closed-book (không đưa tài liệu), ở một số cấu hình", ha="center", fontsize=7.8, color=ink(t, RED))
    for xx, lab in [(0.02, "đầu context"), (0.5, "giữa"), (0.98, "cuối")]:
        ax.text(xx, 0.44, lab, ha="center", fontsize=8, color=t["fg2"])
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(0.42, 0.82)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("vị trí của tài liệu chứa đáp án trong context")
    ax.set_ylabel("độ chính xác")
    ax.set_title("Hình dạng chữ U (sơ đồ định tính, không phải số liệu đo)", fontsize=9)


# ---------------------------------------------------------------- 3.3 FLOPs prefill
@figure("prefill-flops", size=(8.8, 3.0))
def _(fig, t):
    a1, a2 = fig.subplots(1, 2, gridspec_kw=dict(wspace=0.32, width_ratios=[1.3, 1]))
    N, L, d = 6.7e9, 32, 4096
    n = np.logspace(3, 6, 300)
    lin = 2 * N * n; quad = 2 * L * n ** 2 * d
    a1.plot(n, lin, color=t["c"][BLUE], lw=1.8, label="2Nn (nhân ma trận trọng số)")
    a1.plot(n, quad, color=t["c"][ORANGE], lw=1.8, label="2Ln²d (attention)")
    a1.plot(n, lin + quad, color=t["fg"], lw=1.2, ls="--", label="tổng")
    for nn in [8000, 640000]:
        a1.axvline(nn, color=t["line"], ls=":", lw=1)
        a1.text(nn * 1.1, 2e12, f"{nn // 1000}K", fontsize=8, color=t["fg2"])
    a1.set_xscale("log"); a1.set_yscale("log")
    a1.set_ylim(1e12, 1e19)
    a1.set_xlabel("độ dài prompt n (token)")
    a1.set_ylabel("FLOPs prefill")
    a1.legend(fontsize=7.6, loc="upper left")
    a1.set_title("Cấu hình kiểu 7B: số hạng bậc hai vượt lên", fontsize=9)
    ygrid(a1, t)
    ns = [8000, 640000]
    sh = [2 * L * k ** 2 * d / (2 * N * k + 2 * L * k ** 2 * d) for k in ns]
    labs = ["8K token", "640K token"]
    a2.bar(range(2), [1 - s for s in sh], color=t["c"][BLUE], width=0.55, label="nhân ma trận trọng số")
    a2.bar(range(2), sh, bottom=[1 - s for s in sh], color=t["c"][ORANGE], width=0.55, label="attention")
    for k, s in enumerate(sh):
        a2.text(k, 1 - s / 2, f"{s:.0%}", ha="center", va="center", fontsize=8.6, color="white", weight="bold")
    a2.set_xticks(range(2), labs)
    a2.set_ylim(0, 1.32)
    a2.set_yticks([0, .25, .5, .75, 1], ["0%", "25%", "50%", "75%", "100%"])
    a2.legend(fontsize=7.6, loc="upper center", ncol=2, handlelength=1, columnspacing=0.8)
    a2.set_title("Tỉ trọng attention; tổng tăng ~930 lần", fontsize=9)
    ygrid(a2, t)


# ---------------------------------------------------------------- 3.3 chi phí tiền
@figure("context-cost", size=(8.4, 2.5))
def _(fig, t):
    ax = fig.subplots()
    names = ["RAG (~6.000 token/lời gọi)", "Long-context + prompt caching", "Long-context nhồi toàn bộ Help Center"]
    vals = [31.5, 340, 3390]
    cols = [t["c"][GREEN], t["c"][ORANGE], t["c"][RED]]
    y = np.arange(3)[::-1]
    ax.barh(y, vals, color=cols, height=0.55)
    for yi, v, lab in zip(y, vals, ["~$31,5", "~$340 + phí ghi cache", "~$3.390"]):
        ax.text(v + 40, yi, lab, va="center", fontsize=8.4, color=t["fg"])
    ax.set_yticks(y, names, fontsize=8.2)
    ax.set_xlim(0, 4200)
    ax.set_xlabel("chi phí input mỗi ngày (USD, giá giả định $1 / 1 triệu token, 5.250 lời gọi)")
    ax.set_title("Cùng lưu lượng, ba chiến lược đưa tri thức vào prompt", fontsize=9)
    xgrid(ax, t)


# ---------------------------------------------------------------- 4.3 ma trận quyết định
@figure("strategy-matrix", size=(8.8, 3.6))
def _(fig, t):
    ax = fig.subplots()
    cols = ["Prompt-only", "RAG", "Fine-tune", "Long-context", "CAG"]
    rows = ["Tri thức riêng, đổi hằng tuần", "Kho lớn (≫ context)", "Trích dẫn / kiểm chứng",
            "Phân quyền tenant / ACL", "Chi phí mỗi request", "Độ trễ", "Văn phong, định dạng", "Độ phức tạp vận hành"]
    M = [["−−", "++", "−−", "+", "−"],
         ["−−", "++", "−", "−−", "−−"],
         ["−−", "++", "−−", "+", "+"],
         ["−−", "++", "−−", "−", "−"],
         ["++", "+", "++", "−−", "+"],
         ["++", "+", "++", "−−", "+"],
         ["+", "+", "++", "+", "+"],
         ["++", "−", "−", "+", "−"]]
    score = {"++": 2, "+": 1, "−": -1, "−−": -2}
    for i, r in enumerate(M):
        for j, v in enumerate(r):
            s = score[v]
            c = t["c"][GREEN] if s > 0 else t["c"][RED]
            ax.add_patch(Rectangle((j, i), 0.96, 0.92, fc=tint(c, t, 0.18 + 0.27 * abs(s)), ec="none"))
            ax.text(j + 0.48, i + 0.46, v, ha="center", va="center", fontsize=9.4, color=t["fg"], weight="bold")
    ax.add_patch(Rectangle((1, 0), 0.96, 8 - 0.08, fc="none", ec=t["fg"], lw=1.6))
    ax.set_xlim(0, 5); ax.set_ylim(8, 0)
    ax.set_xticks(np.arange(5) + 0.48, cols, fontsize=8.4)
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(8) + 0.46, rows, fontsize=8.2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)


# ---------------------------------------------------------------- 5.3 RAG-Sequence vs RAG-Token
@figure("rag-seq-vs-token", size=(8.8, 3.1))
def _(fig, t):
    a1, a2, a3 = fig.subplots(1, 3, gridspec_kw=dict(wspace=0.42, width_ratios=[1, 1.35, 0.9]))
    pz = np.array([0.629, 0.231, 0.140])
    zl = ["z₁ bảng giá\nhiện hành", "z₂ blog\ncũ 2023", "z₃ hướng dẫn\nmời user"]
    cz = [t["c"][GREEN], t["c"][ORANGE], t["c"][BLUE]]
    a1.bar(range(3), pz, color=cz, width=0.6)
    for k, v in enumerate(pz):
        a1.text(k, v + 0.02, f"{v:.3f}".replace(".", ","), ha="center", fontsize=7.8, color=t["fg"])
    a1.set_xticks(range(3), zl, fontsize=7.2)
    a1.set_ylim(0, 1); a1.set_title("Retriever p_η(z | x)", fontsize=9)
    ygrid(a1, t)
    p1 = [0.9, 0.5, 0.1]; p2 = [0.8, 0.1, 0.3]
    w = 0.26
    for k, (v, lab, c) in enumerate([(p1, "p(y₁ = «10» | z)", t["c"][VIOLET]), (p2, "p(y₂ = «user» | z, y₁)", t["c"][AQUA]),
                                     ([a * b for a, b in zip(p1, p2)], "p(y | z) cả chuỗi", t["muted"])]):
        a2.bar(np.arange(3) + (k - 1) * w, v, w * .92, color=c, label=lab)
        for xi, vi in zip(range(3), v):
            a2.text(xi + (k - 1) * w, vi + 0.02, f"{vi:g}".replace(".", ","), ha="center", fontsize=6.8, color=t["fg"])
    a2.set_xticks(range(3), ["z₁", "z₂", "z₃"])
    a2.set_ylim(0, 1.3); a2.legend(fontsize=7.2, loc="upper right")
    a2.set_title("Generator theo từng tài liệu", fontsize=9)
    ygrid(a2, t)
    res = [0.468, 0.395]
    a3.bar(range(2), res, color=[t["c"][GREEN], t["c"][VIOLET]], width=0.55)
    for k, v in enumerate(res):
        a3.text(k, v + 0.02, f"{v:.3f}".replace(".", ","), ha="center", fontsize=8.4, color=t["fg"], weight="bold")
    a3.set_xticks(range(2), ["RAG-\nSequence", "RAG-\nToken"])
    a3.set_ylim(0, 1)
    a3.set_title("p(y | x)", fontsize=9)
    ygrid(a3, t)


# ---------------------------------------------------------------- 5.4 REALM posterior
@figure("realm-posterior", size=(7.8, 2.9))
def _(fig, t):
    ax = fig.subplots()
    prior = np.array([0.629, 0.231, 0.140]); post = np.array([0.966, 0.025, 0.009])
    x = np.arange(3); w = 0.34
    ax.bar(x - w / 2, prior, w * .92, color=t["muted"], label="trước khi biết đáp án: p(z | x)")
    ax.bar(x + w / 2, post, w * .92, color=t["c"][BLUE], label="posterior: p(z | x, y)")
    for k in x:
        g = post[k] - prior[k]
        ax.text(k - w / 2, prior[k] + 0.02, f"{prior[k]:.3f}".replace(".", ","), ha="center", fontsize=7.6, color=t["fg"])
        ax.text(k + w / 2, post[k] + 0.02, f"{post[k]:.3f}".replace(".", ","), ha="center", fontsize=7.6, color=t["fg"])
        col = t["c"][GREEN] if g > 0 else t["c"][RED]
        ax.text(k, 1.12, f"gradient {g:+.3f}".replace(".", ",").replace("-", "−"), ha="center", fontsize=8.4,
                color=ink(t, GREEN if g > 0 else RED), weight="bold")
    ax.set_xticks(x, ["z₁ bảng giá hiện hành", "z₂ blog cũ 2023", "z₃ hướng dẫn mời user"])
    ax.set_ylim(0, 1.25)
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.legend(fontsize=7.8, loc="center right")
    ax.set_title("Điểm retriever được đẩy theo hướng posterior − prior", fontsize=9)
    ygrid(ax, t)


# ---------------------------------------------------------------- 5.5 FiD
@figure("fid-cost", size=(8.4, 3.2))
def _(fig, t):
    ax = canvas(fig, (0, 118), (0, 46))
    k = 5; S = 30
    x0, y0 = 6, 6
    ax.add_patch(Rectangle((x0, y0), S, S, fc=tint(t["c"][RED], t, .35), ec=t["c"][RED], lw=1.2))
    for i in range(1, k):
        ax.plot([x0 + i * S / k] * 2, [y0, y0 + S], color=t["panel"], lw=0.8)
        ax.plot([x0, x0 + S], [y0 + i * S / k] * 2, color=t["panel"], lw=0.8)
    ax.text(x0 + S / 2, 40, "Nối k tài liệu rồi encode", ha="center", fontsize=9, weight="bold", color=t["fg"])
    ax.text(x0 + S / 2, 1.5, "chi phí ∝ (kℓ)²", ha="center", fontsize=8.6, color=t["fg"])
    x1 = 50
    ax.add_patch(Rectangle((x1, y0), S, S, fc="none", ec=t["line"], lw=1, ls="--"))
    for i in range(k):
        ax.add_patch(Rectangle((x1 + i * S / k, y0 + S - (i + 1) * S / k), S / k, S / k,
                               fc=tint(t["c"][GREEN], t, .45), ec=t["c"][GREEN], lw=1))
    ax.text(x1 + S / 2, 40, "FiD: encode từng (x, zⱼ) riêng", ha="center", fontsize=9, weight="bold", color=t["fg"])
    ax.text(x1 + S / 2, 1.5, "chi phí ∝ k·ℓ²", ha="center", fontsize=8.6, color=t["fg"])
    ax.text(88, 30, "Ví dụ k = 100, ℓ = 250:", fontsize=8.4, color=t["fg"], weight="bold")
    ax.text(88, 24, "(kℓ)² = 6,25·10⁸\nk·ℓ² = 6,25·10⁶\n→ rẻ hơn 100 lần", fontsize=8.4, color=t["fg"], va="top", linespacing=1.5)
    ax.text(88, 9, "sau đó decoder\ncross-attend lên mọi\ntrạng thái ẩn đã nối", fontsize=7.8, color=t["fg2"], va="center")
    ax.text(x0 + S / 2, 37, "(minh họa k = 5)", ha="center", fontsize=7.6, color=t["fg2"])
    ax.text(x1 + S / 2, 37, "(minh họa k = 5)", ha="center", fontsize=7.6, color=t["fg2"])


# ---------------------------------------------------------------- 6.1 xác suất nhân chuỗi
@figure("pipeline-chain", size=(8.8, 2.6))
def _(fig, t):
    ax = canvas(fig, (0, 124), (0, 34))
    steps = [("Kho có đáp án", 0.95, AQUA), ("Đáp án lọt top-k", 0.85, BLUE), ("LLM dùng đúng", 0.90, VIOLET)]
    cum = 1
    for k, (a, p, c) in enumerate(steps):
        x = 2 + k * 30
        cum *= p
        box(ax, x, 10, 24, 14, "", t, color=t["c"][c], fill=tint(t["c"][c], t, .15))
        ax.text(x + 12, 19.5, a, ha="center", fontsize=8.6, weight="bold", color=t["fg"])
        ax.text(x + 12, 14, f"× {p:.2f}".replace(".", ","), ha="center", fontsize=10, color=t["fg"])
        arrow(ax, x + 24.3, 17, x + 29.7, 17, t, lw=1.4)
        ax.text(x + 12, 6.5, f"còn {cum:.3f}".replace(".", ","), ha="center", fontsize=7.8, color=t["fg2"])
    box(ax, 92, 10, 30, 14, "P(đúng) ≈ 0,73", t, color=t["c"][ORANGE], fill=tint(t["c"][ORANGE], t, .18), fs=10, weight="bold")
    ax.text(2, 30.5, "Ba khâu đều khá tốt, nhân lại thì không (giá trị minh họa của mục 6.1)", fontsize=9, weight="bold", color=t["fg"])
    ax.text(2, 1.5, "Mỗi khâu cần được đo riêng (Module 10) — chỉ đo đầu ra cuối thì không biết sửa khâu nào.",
            fontsize=7.8, color=t["fg2"])


# ---------------------------------------------------------------- 6.2 bảy điểm hỏng
@figure("failure-points", size=(8.8, 3.3))
def _(fig, t):
    ax = canvas(fig, (0, 124), (0, 46))
    stages = [("Kho tri thức", 2), ("Retrieve top-k", 27), ("Ghép context", 52), ("LLM sinh", 77), ("Câu trả lời", 102)]
    for k, (s, x) in enumerate(stages):
        box(ax, x, 26, 20, 9, s, t, color=t["c"][BLUE], fill=tint(t["c"][BLUE], t, .14), fs=8.4, weight="bold")
        if k < 4:
            arrow(ax, x + 20.3, 30.5, x + 24.7, 30.5, t, lw=1.4)
    fps = [(12, "FP1", "Missing content"), (37, "FP2", "Missed top-k"), (62, "FP3", "Not in context"),
           (87, "FP4", "Not extracted")]
    for x, a, b in fps:
        box(ax, x - 9, 12, 18, 9, f"{a}\n{b}", t, color=t["c"][RED], fill=tint(t["c"][RED], t, .12), fs=7.6)
        ax.plot([x, x], [21.2, 25.8], color=t["c"][RED], lw=1.1)
    for k, (a, b) in enumerate([("FP5", "Wrong format"), ("FP6", "Incorrect specificity"), ("FP7", "Incomplete")]):
        y = 18 - k * 7
        box(ax, 100, y - 3, 22, 6, f"{a} {b}", t, color=t["c"][RED], fill=tint(t["c"][RED], t, .12), fs=7.4)
    ax.plot([112, 112], [21.2, 25.8], color=t["c"][RED], lw=1.1)
    ax.text(2, 42, "Bảy điểm hỏng của Barnett et al. (2024) trên pipeline naive", fontsize=9, weight="bold", color=t["fg"])
    ax.text(2, 3, "Thêm cho case Zendesk: rò rỉ giữa tenant (ở khâu retrieve), prompt injection (ở khâu LLM).",
            fontsize=7.8, color=ink(t, MAGENTA))


# ---------------------------------------------------------------- 6.3 naive / advanced / modular
@figure("rag-paradigms", size=(8.8, 3.4))
def _(fig, t):
    ax = canvas(fig, (0, 124), (0, 50))

    def chain(y, items, x0=16, w=17, gap=4, hl=()):
        x = x0
        for k, it in enumerate(items):
            c = t["c"][ORANGE] if k in hl else t["c"][BLUE]
            box(ax, x, y, w, 6.5, it, t, color=c, fill=tint(c, t, .15), fs=7.6)
            if k < len(items) - 1:
                arrow(ax, x + w + 0.2, y + 3.25, x + w + gap - 0.2, y + 3.25, t, lw=1.1)
            x += w + gap
    ax.text(2, 44.5, "Naive", fontsize=9, weight="bold", color=t["fg"])
    chain(42, ["Indexing", "Retrieve", "Generate"])
    ax.text(2, 31, "Advanced", fontsize=9, weight="bold", color=t["fg"])
    chain(28.5, ["Indexing", "Pre-retrieval\n(rewrite, metadata)", "Retrieve", "Post-retrieval\n(rerank, nén)", "Generate"], hl=(1, 3))
    ax.text(2, 13, "Modular", fontsize=9, weight="bold", color=t["fg"])
    box(ax, 16, 9, 14, 7, "Router\n(intent)", t, color=t["c"][VIOLET], fill=tint(t["c"][VIOLET], t, .15), fs=7.6)
    mods = [("Retrieve\nHelp Center", 19), ("Tool / API\ntài khoản", 9.5), ("Truy xuất\nlặp", 0.5)]
    for name, y in mods:
        box(ax, 40, y, 18, 7, name, t, color=t["c"][BLUE], fill=tint(t["c"][BLUE], t, .15), fs=7.4)
        arrow(ax, 30.3, 12.5, 39.7, y + 3.5, t, lw=1)
        arrow(ax, 58.3, y + 3.5, 67.7, 12.5, t, lw=1)
    box(ax, 68, 9, 16, 7, "Generate", t, color=t["c"][BLUE], fill=tint(t["c"][BLUE], t, .15), fs=7.6)
    box(ax, 92, 9, 18, 7, "Tự đánh giá", t, color=t["c"][VIOLET], fill=tint(t["c"][VIOLET], t, .15), fs=7.6)
    arrow(ax, 84.3, 12.5, 91.7, 12.5, t, lw=1)
    arrow(ax, 101, 8.7, 49, 0.3, t, lw=1, ls="--", rad=-0.12, color=t["c"][VIOLET])
    ax.text(112, 12.5, "→ vòng\nlặp lại", fontsize=7.4, color=t["fg2"], va="center")


if __name__ == "__main__":
    run("02")
