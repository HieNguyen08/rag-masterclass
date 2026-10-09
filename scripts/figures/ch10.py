"""Hình minh họa cho Module 10 — Đánh giá RAG, confidence và escalation."""
import numpy as np
from matplotlib.patches import Patch, Rectangle

from figkit import (AQUA, BLUE, GREEN, MAGENTA, ORANGE, RED, VIOLET, YELLOW, arrow, box,
                    canvas, figure, note, run, tint, xgrid, ygrid)


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z / (1 + z * z / n) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return c - h, c + h


# ---------------------------------------------------------------- 2.2–2.6 metric retrieval
@figure("retrieval-metrics", size=(8.8, 2.9))
def _(fig, t):
    ax = canvas(fig, (0, 124), (0, 38))
    L = ["B", "A", "D", "C", "E"]; rel = [0, 1, 0, 1, 0]
    for i, (d, r) in enumerate(zip(L, rel)):
        x = 2 + i * 11
        c = t["c"][GREEN] if r else t["line"]
        box(ax, x, 18, 9, 9, d, t, color=c, fill=tint(t["c"][GREEN], t, .2) if r else t["panel"], fs=11, weight="bold" if r else "normal")
        ax.text(x + 4.5, 29, f"#{i + 1}", ha="center", fontsize=8.4, color=t["fg2"])
        if r:
            ax.text(x + 4.5, 14.5, f"P@{i + 1} = {sum(rel[:i + 1])}/{i + 1}", ha="center", fontsize=8, color=t["fg"])
    box(ax, 59, 18, 9, 9, "F", t, color=t["c"][RED], fill=tint(t["c"][RED], t, .1), fs=11, ls="--")
    ax.text(63.5, 14.5, "không lấy về", ha="center", fontsize=8, color=t["c"][RED] if t["name"] == "light" else t["fg"])
    ax.text(2, 34.5, "Top-5 của retriever; liên quan: A, C, F (|R| = 3)", fontsize=9, weight="bold", color=t["fg"])
    vals = [("P@5", "2/5 = 0.40"), ("R@5", "2/3 ≈ 0.667"), ("RR", "1/2 = 0.50"), ("AP", "(0.5 + 0.5 + 0)/3 ≈ 0.333"),
            ("CP@5 (RAGAS)", "(0.5 + 0.5)/2 = 0.50")]
    for i, (a, b) in enumerate(vals):
        y = 30 - i * 6
        ax.text(76, y, a, fontsize=8.8, color=t["fg"], weight="bold")
        ax.text(96, y, b, fontsize=8.6, color=t["fg"])
    ax.text(2, 4, "AP phạt cả việc thiếu F; CP chỉ đo thứ tự của những gì đã lấy về.", fontsize=8.2, color=t["fg2"])


# ---------------------------------------------------------------- 2.5 nDCG
@figure("ndcg", size=(8.4, 3.0))
def _(fig, t):
    ax = fig.subplots()
    act = [2, 3, 0, 1, 0]; ideal = [3, 2, 1, 0, 0]
    disc = [1 / np.log2(i + 2) for i in range(5)]
    x = np.arange(1, 6); w = 0.36
    ga = [(2 ** r - 1) * d for r, d in zip(act, disc)]
    gi = [(2 ** r - 1) * d for r, d in zip(ideal, disc)]
    ax.bar(x - w / 2, ga, w * 0.92, color=t["c"][BLUE], label=f"thứ tự thực tế (2, 3, 0, 1, 0): DCG = {sum(ga):.3f}")
    ax.bar(x + w / 2, gi, w * 0.92, color=t["c"][GREEN], label=f"thứ tự lý tưởng (3, 2, 1, 0, 0): IDCG = {sum(gi):.3f}")
    ax.plot(x, [7 * d for d in disc], color=t["muted"], ls=":", lw=1.3, marker="o", ms=3, label="chiết khấu 1/log₂(i+1) × 7")
    ax.set_xticks(x, [f"vị trí {i}" for i in x])
    ax.set_ylabel("(2^rel − 1) / log₂(i+1)")
    ax.set_ylim(0, 9.5)
    ax.set_title(f"nDCG@5 = {sum(ga):.3f} / {sum(gi):.3f} ≈ {sum(ga) / sum(gi):.3f}", fontsize=9.6)
    ax.legend(fontsize=8, loc="upper right")
    ygrid(ax, t)


# ---------------------------------------------------------------- 2.7 pooling
@figure("pooling-labels", size=(8.6, 2.9))
def _(fig, t):
    ax = canvas(fig, (0, 124), (0, 40))
    rets = [("BM25", BLUE), ("dense", AQUA), ("hybrid", VIOLET)]
    for i, (n, c) in enumerate(rets):
        box(ax, 2, 28 - i * 10, 16, 7, f"{n} top-10", t, color=t["c"][c], fill=tint(t["c"][c], t, .15), fs=8.4)
        arrow(ax, 18.3, 31.5 - i * 10, 33.7, 21.5, t, lw=1)
    box(ax, 34, 14, 22, 15, "pool (hợp, bỏ trùng)", t, fs=8.6)
    arrow(ax, 56.3, 21.5, 63.7, 21.5, t, lw=1.6)
    box(ax, 64, 14, 24, 15, "LLM gán nhãn trước,\nngười kiểm lại mọi\n«liên quan» + 10–20%\n«không liên quan»", t,
        color=t["c"][GREEN], fill=tint(t["c"][GREEN], t, .12), fs=7.8)
    box(ax, 94, 18, 28, 9, "tài liệu liên quan mà\nkhông retriever nào lấy", t, color=t["c"][RED], fill=tint(t["c"][RED], t, .1), fs=7.8, ls="--")
    ax.text(108, 13.5, "→ không bao giờ có nhãn", ha="center", fontsize=7.8, color=t["fg2"])
    ax.text(2, 3, "Hệ quả: recall đo bằng pooling luôn là cận trên của recall thật.", fontsize=8.4, color=t["fg"])


# ---------------------------------------------------------------- 3.2 chẩn đoán vị trí lỗi
@figure("ragas-diagnosis", size=(8.8, 2.9))
def _(fig, t):
    ax = canvas(fig, (0, 124), (0, 38))
    rows = [("context recall thấp", "→ sửa retrieval", "Module 05–06", BLUE),
            ("context recall cao, faithfulness thấp", "→ model bịa: sửa prompt / model", "Module 07", VIOLET),
            ("faithfulness cao, correctness thấp", "→ context sai / lỗi thời: sửa dữ liệu", "Module 04", ORANGE)]
    for i, (a, b, m, c) in enumerate(rows):
        y = 26 - i * 10.5
        box(ax, 2, y, 46, 8, a, t, color=t["c"][c], fill=tint(t["c"][c], t, .14), fs=8.6)
        arrow(ax, 48.3, y + 4, 53.7, y + 4, t, lw=1.4)
        box(ax, 54, y, 50, 8, b, t, fs=8.4, ha="left")
        ax.text(106, y + 4, m, fontsize=8.4, color=t["fg2"], va="center")
    ax.text(2, 36, "Ba metric cùng nhau chỉ ra lỗi nằm ở đâu — giá trị hơn một điểm tổng", fontsize=9, weight="bold", color=t["fg"])


# ---------------------------------------------------------------- 3.5 PPI
@figure("ppi", size=(7.2, 2.9))
def _(fig, t):
    ax = fig.subplots()
    labels = ["judge trên\n5.000 draft", "judge trên\n300 có nhãn", "người trên\n300 có nhãn", "ước lượng PPI\n0.88 − 0.05"]
    vals = [0.88, 0.89, 0.84, 0.83]
    cols = [t["c"][BLUE], t["c"][BLUE], t["c"][GREEN], t["c"][VIOLET]]
    ax.scatter(range(4), vals, s=80, color=cols, zorder=4)
    lo, hi = wilson(252, 300)
    ax.errorbar([2], [0.84], yerr=[[0.84 - lo], [hi - 0.84]], color=t["c"][GREEN], capsize=5, lw=1.6)
    for i, v in enumerate(vals):
        ax.text(i + 0.12, v, f"{v:.2f}", va="center", fontsize=9, color=t["fg"])
    ax.text(1.5, 0.905, "độ lệch judge = 0.89 − 0.84 = +0.05", ha="center", fontsize=8.2, color=t["fg2"])
    ax.set_xticks(range(4), labels, fontsize=8)
    ax.set_xlim(-0.4, 3.6); ax.set_ylim(0.78, 0.92)
    ax.set_ylabel("tỷ lệ đạt")
    ax.set_title("PPI: khử độ lệch của judge bằng nhãn người (thanh lỗi: Wilson 95% chỉ dùng nhãn người)", fontsize=9)
    ygrid(ax, t)


# ---------------------------------------------------------------- 4.3 kappa
@figure("judge-kappa", size=(8.4, 3.1))
def _(fig, t):
    a1 = fig.add_axes([0.04, 0.12, 0.36, 0.72]); a2 = fig.add_axes([0.48, 0.0, 0.52, 1.0])
    from matplotlib.colors import LinearSegmentedColormap
    M = np.array([[140, 10], [20, 30]])
    cmap = LinearSegmentedColormap.from_list("s", [t["bg"], t["c"][BLUE]])
    a1.imshow(M, cmap=cmap, vmin=0, vmax=170)
    labs = [["đồng thuận\n140", "judge bỏ lỡ\n…"], ["judge báo nhầm", ""]]
    names = [["140\ncùng PASS", "10"], ["20", "30\ncùng FAIL"]]
    for i in range(2):
        for j in range(2):
            a1.text(j, i, names[i][j], ha="center", va="center", fontsize=9.4, color="#ffffff" if M[i, j] > 100 else t["fg"])
    a1.set_xticks([0, 1], ["judge PASS", "judge FAIL"]); a1.set_yticks([0, 1], ["người PASS", "người FAIL"])
    for s in a1.spines.values():
        s.set_visible(False)
    a1.tick_params(length=0)
    a1.set_title("200 draft", fontsize=9.4)
    a2.set_xlim(0, 60); a2.set_ylim(0, 40); a2.axis("off")
    items = [("p_o = 170/200 = 0.85", "đồng thuận quan sát", False), ("p_e = 0.75·0.80 + 0.25·0.20 = 0.65", "trùng do ngẫu nhiên", False),
             ("κ = (0.85 − 0.65)/0.35 ≈ 0.57", "mức «vừa phải»", True),
             ("Se = 30/50 = 0.60", "bỏ sót 40% draft lỗi", True), ("Sp = 140/150 ≈ 0.933", "ít báo nhầm", False)]
    for i, (a, b, bold) in enumerate(items):
        y = 34 - i * 7
        a2.text(2, y, a, fontsize=9, color=t["fg"], weight="bold" if bold else "normal")
        a2.text(2, y - 3, b, fontsize=7.8, color=t["fg2"])


# ---------------------------------------------------------------- 4.4 Rogan–Gladen
@figure("rogan-gladen", size=(6.8, 3.0))
def _(fig, t):
    ax = fig.subplots()
    pi = np.linspace(0, 0.4, 200); Se, Sp = 0.60, 0.933
    pobs = Se * pi + (1 - Sp) * (1 - pi)
    ax.plot(pi, pobs, color=t["c"][BLUE], label="p_obs = Se·π + (1 − Sp)(1 − π)")
    ax.plot(pi, pi, color=t["muted"], ls=":", lw=1.2, label="judge hoàn hảo: p_obs = π")
    pt = (0.12 + Sp - 1) / (Se + Sp - 1)
    ax.plot([0, pt, pt], [0.12, 0.12, 0], color=t["c"][ORANGE], ls="--", lw=1.2)
    ax.scatter([pt], [0.12], color=t["c"][ORANGE], s=34, zorder=4)
    ax.text(pt + 0.012, 0.125, f"judge báo 12% FAIL\n→ tỷ lệ lỗi thật π ≈ {pt:.3f}", fontsize=8.4, color=t["fg"], va="top")
    ax.text(0.2, 0.03, "tại π = 0: p_obs = 1 − Sp = 0.067\n(báo nhầm ngay cả khi không có lỗi)", fontsize=7.8, color=t["fg2"])
    ax.set_xlabel("tỷ lệ lỗi thật π"); ax.set_ylabel("tỷ lệ FAIL judge báo")
    ax.set_xlim(0, 0.4); ax.set_ylim(0, 0.4)
    ax.set_title("Hiệu chỉnh tỷ lệ lỗi quan sát (Se = 0.60, Sp = 0.933)", fontsize=9.6)
    ax.legend(fontsize=8, loc="upper left")
    ygrid(ax, t)


# ---------------------------------------------------------------- 5.1 golden set
@figure("golden-composition", size=(8.8, 2.4))
def _(fig, t):
    ax = canvas(fig, (0, 124), (0, 30))
    parts = [("FAQ trả lời được", 40, BLUE), ("nhiều câu hỏi", 10, AQUA), ("không trả lời được", 10, VIOLET),
             ("chính sách nhạy cảm", 15, ORANGE), ("muốn gặp người", 10, YELLOW), ("tấn công", 5, RED), ("đa ngôn ngữ khó", 10, MAGENTA)]
    x = 2
    for name, p, c in parts:
        w = 1.2 * p
        ax.add_patch(Rectangle((x, 16), w - 0.4, 8, color=t["c"][c], lw=0))
        ax.text(x + w / 2, 20, f"{p}%", ha="center", va="center", fontsize=8.4, color="#ffffff", weight="bold")
        x += w
    for i, (name, p, c) in enumerate(parts):
        col = i % 4; row = i // 4
        ax.add_patch(Rectangle((2 + col * 30, 9 - row * 5), 2, 2, color=t["c"][c], lw=0))
        ax.text(5 + col * 30, 10 - row * 5, name, fontsize=7.8, color=t["fg"], va="center")
    ax.text(2, 27, "Golden set: phân bố thật + cố ý dư thừa ở chỗ nguy hiểm; mỗi nhóm × 3 ngôn ngữ, 30–50 mẫu/tầng → ~800–1.000 mẫu",
            fontsize=8.6, weight="bold", color=t["fg"])


# ---------------------------------------------------------------- 6.1 khoảng tin cậy
@figure("wilson-intervals", size=(8.8, 3.0))
def _(fig, t):
    a1, a2 = fig.subplots(1, 2, gridspec_kw=dict(wspace=0.32))
    for i, (k, lab, c) in enumerate([(178, "phiên bản cũ 89%", ORANGE), (182, "phiên bản mới 91%", BLUE)]):
        lo, hi = wilson(k, 200)
        a1.errorbar([k / 200], [i], xerr=[[k / 200 - lo], [hi - k / 200]], fmt="o", color=t["c"][c], capsize=5, lw=2, ms=7)
        a1.text(hi + 0.005, i, f"[{lo:.3f}; {hi:.3f}]", va="center", fontsize=8.4, color=t["fg"])
    a1.set_yticks([0, 1], ["cũ: 178/200", "mới: 182/200"]); a1.set_ylim(-0.6, 1.6)
    a1.set_xlim(0.82, 0.99); a1.set_xlabel("tỷ lệ đạt (Wilson 95%)")
    a1.set_title("n = 200: hai khoảng chồng lấn gần hết", fontsize=9.4)
    xgrid(a1, t)
    E = np.linspace(0.01, 0.08, 200)
    for p, c in [(0.9, BLUE), (0.95, AQUA), (0.5, ORANGE)]:
        a2.plot(E, 1.96 ** 2 * p * (1 - p) / E ** 2, color=t["c"][c], label=f"p ≈ {p}")
    a2.scatter([0.03], [1.96 ** 2 * 0.09 / 0.0009], color=t["c"][BLUE], s=30, zorder=4)
    a2.text(0.032, 450, "p = 0.9, E = ±0.03\n→ n ≈ 385 mỗi tầng", fontsize=8.2, color=t["fg"])
    a2.set_yscale("log"); a2.set_xlabel("sai số mong muốn ±E"); a2.set_ylabel("cỡ mẫu n (log)")
    a2.set_title("n ≈ z²·p(1 − p) / E²", fontsize=9.4)
    a2.legend(fontsize=8)
    ygrid(a2, t)


# ---------------------------------------------------------------- 6.3 McNemar
@figure("mcnemar", size=(7.6, 2.8))
def _(fig, t):
    a1 = fig.add_axes([0.02, 0.12, 0.38, 0.72]); a2 = fig.add_axes([0.48, 0.0, 0.52, 1.0])
    M = np.array([[250, 18], [7, 25]])
    for i in range(2):
        for j in range(2):
            disc = i != j
            c = t["c"][ORANGE] if disc else t["panel2"]
            a1.add_patch(Rectangle((j, 1 - i), 1, 1, color=tint(t["c"][ORANGE], t, .35) if disc else c, ec=t["bg"], lw=2))
            a1.text(j + 0.5, 1.5 - i, ("b = " if (i, j) == (0, 1) else "c = " if (i, j) == (1, 0) else "") + str(M[i, j]),
                    ha="center", va="center", fontsize=10, color=t["fg"], weight="bold" if disc else "normal")
    a1.set_xlim(0, 2); a1.set_ylim(0, 2)
    a1.set_xticks([0.5, 1.5], ["B đúng", "B sai"]); a1.set_yticks([1.5, 0.5], ["A đúng", "A sai"])
    for s in a1.spines.values():
        s.set_visible(False)
    a1.tick_params(length=0)
    a2.set_xlim(0, 60); a2.set_ylim(0, 40); a2.axis("off")
    a2.text(2, 33, "Chỉ ô bất đồng (cam) mang thông tin", fontsize=9.2, weight="bold", color=t["fg"])
    a2.text(2, 25, "χ² = (|18 − 7| − 1)² / 25 = 4.0", fontsize=9, color=t["fg"])
    a2.text(2, 19, "p ≈ 0.046 (nhị phân chính xác: 0.043)", fontsize=9, color=t["fg"])
    a2.text(2, 11, "A tốt hơn B có ý nghĩa ở mức 5%,\ndù tổng chỉ chênh 11/300 ≈ 3.7 điểm", fontsize=8.4, color=t["fg2"])


# ---------------------------------------------------------------- 6.5 quy tắc số 3
@figure("rule-of-three", size=(7.2, 3.0))
def _(fig, t):
    ax = fig.subplots()
    n = np.arange(30, 1001)
    ax.plot(n, 1 - 0.05 ** (1 / n), color=t["c"][BLUE], label="0 lỗi: r_max = 1 − 0.05^(1/n)")
    ax.plot(n, 3 / n, color=t["muted"], ls=":", lw=1.3, label="xấp xỉ 3/n")
    from scipy.stats import beta
    ub2 = [beta.ppf(0.975, 3, nn - 2) for nn in n]
    ax.plot(n, ub2, color=t["c"][ORANGE], label="2 lỗi: cận trên Clopper–Pearson (khoảng 95% hai phía)")
    ax.axhline(0.02, color=t["c"][RED], ls="--", lw=1.2)
    ax.text(990, 0.0215, "ràng buộc 2%", ha="right", fontsize=8.2, color=t["fg"])
    for nn, dy in ((150, -0.008), (300, -0.007)):
        v = 1 - 0.05 ** (1 / nn)
        ax.scatter([nn], [v], color=t["c"][BLUE], s=28, zorder=4)

    ax.text(560, 0.042, "0 lỗi: n = 150 → 0.0198; n = 300 → 0.0099", fontsize=8.2, color=t["c"][BLUE] if t["name"] == "light" else t["fg"])
    v = beta.ppf(0.975, 3, 298)
    ax.scatter([300], [v], color=t["c"][ORANGE], s=28, zorder=4)
    ax.text(312, v + 0.004, f"2 lỗi / 300: {v:.3f} > 2%", fontsize=8.2, color=t["fg"])
    ax.set_ylim(0, 0.1); ax.set_xlim(30, 1000)
    ax.set_xlabel("số ticket kiểm tra n"); ax.set_ylabel("cận trên 95% của tỷ lệ lỗi")
    ax.set_title("Cái phải vượt qua là cận trên, không phải tỷ lệ quan sát", fontsize=9.6)
    ax.legend(fontsize=7.8, loc="upper right")
    ygrid(ax, t)


# ---------------------------------------------------------------- 7.1/7.7 risk–coverage
P10 = [0.98, 0.96, 0.95, 0.93, 0.91, 0.88, 0.85, 0.80, 0.70, 0.55]
OK10 = [1, 1, 1, 0, 1, 1, 0, 1, 0, 0]


@figure("risk-coverage", size=(8.8, 3.1))
def _(fig, t):
    a1, a2 = fig.subplots(1, 2, gridspec_kw=dict(wspace=0.3, width_ratios=[1.2, 1]))
    cov = np.arange(1, 11) / 10
    risk = np.array([(j - sum(OK10[:j])) / j for j in range(1, 11)])
    a1.step(cov, risk, where="post", color=t["c"][BLUE], lw=2)
    a1.scatter(cov, risk, s=28, color=[t["c"][GREEN] if o else t["c"][RED] for o in OK10], zorder=4)
    a1.fill_between(cov, 0, risk, step="post", color=t["c"][BLUE], alpha=0.1, lw=0)
    for c, r, p in zip(cov, risk, P10):
        a1.text(c - 0.012, r + 0.02, f"{p:.2f}", ha="right", fontsize=7, color=t["fg2"])
    a1.set_xlabel("coverage (tỷ lệ tự gửi)"); a1.set_ylabel("risk (tỷ lệ sai trong số đã gửi)")
    a1.set_ylim(0, 0.5); a1.set_xlim(0.05, 1.03)
    a1.set_title(f"10 ticket xếp theo p̂ (số trên điểm): AURC ≈ {risk.mean():.3f}", fontsize=9.4)
    a1.legend(handles=[Patch(color=t["c"][GREEN], label="ticket đúng"), Patch(color=t["c"][RED], label="ticket sai")], fontsize=8, loc="upper left")
    ygrid(a1, t)
    a2.set_xlim(0, 60); a2.set_ylim(0, 40); a2.axis("off")
    steps = ["1. Trên tập hiệu chuẩn của intent:\n   với mỗi τ đếm n_τ và k_τ lỗi", "2. Tính cận trên 95% của risk",
             "3. Chọn τ nhỏ nhất có cận trên ≤ 2%", "4. τ cuối = max(τ_chi phí, τ_rủi ro)"]
    for i, s in enumerate(steps):
        box(a2, 1, 30 - i * 9.3, 58, 7.6, s, t, color=t["c"][VIOLET], fill=tint(t["c"][VIOLET], t, .1), fs=8, ha="left", radius=0.6)
    a2.text(1, -1.5, "intent ít dữ liệu → cận trên rộng → chưa được tự động hóa", fontsize=7.8, color=t["fg2"])


# ---------------------------------------------------------------- 7.3 semantic entropy
@figure("semantic-entropy", size=(8.4, 2.6))
def _(fig, t):
    ax = canvas(fig, (0, 124), (0, 34))
    def row(y, groups, se, lab):
        ax.text(2, y + 4, lab, fontsize=8.8, color=t["fg"], weight="bold", va="center")
        x = 30
        cols = [t["c"][BLUE], t["c"][ORANGE], t["c"][VIOLET]]
        for gi, g in enumerate(groups):
            for _ in range(g):
                box(ax, x, y, 7, 8, "", t, color=cols[gi], fill=tint(cols[gi], t, .3), radius=0.6)
                x += 8
            x += 4
        ax.text(x + 2, y + 4, f"SE = {se}", fontsize=9, color=t["fg"], va="center")
    row(19, [5], "0", "5 mẫu cùng nghĩa")
    row(5, [3, 1, 1], "−(0.6 ln 0.6 + 2·0.2 ln 0.2) ≈ 0.950", "3 cụm: 3 · 1 · 1")
    ax.text(2, 31, "Entropy ngữ nghĩa trên N = 5 câu trả lời lấy mẫu (tối đa ln 5 ≈ 1.609)", fontsize=9, weight="bold", color=t["fg"])


# ---------------------------------------------------------------- 7.4 mô hình meta
@figure("meta-model", size=(8.8, 3.0))
def _(fig, t):
    ax = canvas(fig, (0, 124), (0, 40))
    sig = [("(a) retrieval: s₁, s₁ − s₂", BLUE), ("(b) xác suất token", AQUA), ("(c) confidence tự khai", YELLOW),
           ("(d) entropy ngữ nghĩa", VIOLET), ("(e) verifier: groundedness, unanswered", GREEN), ("(f) ngữ cảnh: intent, ngôn ngữ", ORANGE)]
    for i, (s, c) in enumerate(sig):
        y = 33 - i * 6
        box(ax, 2, y, 44, 5, s, t, color=t["c"][c], fill=tint(t["c"][c], t, .12), fs=7.8, ha="left", radius=0.6)
        arrow(ax, 46.3, y + 2.5, 57.7, 20, t, lw=0.9)
    box(ax, 58, 13, 26, 14, "logistic\np̂ = σ(wᵀz + b)", t, color=t["c"][VIOLET], fill=tint(t["c"][VIOLET], t, .15), fs=8.8)
    arrow(ax, 84.3, 20, 90.7, 20, t, lw=1.6)
    box(ax, 91, 13, 31, 14, "hiệu chuẩn\n(temperature / Platt)\ntrên tập riêng", t, color=t["c"][GREEN], fill=tint(t["c"][GREEN], t, .15), fs=8.2)
    ax.text(58, 6, "nhãn y: agent gửi gần nguyên văn → 1;\nviết lại / bỏ draft → 0 (nhiễu, kiểm lại 5–10%)", fontsize=7.8, color=t["fg2"])


# ---------------------------------------------------------------- 7.5 reliability + calibration
@figure("reliability", size=(8.8, 3.3))
def _(fig, t):
    a1, a2 = fig.subplots(1, 2, gridspec_kw=dict(wspace=0.3))
    n = [50, 80, 150, 270, 450]; conf = [0.12, 0.31, 0.52, 0.71, 0.91]; acc = [0.20, 0.35, 0.45, 0.60, 0.78]
    a1.plot([0, 1], [0, 1], color=t["muted"], ls=":", lw=1.2, label="hiệu chuẩn hoàn hảo")
    a1.bar(conf, acc, width=0.12, color=t["c"][BLUE], alpha=0.85, label="acc thực tế mỗi khoảng")
    for c, a, k in zip(conf, acc, n):
        a1.plot([c, c], [a, c], color=t["c"][RED], lw=2)
        a1.text(c, a - 0.07 if a > 0.3 else a + 0.03, f"n={k}", ha="center", fontsize=7.4, color=t["fg2"] if a <= 0.3 else "#ffffff")
    a1.set_xlim(0, 1); a1.set_ylim(0, 1)
    a1.set_xlabel("confidence tự khai trung bình"); a1.set_ylabel("tỷ lệ đúng thực tế")
    a1.set_title("Reliability diagram: ECE ≈ 0.106, MCE = 0.13", fontsize=9.4)
    a1.legend(fontsize=7.8, loc="upper left")
    ygrid(a1, t)
    p = np.linspace(0.01, 0.99, 300); u = np.log(p / (1 - p))
    a2.plot(p, p, color=t["muted"], ls=":", lw=1.2, label="không đổi")
    a2.plot(p, 1 / (1 + np.exp(-u / 1.8)), color=t["c"][ORANGE], label="temperature T = 1.8")
    a2.plot(p, 1 / (1 + np.exp(-(0.6 * u - 0.2))), color=t["c"][VIOLET], label="Platt a = 0.6, b = −0.2")
    for f, c, lab in [(1 / (1 + np.exp(-np.log(19) / 1.8)), ORANGE, "0.837"), (1 / (1 + np.exp(-(0.6 * np.log(19) - 0.2))), VIOLET, "0.827")]:
        a2.scatter([0.95], [f], color=t["c"][c], s=28, zorder=4)
    a2.text(0.97, 0.12, "p̂ = 0.95 → 0.837 (T)\n/ 0.827 (Platt)", ha="right", fontsize=8, color=t["fg"])
    a2.set_xlabel("p̂ thô"); a2.set_ylabel("p̂ sau hiệu chuẩn")
    a2.set_title("Biến đổi đơn điệu: sửa con số, không sửa thứ hạng", fontsize=9.4)
    a2.legend(fontsize=7.8, loc="upper left")
    ygrid(a2, t)


# ---------------------------------------------------------------- 7.6 chi phí ba hành động
@figure("three-action-cost", size=(9.2, 3.2))
def _(fig, t):
    a1, a2 = fig.subplots(1, 2, gridspec_kw=dict(wspace=0.62, width_ratios=[1.25, 1]))
    p = np.linspace(0, 1, 400); Cw, Cr, Cf, Ch = 8, 0.5, 1.5, 1.5
    send = (1 - p) * Cw; draft = Cr + (1 - p) * Cf; esc = np.full_like(p, Ch)
    a1.plot(p, send, color=t["c"][GREEN], label="SEND: (1 − p)·C_w")
    a1.plot(p, draft, color=t["c"][BLUE], label="DRAFT: C_r + (1 − p)·C_fix")
    a1.plot(p, esc, color=t["c"][RED], label="ESCALATE: C_h")
    best = np.minimum(np.minimum(send, draft), esc)
    a1.plot(p, best, color=t["fg"], lw=3.5, alpha=0.18)
    for x, lab in [(1 / 3, "0.333"), (1 - 0.5 / 6.5, "0.923")]:
        a1.axvline(x, color=t["muted"], ls="--", lw=1)
        a1.text(x, 3.6, lab, ha="center", fontsize=8.2, color=t["fg"])
    a1.text(0.16, 2.6, "ESC", ha="center", fontsize=9, color=t["fg"], weight="bold")
    a1.text(0.63, 2.6, "DRAFT", ha="center", fontsize=9, color=t["fg"], weight="bold")
    a1.text(0.965, 2.6, "SEND", ha="center", fontsize=8.6, color=t["fg"], weight="bold")
    a1.set_ylim(0, 4); a1.set_xlabel("p̂ đã hiệu chuẩn"); a1.set_ylabel("chi phí kỳ vọng (USD, giả định)")
    a1.set_title("how_to: C_w = 8, C_r = 0.5, C_fix = C_h = 1.5", fontsize=9.4)
    a1.legend(fontsize=7.6, loc="lower left")
    ygrid(a1, t)
    intents = ["how_to", "feature_question", "bug_report", "account_access"]
    Cws = [8, 10, 15, 100]
    taus = [1 - 1.5 / c for c in Cws]
    a2.barh(range(4), taus, color=[t["c"][BLUE], t["c"][AQUA], t["c"][ORANGE], t["c"][RED]], height=0.55)
    for i, (tv, c) in enumerate(zip(taus, Cws)):
        a2.text(tv + 0.005, i, f"{tv:.4g} (C_w = {c})", va="center", fontsize=8, color=t["fg"])
    a2.set_yticks(range(4), intents); a2.invert_yaxis(); a2.set_xlim(0.7, 1.12)
    a2.set_xlabel("τ* = 1 − C_h / C_w (chỉ SEND vs ESC)")
    a2.set_title("Ngưỡng theo intent; billing_refund luôn có người", fontsize=9.2)
    xgrid(a2, t)


# ---------------------------------------------------------------- 8.1 NED
@figure("ned-levels", size=(8.4, 2.4))
def _(fig, t):
    ax = canvas(fig, (0, 124), (0, 30))
    x0, x1 = 4, 120
    X = lambda v: x0 + (x1 - x0) * v
    bands = [(0, 0.1, "y = 1", "nguyên văn /\nchỉnh nhỏ", GREEN), (0.1, 0.4, "y = 0 (hoặc nhãn mềm)", "sửa một phần nội dung", YELLOW),
             (0.4, 1.0, "y = 0", "viết lại / bỏ draft", RED)]
    for a, b, lab, desc, c in bands:
        ax.add_patch(Rectangle((X(a), 13), X(b) - X(a), 7, color=tint(t["c"][c], t, .35), lw=0))
        ax.text((X(a) + X(b)) / 2, 16.5, lab, ha="center", va="center", fontsize=8, color=t["fg"], weight="bold")
        ax.text((X(a) + X(b)) / 2, 22, desc, ha="center", va="bottom", fontsize=7.6, color=t["fg2"])
    for v in (0, 0.1, 0.4, 1.0):
        ax.text(X(v), 10, f"{v:g}", ha="center", fontsize=8, color=t["fg2"])
    ax.annotate("ví dụ: 10 thao tác / 46 từ ≈ 0.22", xy=(X(0.217), 12.8), xytext=(X(0.45), 7), fontsize=8.2,
                color=t["fg"], arrowprops=dict(arrowstyle="-|>", color=t["fg2"], lw=1))
    ax.text(4, 1.5, "NED = ED_từ(draft, bản gửi) / max(|d|, |f|) — tính sau khi bỏ lời chào và chữ ký", fontsize=8.2, color=t["fg2"])


# ---------------------------------------------------------------- 8.3 cỡ mẫu A/B
@figure("ab-sample-size", size=(7.2, 3.0))
def _(fig, t):
    ax = fig.subplots()
    d = np.linspace(0.01, 0.08, 300); p1 = 0.85
    n = 7.84 * (p1 * (1 - p1) + (p1 - d) * (1 - p1 + d)) / d ** 2
    ax.plot(d, n, color=t["c"][BLUE], label="phản hồi CSAT mỗi nhóm")
    ax.plot(d, n / 0.2, color=t["c"][ORANGE], ls="--", label="ticket mỗi nhóm (20% khách trả lời khảo sát)")
    nd = 7.84 * 0.2751 / 0.0009
    ax.scatter([0.03, 0.03], [nd, nd / 0.2], color=[t["c"][BLUE], t["c"][ORANGE]], s=30, zorder=4)
    ax.text(0.031, nd * 0.55, "δ = 0.03: ≈ 2.400 phản hồi/nhóm", fontsize=8.2, color=t["fg"])
    ax.text(0.031, nd / 0.2 * 1.25, "≈ 12.000 ticket/nhóm", fontsize=8.2, color=t["fg"])
    ax.set_yscale("log"); ax.set_xlabel("mức giảm CSAT muốn phát hiện δ (từ 0.85)")
    ax.set_ylabel("cỡ mẫu (log)")
    ax.set_title("CSAT là thước đo chậm: α = 5% hai phía, power 80%", fontsize=9.6)
    ax.legend(fontsize=8, loc="upper right")
    ygrid(ax, t)


# ---------------------------------------------------------------- 9.2 gate
@figure("eval-gates", size=(8.8, 3.0))
def _(fig, t):
    ax = canvas(fig, (0, 124), (0, 40))
    ax.text(2, 37, "Gate an toàn: ngưỡng tuyệt đối", fontsize=9, weight="bold", color=t["fg"])
    abs_g = ["recall nhóm phải escalate ≥ 0.98; không case must_escalate nào bị SEND", "không vi phạm must_not nào", "ECE ≤ 0.05 trên tập hiệu chuẩn"]
    for i, s in enumerate(abs_g):
        box(ax, 2, 28 - i * 7, 58, 5.6, s, t, color=t["c"][RED], fill=tint(t["c"][RED], t, .1), fs=7.6, ha="left", radius=0.6)
    ax.text(64, 37, "Gate chất lượng: so cặp có kiểm định với baseline", fontsize=9, weight="bold", color=t["fg"])
    rel = ["groundedness giảm có ý nghĩa (McNemar p < 0.05)", "Recall@k giảm > 2 điểm ở một ngôn ngữ (bootstrap)", "token / p95 latency tăng > 20%"]
    for i, s in enumerate(rel):
        box(ax, 64, 28 - i * 7, 58, 5.6, s, t, color=t["c"][BLUE], fill=tint(t["c"][BLUE], t, .1), fs=7.6, ha="left", radius=0.6)
    ax.text(2, 3, "Chạy trên mọi PR đổi prompt/model/retrieval/policy (tập smoke ~150 case), toàn bộ golden set hằng đêm và trước rollout.",
            fontsize=8, color=t["fg2"])


if __name__ == "__main__":
    run("10")
