"""Hình minh họa cho Module 13 — RAG đa phương thức."""
import numpy as np
from matplotlib.patches import Rectangle

from figkit import (AQUA, BLUE, GREEN, MAGENTA, ORANGE, RED, VIOLET, YELLOW, arrow, box,
                    canvas, figure, run, tint, xgrid, ygrid)


def ink(t, c):
    return t["c"][c] if t["name"] == "light" else t["fg"]


def vn(x, fmt="{:.2f}"):
    return fmt.format(x).replace(".", ",")


# ---------------------------------------------------------------- 1.2 mất mát do parse
@figure("parse-loss", size=(8.8, 2.8))
def _(fig, t):
    ax = canvas(fig, (0, 124), (0, 38))
    steps = [("Bố cục", 0.95, AQUA), ("OCR", 0.97, AQUA), ("Cấu trúc bảng", 0.80, ORANGE),
             ("Retrieval", 0.85, BLUE), ("Generation", 0.90, VIOLET)]
    cum = 1.0
    for k, (a, p, c) in enumerate(steps):
        x = 2 + k * 21
        cum *= p
        box(ax, x, 12, 17, 13, "", t, color=t["c"][c], fill=tint(t["c"][c], t, .15), lw=2 if c == ORANGE else 1.4)
        ax.text(x + 8.5, 21, a, ha="center", fontsize=8.4, weight="bold", color=t["fg"])
        ax.text(x + 8.5, 15.5, "× " + vn(p), ha="center", fontsize=9.6, color=t["fg"])
        ax.text(x + 8.5, 8.5, "còn " + vn(cum), ha="center", fontsize=7.8, color=t["fg2"])
        if k < 4:
            arrow(ax, x + 17.3, 18.5, x + 20.7, 18.5, t, lw=1.3)
    ax.plot([2, 63], [5.5, 5.5], color=t["c"][ORANGE], lw=1.2)
    ax.text(32.5, 1.5, "parse: P(I) ≈ 0,74 — lỗi ở đây không sửa được ở khâu sau", ha="center", fontsize=7.8, color=ink(t, ORANGE))
    box(ax, 107, 12, 15, 13, "≈ 0,56", t, color=t["c"][RED], fill=tint(t["c"][RED], t, .15), fs=10.5, weight="bold")
    arrow(ax, 101.3, 18.5, 106.7, 18.5, t, lw=1.3)
    ax.text(2, 34, "Xác suất cả chuỗi đúng (giá trị giả định của mục 1.2)", fontsize=9, weight="bold", color=t["fg"])


# ---------------------------------------------------------------- 2.1 visual token
@figure("visual-tokens", size=(8.8, 3.1))
def _(fig, t):
    a1, a2 = fig.subplots(1, 2, gridspec_kw=dict(wspace=0.35, width_ratios=[1.15, 1]))
    W, H = 1920, 1088
    a1.add_patch(Rectangle((0, 0), W, H, fc=t["panel"], ec=t["fg2"], lw=1))
    for x in range(0, W + 1, 32):
        a1.plot([x, x], [0, H], color=t["grid"], lw=0.35)
    for y in range(0, H + 1, 32):
        a1.plot([0, W], [y, y], color=t["grid"], lw=0.35)
    # vài "phần tử giao diện"
    a1.add_patch(Rectangle((0, H - 96), W, 96, fc=tint(t["c"][BLUE], t, .35), ec="none"))
    a1.add_patch(Rectangle((480, 448), 960, 192, fc=tint(t["c"][RED], t, .3), ec=t["c"][RED], lw=1))
    a1.text(960, 544, "ERR-4012: Thanh toán thất bại", ha="center", va="center", fontsize=7.6, color=t["fg"])
    a1.add_patch(Rectangle((480, 448), 32, 32, fc=t["c"][ORANGE], ec="none"))
    a1.annotate("1 token = ô 32×32 px", xy=(496, 464), xytext=(80, 220), fontsize=7.8, color=t["fg"],
                arrowprops=dict(arrowstyle="-|>", color=t["fg2"], lw=1))
    a1.set_xlim(-20, W + 20); a1.set_ylim(-20, H + 20); a1.set_aspect("equal"); a1.axis("off")
    a1.set_title("Full HD 1920×1080 → 60 × 34 = 2.040 token", fontsize=9)
    labs = ["ảnh chụp\nmàn hình\nFull HD", "trang A4\n150 dpi\n(ảnh)", "trang A4\n(văn bản OCR,\nước lượng)"]
    vals = [2040, 2145, 650]
    cols = [t["c"][BLUE], t["c"][VIOLET], t["c"][GREEN]]
    a2.bar(range(3), vals, color=cols, width=0.6)
    a2.errorbar([2], [650], yerr=[[150], [150]], fmt="none", ecolor=t["fg"], capsize=4, lw=1)
    for k, v in enumerate(vals):
        a2.text(k, v + (190 if k == 2 else 40), ("~500–800" if k == 2 else f"{v:,}".replace(",", ".")), ha="center", fontsize=8.2, color=t["fg"])
    a2.set_xticks(range(3), labs, fontsize=7.6)
    a2.set_ylim(0, 2600)
    a2.set_ylabel("token")
    a2.set_title("Ảnh trang đắt gấp ~3–4 lần văn bản", fontsize=9)
    ygrid(a2, t)


# ---------------------------------------------------------------- 2.2 SigLIP
@figure("siglip-example", size=(8.4, 2.9))
def _(fig, t):
    a1, a2 = fig.subplots(1, 2, gridspec_kw=dict(wspace=0.45))
    S = np.array([[0.8, 0.1], [0.2, 0.7]]); tt, b = 10, -5
    z = np.where(np.eye(2) == 1, 1, -1)
    L = -np.log(1 / (1 + np.exp(-z * (tt * S + b))))
    for ax, M, title, fmt in [(a1, S, "Tương đồng xᵢᵀyⱼ", "{:.1f}"), (a2, L, "−ln σ(zᵢⱼ(t·xᵢᵀyⱼ + b))", "{:.3f}")]:
        mx = M.max()
        for i in range(2):
            for j in range(2):
                c = t["c"][GREEN] if i == j else t["c"][RED]
                ax.add_patch(Rectangle((j, i), 0.95, 0.95, fc=tint(c, t, 0.15 + 0.55 * M[i, j] / mx), ec="none"))
                ax.text(j + 0.475, i + 0.475, vn(M[i, j], fmt), ha="center", va="center", fontsize=10, weight="bold", color=t["fg"])
                ax.text(j + 0.475, i + 0.2, "cặp đúng" if i == j else "cặp sai", ha="center", fontsize=7, color=t["fg2"])
        ax.set_xlim(0, 2); ax.set_ylim(2, 0)
        ax.set_xticks([0.475, 1.475], ["văn bản 1", "văn bản 2"])
        ax.set_yticks([0.475, 1.475], ["ảnh 1", "ảnh 2"])
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title, fontsize=9)
    a2.text(1.0, 2.35, "t = 10, b = −5 → loss = (0,049 + 0,018 + 0,049 + 0,127) / 2 ≈ 0,121",
            ha="center", fontsize=7.8, color=t["fg2"])


# ---------------------------------------------------------------- 2.3 modality gap
@figure("modality-gap", size=(8.8, 3.1))
def _(fig, t):
    a1, a2 = fig.subplots(1, 2, gridspec_kw=dict(wspace=0.3, width_ratios=[1.1, 1]))
    rng = np.random.default_rng(3)
    ct = np.array([-0.55, 0.35]); ci = np.array([0.55, -0.25])
    T = ct + rng.normal(0, 0.12, (40, 2)); I = ci + rng.normal(0, 0.12, (40, 2))
    a1.scatter(T[:, 0], T[:, 1], s=14, color=t["c"][BLUE], alpha=0.75, label="embedding văn bản")
    a1.scatter(I[:, 0], I[:, 1], s=14, color=t["c"][ORANGE], alpha=0.75, label="embedding ảnh")
    q = ct + np.array([0.12, 0.1])
    a1.plot(*q, "*", color=t["fg"], ms=12, label="query (văn bản)")
    a1.annotate("", xy=ci, xytext=ct, arrowprops=dict(arrowstyle="<->", color=t["fg2"], lw=1.2, ls="--"))
    a1.text(0.12, 0.22, "modality gap", fontsize=8, color=t["fg2"], ha="left")
    a1.set_xlim(-1.1, 1.1); a1.set_ylim(-0.75, 0.9); a1.set_aspect("equal")
    a1.set_xticks([]); a1.set_yticks([])
    a1.legend(fontsize=7.4, loc="lower left")
    a1.set_title("Hai modality nằm ở hai vùng tách biệt", fontsize=9)
    labs = ["ảnh sơ đồ\n(liên quan 0,6)", "đoạn văn bản\n(liên quan 0,3)"]
    raw = [0.7, 0.8]; cen = [0.6, 0.3]
    x = np.arange(2); w = 0.36
    a2.bar(x - w / 2, raw, w * .92, color=t["muted"], label="điểm thô (cộng thưởng modality)")
    a2.bar(x + w / 2, cen, w * .92, color=t["c"][GREEN], label="sau khi trừ trung bình modality")
    for xi, a, b in zip(x, raw, cen):
        a2.text(xi - w / 2, a + 0.02, vn(a, "{:.1f}"), ha="center", fontsize=8, color=t["fg"])
        a2.text(xi + w / 2, b + 0.02, vn(b, "{:.1f}"), ha="center", fontsize=8, color=t["fg"])
    a2.set_xticks(x, labs, fontsize=7.8)
    a2.set_ylim(0, 1.25)
    a2.legend(fontsize=7.4, loc="upper left")
    a2.set_title("Thứ hạng đảo lại khi khử modality gap", fontsize=9)
    ygrid(a2, t)


# ---------------------------------------------------------------- 4.2 MaxSim
@figure("maxsim-example", size=(8.8, 2.9))
def _(fig, t):
    Q = np.array([[1, 0], [0, 1.]])
    pages = [("Trang A", np.array([[0.9, 0.1], [0.2, 0.8], [0.5, 0.5]])), ("Trang B", np.full((3, 2), 0.5))]
    axs = fig.subplots(1, 2, gridspec_kw=dict(wspace=0.5))
    for ax, (name, D) in zip(axs, pages):
        M = Q @ D.T
        for i in range(2):
            j_best = int(M[i].argmax())
            for j in range(3):
                best = j == j_best
                ax.add_patch(Rectangle((j, i), 0.95, 0.95, fc=tint(t["c"][BLUE], t, 0.12 + 0.6 * M[i, j]),
                                       ec=t["c"][ORANGE] if best else "none", lw=2.2))
                ax.text(j + 0.475, i + 0.5, vn(M[i, j], "{:.1f}"), ha="center", va="center", fontsize=9.6,
                        color=t["fg"], weight="bold" if best else "normal")
            ax.text(3.15, i + 0.5, "max " + vn(M[i, j_best], "{:.1f}"), va="center", fontsize=8.2, color=t["fg"])
        total = M.max(1).sum()
        ax.set_xlim(0, 4.2); ax.set_ylim(2, 0)
        ax.set_xticks([0.475, 1.475, 2.475], ["patch 1", "patch 2", "patch 3"], fontsize=7.8)
        ax.set_yticks([0.5, 1.5], ["«webhook»", "«Pro»"], fontsize=8)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"{name}: MaxSim = {vn(total, '{:.1f}')}  ·  mean-pool = 0,5", fontsize=9)


# ---------------------------------------------------------------- 4.3 lưu trữ
@figure("storage-cost", size=(8.4, 2.6))
def _(fig, t):
    ax = fig.subplots()
    names = ["ColPali FP16 (1.030 × 128)", "+ token pooling giảm 3 lần", "ColPali nhị phân (1 bit/chiều)", "Một vector 2.048 chiều FP16"]
    mb = [1213, 404, 75.8, 18.8]
    labs = ["~1,21 GB", "~0,40 GB", "~76 MB", "~19 MB"]
    cols = [t["c"][RED], t["c"][ORANGE], t["c"][YELLOW], t["c"][GREEN]]
    y = np.arange(4)[::-1]
    for yi, v, c in zip(y, mb, cols):
        ax.plot([10, v], [yi, yi], color=t["grid"], lw=2, zorder=1)
        ax.plot(v, yi, "o", color=c, ms=9, zorder=2)
    for yi, v, lab in zip(y, mb, labs):
        ax.text(v * 1.18, yi, lab, va="center", fontsize=8.4, color=t["fg"])
    ax.set_xscale("log")
    ax.set_xlim(10, 4000)
    ax.set_yticks(y, names, fontsize=8.2)
    ax.set_ylim(-0.6, 3.6)
    ax.set_xlabel("dung lượng chỉ mục cho 4.600 trang (MB, thang log)")
    ax.set_title("Đa vector đắt gấp ~64 lần một vector — nhưng vẫn nhỏ ở quy mô vài nghìn trang", fontsize=9)
    xgrid(ax, t)


# ---------------------------------------------------------------- 4.5 ViDoRe
@figure("vidore-findings", size=(8.8, 3.0))
def _(fig, t):
    a1, a2 = fig.subplots(1, 2, gridspec_kw=dict(wspace=0.55))
    v = [81.3, 67.0]
    a1.bar(range(2), v, color=[t["c"][VIOLET], t["c"][BLUE]], width=0.55)
    for k, x in enumerate(v):
        a1.text(k, x + 1.5, vn(x, "{:.1f}"), ha="center", fontsize=8.6, color=t["fg"], weight="bold")
    a1.set_xticks(range(2), ["ColPali\n(ảnh trang)", "Unstructured + chú thích\n+ BGE-M3 (văn bản)"], fontsize=7.8)
    a1.set_ylim(0, 100); a1.set_ylabel("nDCG@5 trung bình")
    a1.set_title("ViDoRe V1 (Faysse et al., 2024)", fontsize=9)
    ygrid(a1, t)
    g = [0.602, 0.089, 0.065]
    a2.bar(range(3), g, color=[t["muted"], t["c"][ORANGE], t["c"][ORANGE]], width=0.55)
    for k, x in enumerate(g):
        a2.text(k, x + 0.015, vn(x, "{:.3f}"), ha="center", fontsize=8.4, color=t["fg"])
    a2.set_xticks(range(3), ["người gán\nnhãn", "Qwen3-VL-\n30B-A3B", "Gemini 3\nPro"], fontsize=7.8)
    a2.set_ylim(0, 0.75); a2.set_ylabel("F1 định vị vùng chứng cứ")
    a2.set_title("ViDoRe V3: định vị vùng còn rất yếu", fontsize=9)
    ygrid(a2, t)


# ---------------------------------------------------------------- 7.1 CER
@figure("cer-example", size=(8.8, 2.3))
def _(fig, t):
    ax = canvas(fig, (0, 124), (0, 30))
    ref = "Không thể kết nối"; hyp = "Khong the ket noi"
    cw = 5.6
    x0 = 18
    ax.text(2, 19.5, "đúng", fontsize=8.4, color=t["fg2"], va="center")
    ax.text(2, 11.5, "OCR", fontsize=8.4, color=t["fg2"], va="center")
    for k, (a, b) in enumerate(zip(ref, hyp)):
        x = x0 + k * cw
        bad = a != b
        for y, ch in [(16.5, a), (8.5, b)]:
            if ch == " ":
                continue
            box(ax, x, y, cw - 0.6, 6, ch, t, color=t["c"][RED] if bad else t["line"],
                fill=tint(t["c"][RED], t, .2) if bad else t["panel"], fs=9, radius=0.4, weight="bold" if bad else "normal")
    ax.text(2, 27, "Mất dấu tiếng Việt: 4 ký tự sai trên 17", fontsize=9, weight="bold", color=t["fg"])
    ax.text(x0, 3, "CER = 4/17 ≈ 23,5%   ·   WER = 4/4 = 100% (từ nào cũng sai)", fontsize=8.4, color=t["fg"])


# ---------------------------------------------------------------- 7.2 phân tầng
@figure("stratified-recall", size=(7.6, 2.8))
def _(fig, t):
    ax = fig.subplots()
    labs = ["đáp án trong đoạn chữ\n(n = 200)", "đáp án trong bảng / hình\n(n = 50)"]
    p = np.array([0.90, 0.60]); n = np.array([200, 50])
    ci = 1.96 * np.sqrt(p * (1 - p) / n)
    ax.bar(range(2), p, color=[t["c"][BLUE], t["c"][ORANGE]], width=0.5)
    ax.errorbar(range(2), p, yerr=ci, fmt="none", ecolor=t["fg"], capsize=5, lw=1.1)
    for k in range(2):
        ax.text(k + 0.3, p[k], f"{vn(p[k])} ± {vn(ci[k])}", va="center", fontsize=8.2, color=t["fg"])
    ax.axhline(0.84, color=t["fg2"], ls="--", lw=1)
    ax.text(1.62, 0.87, "tổng gộp 0,84", ha="right", fontsize=8, color=t["fg2"])
    ax.set_xticks(range(2), labs, fontsize=8)
    ax.set_xlim(-0.5, 1.75)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Recall@5")
    ax.set_title("Con số tổng che mất phân tầng yếu (giả định, khoảng tin cậy 95% xấp xỉ Wald)", fontsize=9)
    ygrid(ax, t)


if __name__ == "__main__":
    run("13")
