"""Hình minh họa cho Module 03 — Embedding & biểu diễn ngữ nghĩa."""
import numpy as np
from scipy.stats import skew

from figkit import (AQUA, BLUE, GREEN, MAGENTA, ORANGE, RED, VIOLET, YELLOW, arrow, box,
                    canvas, figure, note, run, tint, xgrid, ygrid)


# ---------------------------------------------------------------- 1.1 one-hot vs dense
@figure("onehot-vs-dense", size=(8.8, 3.3))
def _(fig, t):
    from matplotlib.patches import Ellipse
    a1, a2 = fig.subplots(1, 2, gridspec_kw=dict(width_ratios=[1, 1.4], wspace=0.22))
    q = np.array([1, 1, 1, 0, 0, 0]); d1 = np.array([1, 0, 1, 1, 0, 0]); d2 = np.array([0, 0, 0, 0, 1, 1])
    cos = lambda a, b: a @ b / np.linalg.norm(a) / np.linalg.norm(b)
    vals = [cos(q, d1), cos(q, d2)]
    labels = ["d₁ «không đăng\nnhập được»", "d₂ «sign-in error»\n(bài đúng)"]
    bars = a1.bar(labels, vals, width=0.55, color=[t["c"][BLUE], t["c"][ORANGE]])
    for b, v in zip(bars, vals):
        a1.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.3f}", ha="center", color=t["fg"], fontsize=9.5)
    a1.set_ylim(0, 1); a1.set_ylabel("cos(q, d), vector đếm từ")
    a1.set_title("Đếm từ: q = «không login được»", fontsize=10)
    note(a1, 0.62, 0.78, "điểm của d₁ chỉ đến từ\n«không», «được»", t, ha="center")
    ygrid(a1, t)

    a2.set_title("Embedding tốt (sơ đồ minh họa)", fontsize=10)
    a2.set_xlim(-1.2, 1.25); a2.set_ylim(-1.05, 1.05); a2.set_xticks([]); a2.set_yticks([])
    for s in a2.spines.values():
        s.set_visible(False)
    a2.add_patch(Ellipse((0.42, 0.25), 1.45, 1.3, color=tint(t["c"][AQUA], t, 0.2), lw=0))
    pts = [
        (0.05, 0.62, "không login được (q)", BLUE, "left"),
        (0.30, 0.38, "không đăng nhập được", BLUE, "left"),
        (0.45, 0.10, "sign-in error", ORANGE, "left"),
        (0.25, -0.18, "パスワード再設定後に\nサインインできない", AQUA, "left"),
        (-0.95, -0.70, "xuất hóa đơn PDF", VIOLET, "left"),
        (-0.95, 0.72, "đổi gói Pro", MAGENTA, "left"),
    ]
    for x, y, s, c, ha in pts:
        a2.scatter([x], [y], s=46, color=t["c"][c], zorder=3, edgecolor=t["bg"], linewidth=1.2)
        a2.text(x + 0.06, y, s, fontsize=8.6, color=t["fg"], va="center", ha=ha)
    note(a2, 0.15, -0.98, "cùng ý nghĩa → gần nhau, dù không trùng chữ hay khác ngôn ngữ", t, ha="center")


# ---------------------------------------------------------------- 1.2 skip-gram


# ---------------------------------------------------------------- 1.3 ba vùng Zendesk
# ---------------------------------------------------------------- 2. ba kiến trúc
@figure("scoring-architectures", size=(9.0, 3.9))
def _(fig, t):
    ax = canvas(fig, (0, 120), (0, 52))
    cB, cX, cC = t["c"][BLUE], t["c"][ORANGE], t["c"][AQUA]
    tokq, tokd = t["c"][BLUE], t["c"][VIOLET]

    def col_title(x, s, c):
        ax.text(x, 49, s, ha="center", fontsize=10.5, weight="bold", color=t["fg"])
        ax.plot([x - 17, x + 17], [46.5, 46.5], color=c, lw=2.5, solid_capstyle="round")

    # Bi-encoder
    col_title(20, "Bi-encoder", cB)
    box(ax, 4, 38, 13, 5, "query q", t, color=tokq, fill=tint(tokq, t, .15), fs=9)
    box(ax, 23, 38, 13, 5, "tài liệu d", t, color=tokd, fill=tint(tokd, t, .15), fs=9)
    box(ax, 4, 26, 13, 6, "Encoder", t, fs=9)
    box(ax, 23, 26, 13, 6, "Encoder\n(offline)", t, fs=8.6)
    for x in (10.5, 29.5):
        arrow(ax, x, 38, x, 32, t)
    for i in range(6):
        ax.add_patch(__import__("matplotlib").patches.Rectangle((6.5 + i * 1.3, 18), 1.1, 3.5, color=tokq))
        ax.add_patch(__import__("matplotlib").patches.Rectangle((25.5 + i * 1.3, 18), 1.1, 3.5, color=tokd))
    for x in (10.5, 29.5):
        arrow(ax, x, 26, x, 21.8, t)
    box(ax, 12, 6, 16, 6, "cos / dot", t, color=cB, fill=tint(cB, t, .15), fs=9.5)
    arrow(ax, 10.5, 17.5, 17, 12.2, t); arrow(ax, 29.5, 17.5, 23, 12.2, t)
    note(ax, 20, 1.5, "1 vector/tài liệu · tính trước được", t, ha="center")

    # Cross-encoder
    col_title(60, "Cross-encoder", cX)
    xs = 42.5
    for s, w, c in [("[CLS]", 6, t["muted"]), ("q", 9, tokq), ("[SEP]", 6, t["muted"]), ("d", 13, tokd)]:
        box(ax, xs, 38, w, 5, s, t, color=c, fill=tint(c, t, .15), fs=8.5, radius=0.8)
        xs += w + 0.6
    box(ax, 44, 23, 32, 10, "Transformer\nmọi token q ⇄ mọi token d", t, fs=9)
    arrow(ax, 60, 38, 60, 33.2, t)
    box(ax, 52, 6, 16, 6, "wᵀh + b", t, color=cX, fill=tint(cX, t, .15), fs=9.5)
    arrow(ax, 60, 23, 60, 12.2, t)
    note(ax, 60, 1.5, "chính xác nhất · k lần forward cho k ứng viên", t, ha="center")

    # ColBERT
    col_title(100, "Late interaction (ColBERT)", cC)
    box(ax, 84, 38, 13, 5, "query q", t, color=tokq, fill=tint(tokq, t, .15), fs=9)
    box(ax, 103, 38, 13, 5, "tài liệu d", t, color=tokd, fill=tint(tokd, t, .15), fs=9)
    box(ax, 84, 26, 13, 6, "Encoder", t, fs=9)
    box(ax, 103, 26, 13, 6, "Encoder\n(offline)", t, fs=8.6)
    for x in (90.5, 109.5):
        arrow(ax, x, 38, x, 32, t); arrow(ax, x, 26, x, 22.3, t)
    for j in range(3):
        for i in range(4):
            ax.add_patch(__import__("matplotlib").patches.Rectangle((86.5 + j * 2.9, 18 + i * 1.0), 2.4, 0.8, color=tokq))
    for j in range(4):
        for i in range(4):
            ax.add_patch(__import__("matplotlib").patches.Rectangle((104.2 + j * 2.9, 18 + i * 1.0), 2.4, 0.8, color=tokd))
    box(ax, 89, 6, 22, 6, "Σᵢ maxⱼ qᵢᵀdⱼ  (MaxSim)", t, color=cC, fill=tint(cC, t, .15), fs=9)
    arrow(ax, 90.5, 17.5, 96, 12.2, t); arrow(ax, 109.5, 17.5, 104, 12.2, t)
    note(ax, 100, 1.5, "1 vector/token · tính trước được · lưu nhiều", t, ha="center")


# ---------------------------------------------------------------- 2.4 MaxSim
@figure("colbert-maxsim", size=(8.0, 2.8))
def _(fig, t):
    S1 = np.array([[0.9, 0.2, 0.1], [0.3, 0.7, 0.4]])
    S2 = np.array([[0.5, 0.5, 0.4], [0.6, 0.3, 0.6]])
    axs = fig.subplots(1, 2, gridspec_kw=dict(wspace=0.45))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("s", [t["bg"], t["c"][BLUE]])
    for ax, S, name in [(axs[0], S1, "d⁽¹⁾"), (axs[1], S2, "d⁽²⁾")]:
        ax.imshow(S, cmap=cmap, vmin=0, vmax=1.1, aspect="auto")
        ax.set_xticks(range(3), [f"tok {j + 1}" for j in range(3)])
        ax.set_yticks(range(2), ["q₁ «reset»", "q₂ «mật khẩu»"])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
        total = 0
        for i in range(2):
            mx = S[i].max(); total += mx
            for j in range(3):
                is_max = j == int(np.argmax(S[i]))
                ax.text(j, i, f"{S[i, j]:.1f}", ha="center", va="center", fontsize=10,
                        color="#ffffff" if S[i, j] > 0.55 else t["fg"], weight="bold" if is_max else "normal")
                if is_max:
                    ax.add_patch(__import__("matplotlib").patches.Rectangle((j - 0.46, i - 0.44), 0.92, 0.88,
                                 fill=False, ec=t["c"][ORANGE], lw=2.2))
        parts = " + ".join(f"{S[i].max():.1f}" for i in range(2))
        ax.set_title(f"{name}:  f = {parts} = {total:.1f}")
    fig.text(0.5, -0.04, "Khung cam: token khớp nhất với mỗi token query (max theo hàng); điểm = tổng các max.",
             ha="center", fontsize=8.6, color=t["fg2"])


# ---------------------------------------------------------------- 3.3 nhiệt độ
COS = np.array([0.82, 0.75, 0.40, 0.30])


def softmax_tau(tau):
    z = COS / tau
    e = np.exp(z - z.max())
    return e / e.sum()


@figure("infonce-temperature", size=(8.6, 3.3))
def _(fig, t):
    a1, a2 = fig.subplots(1, 2, gridspec_kw=dict(width_ratios=[1.15, 1], wspace=0.32))
    taus = [1.0, 0.1, 0.05]
    names = ["d⁺ (0.82)", "d₂ hard (0.75)", "d₃ (0.40)", "d₄ (0.30)"]
    cols = [t["c"][BLUE], t["c"][ORANGE], t["c"][YELLOW], t["c"][MAGENTA]]
    x = np.arange(len(taus)); w = 0.19
    for k in range(4):
        ps = [softmax_tau(tau)[k] for tau in taus]
        a1.bar(x + (k - 1.5) * w, ps, width=w * 0.9, color=cols[k], label=names[k])
    for i, tau in enumerate(taus):
        p = softmax_tau(tau)
        a1.text(i - 1.5 * w, p[0] + 0.025, f"{p[0]:.2f}", ha="center", fontsize=8.5, color=t["fg"])
        a1.text(i, -0.17, f"loss = {-np.log(p[0]):.3f}", ha="center", fontsize=8.5, color=t["fg2"])
    a1.set_xticks(x, [f"τ = {tau}" for tau in taus]); a1.set_ylim(0, 1.22); a1.set_yticks(np.arange(0, 1.01, 0.2))
    a1.set_ylabel("xác suất softmax p(dⱼ)")
    a1.set_title("Softmax trên 4 ứng viên theo τ")
    a1.legend(ncol=4, loc="upper left", fontsize=7.8, columnspacing=0.8, handlelength=1.2)
    ygrid(a1, t)

    tt = np.logspace(-2, 0, 200)
    P = np.array([softmax_tau(v) for v in tt])
    share_hard = P[:, 1] / (1 - P[:, 0])
    a2.plot(tt, P[:, 0], color=t["c"][BLUE], label="p(d⁺)")
    a2.plot(tt, share_hard, color=t["c"][ORANGE], label="tỉ trọng của d₂ trong\ngradient đẩy (p₂ / Σ p_neg)")
    a2.set_xscale("log"); a2.set_ylim(0, 1.02)
    a2.set_xlabel("nhiệt độ τ (log)")
    a2.axvspan(0.01, 0.05, color=t["grid"], alpha=0.6, lw=0)
    note(a2, 0.0105, 0.06, "τ thường dùng\n0.01–0.05", t, fs=8)
    a2.set_title("τ nhỏ → gradient dồn vào hard negative")
    a2.legend(loc="upper right", fontsize=8.0, bbox_to_anchor=(1.0, 1.0), framealpha=1)
    a2.set_ylim(0, 1.35); a2.set_yticks(np.arange(0, 1.01, 0.2))
    ygrid(a2, t)


# ---------------------------------------------------------------- 3.4 gradient hình học
@figure("infonce-gradient", size=(5.4, 4.2))
def _(fig, t):
    ax = fig.subplots()
    ax.set_aspect("equal"); ax.axis("off")
    th = np.linspace(-0.25, np.pi + 0.25, 300)
    ax.plot(np.cos(th), np.sin(th), color=t["line"], lw=1)
    angq = np.deg2rad(90)
    ang = {"d⁺": 90 - np.degrees(np.arccos(.82)), "d₂": 90 + np.degrees(np.arccos(.75)),
           "d₃": 90 - np.degrees(np.arccos(.40)), "d₄": 90 + np.degrees(np.arccos(.30))}
    p = softmax_tau(0.05)
    vec = {k: np.array([np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))]) for k, a in ang.items()}
    zq = np.array([np.cos(angq), np.sin(angq)])
    keys = ["d⁺", "d₂", "d₃", "d₄"]
    cols = [t["c"][BLUE], t["c"][ORANGE], t["c"][YELLOW], t["c"][MAGENTA]]
    for k, c, pk in zip(keys, cols, p):
        v = vec[k]
        ax.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=c, lw=1.8))
        off = 1.13
        ax.text(v[0] * off, v[1] * off, f"{k}\np={pk:.3f}" if pk >= 1e-3 else f"{k}\np≈{pk:.0e}", ha="center", va="center", fontsize=8.6, color=t["fg"])
    ax.annotate("", xy=zq, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=t["fg"], lw=2.2))
    ax.text(zq[0], zq[1] + 0.1, "z_q", ha="center", fontsize=10, color=t["fg"], weight="bold")
    centroid = sum(pk * vec[k] for k, pk in zip(keys, p))
    ax.scatter(*centroid, s=40, color=t["c"][VIOLET], zorder=4)
    ax.text(centroid[0] - 0.02, centroid[1] - 0.14, "E_p[z_d]", fontsize=8.6, color=t["c"][VIOLET], ha="center")
    g = -(centroid - vec["d⁺"])  # hướng cập nhật của z_q (âm gradient)
    g = g / np.linalg.norm(g) * 0.45
    ax.annotate("", xy=zq + g, xytext=zq, arrowprops=dict(arrowstyle="-|>", color=t["c"][GREEN], lw=2.4))
    ax.text(*(zq + g + np.array([0.04, 0.05])), "−∇ℓ: kéo về d⁺,\nđẩy khỏi d₂", fontsize=8.6, color=t["fg"])
    ax.set_xlim(-1.35, 1.45); ax.set_ylim(-0.3, 1.35)
    ax.set_title("InfoNCE (τ = 0.05): chỉ d₂ thực sự tạo lực đẩy")


# ---------------------------------------------------------------- 3.6 cận MI
@figure("infonce-mi-bound", size=(5.6, 3.0))
def _(fig, t):
    ax = fig.subplots()
    B = np.logspace(1, 13, 200, base=2)
    ax.plot(B, np.log(B), color=t["c"][BLUE])
    for b in (32, 256, 4096):
        ax.scatter([b], [np.log(b)], color=t["c"][ORANGE], zorder=3, s=36)
        ax.text(b / 1.3, np.log(b) + 0.35, f"B={b}\n{np.log(b):.2f} nats", ha="right", fontsize=8.5, color=t["fg"])
    ax.set_xscale("log", base=2)
    ax.set_xlabel("batch size B (số ứng viên mỗi query)")
    ax.set_ylabel("log B (nats)")
    ax.set_ylim(0, 10.5)
    ax.set_title("Trần của cận dưới I(Q;D) ≥ log B − L: tăng theo log B")
    ygrid(ax, t)


# ---------------------------------------------------------------- 3.7 alignment & uniformity
def _align_uniform(Zx, Zy, Z):
    la = np.mean(np.sum((Zx - Zy) ** 2, 1))
    D = np.sum((Z[:, None] - Z[None]) ** 2, -1)
    iu = np.triu_indices(len(Z), 1)
    lu = np.log(np.mean(np.exp(-2 * D[iu])))
    return la, lu


@figure("alignment-uniformity", size=(8.6, 3.0))
def _(fig, t):
    rng = np.random.default_rng(3)
    axs = fig.subplots(1, 3, gridspec_kw=dict(wspace=0.15))
    n = 14

    def unit(a):
        return np.c_[np.cos(a), np.sin(a)]
    cases = []
    base = rng.uniform(-0.25, 0.25, n) + 1.2
    cases.append(("Sụp đổ: align tốt,\nuniform tệ", unit(base), unit(base + rng.normal(0, .04, n))))
    base = np.linspace(0, 2 * np.pi, n, endpoint=False) + rng.normal(0, .1, n)
    cases.append(("Trải đều nhưng cặp dương\nxa nhau: uniform tốt, align tệ", unit(base), unit(base + rng.uniform(1.3, 2.2, n))))
    cases.append(("Mục tiêu: cả hai tốt", unit(base), unit(base + rng.normal(0, .08, n))))
    for ax, (title, X, Y) in zip(axs, cases):
        ax.set_aspect("equal"); ax.axis("off")
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(th), np.sin(th), color=t["line"], lw=1)
        for (a, b) in zip(X, Y):
            ax.plot([a[0], b[0]], [a[1], b[1]], color=t["muted"], lw=0.9)
        ax.scatter(X[:, 0], X[:, 1], s=26, color=t["c"][BLUE], zorder=3)
        ax.scatter(Y[:, 0], Y[:, 1], s=26, color=t["c"][ORANGE], zorder=3, marker="s")
        la, lu = _align_uniform(X, Y, np.r_[X, Y])
        ax.set_title(title, fontsize=9.5, loc="center")
        ax.text(0, -1.38, f"L_align = {la:.2f}   L_uniform = {lu:.2f}", ha="center", fontsize=8.6, color=t["fg2"])
        ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.5, 1.3)
    fig.text(0.5, 0.0, "● query  ■ tài liệu dương · đường nối = cặp dương · cả hai loss càng nhỏ càng tốt",
             ha="center", fontsize=8.5, color=t["fg2"])


# ---------------------------------------------------------------- 4. pooling
@figure("pooling", size=(8.8, 4.2))
def _(fig, t):
    from matplotlib.patches import Rectangle
    ax = canvas(fig, (0, 124), (0, 60))
    toks = ["[CLS]", "không", "đăng", "nhập", "được", "[PAD]", "[PAD]"]
    mask = [1, 1, 1, 1, 1, 0, 0]
    x0, w = 22, 10.2
    cx = lambda i: x0 + i * w + w / 2
    ax.text(62, 57.5, "Đầu ra Transformer H — mỗi cột là vector của một token", fontsize=9.5, color=t["fg"],
            weight="bold", ha="center")
    for i, (s, m) in enumerate(zip(toks, mask)):
        x = x0 + i * w
        c = t["c"][BLUE] if m else t["muted"]
        for r in range(5):
            ax.add_patch(Rectangle((x + 1.6, 31 + r * 3.2), w - 3.2, 2.6, color=c, alpha=0.9 if m else 0.3, lw=0))
        ax.text(cx(i), 50.5, s, ha="center", fontsize=8.6, color=t["fg"] if m else t["muted"])
        ax.text(cx(i), 27.5, f"m={m}", ha="center", fontsize=8.4, color=t["fg2"])
    # đánh dấu
    ax.add_patch(Rectangle((x0 + 0.8, 30.2), w - 1.6, 17.4, fill=False, ec=t["c"][ORANGE], lw=2))
    ax.add_patch(Rectangle((x0 + 4 * w + 0.8, 30.2), w - 1.6, 17.4, fill=False, ec=t["c"][VIOLET], lw=2, ls="--"))
    ax.plot([x0 + 1, x0 + 5 * w - 1], [25, 25], color=t["c"][AQUA], lw=2.4)
    ax.text(cx(5.5), 53.5, "padding: bị loại", ha="center", fontsize=8.3, color=t["fg2"])
    rows = [(2, "CLS pooling", "e = h_[CLS]", "vd: BGE-M3 (dense)", ORANGE, cx(0) - 2, 29.5),
            (43, "Mean pooling", "e = Σ mₜhₜ / Σ mₜ", "SBERT, E5 — nhớ nhân mask", AQUA, cx(2), 25),
            (84, "Last-token", "e = h_t*, t* = token thật cuối", "decoder-only: Qwen3-Emb…", VIOLET, cx(4) + 2, 29.5)]
    for x, name, f, ex, c, sx, sy in rows:
        box(ax, x, 2, 38, 12, "", t, color=t["c"][c], fill=tint(t["c"][c], t, .14))
        ax.text(x + 2, 10.6, name, fontsize=9.5, weight="bold", color=t["fg"])
        ax.text(x + 2, 6.7, f, fontsize=9, color=t["fg"])
        ax.text(x + 2, 3.3, ex, fontsize=8.2, color=t["fg2"])
        arrow(ax, sx, sy, x + 19, 14.2, t, color=t["c"][c])
    ax.text(62, 18.2, "", fontsize=1)


# ---------------------------------------------------------------- 5. độ đo


# ---------------------------------------------------------------- 5. độ đo
@figure("similarity-metrics", size=(8.8, 3.3))
def _(fig, t):
    a1, a2 = fig.subplots(1, 2, gridspec_kw=dict(wspace=0.32))
    c = np.linspace(-1, 1, 100)
    a1.plot(c, 2 - 2 * c, color=t["c"][BLUE])
    for cv, lab, ty in [(0.96, "q, d₁", 0.12), (0.48, "q, d₂", 0.62)]:
        a1.scatter([cv], [2 - 2 * cv], color=t["c"][ORANGE], zorder=3, s=36)
        a1.annotate(f"{lab}: cos = {cv} → ‖q−d‖² = {2 - 2 * cv:.2f}", xy=(cv, 2 - 2 * cv), xytext=(-0.98, ty),
                    fontsize=8.4, color=t["fg"], arrowprops=dict(arrowstyle="-", color=t["muted"], lw=0.8))
    a1.set_xlabel("cos(a, b)"); a1.set_ylabel("‖a − b‖²")
    a1.set_title("Vector chuẩn hóa: ‖a−b‖² = 2 − 2cos")
    note(a1, 0.98, 3.6, "giảm đơn điệu\n→ cùng thứ hạng kNN", t, ha="right")
    ygrid(a1, t)

    a2.set_aspect("equal")
    for v, col in [((3, 4), BLUE), ((6, 8), AQUA), ((1, 1), ORANGE)]:
        a2.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=t["c"][col], lw=2))
    a2.plot([0, 3], [0, 4], color=t["c"][BLUE], lw=3.5, zorder=4)
    a2.text(1.9, 3.1, "a=(3,4)", fontsize=8.6, color=t["fg"], ha="right")
    a2.text(6.2, 8.1, "c=(30,40) [vẽ thu nhỏ 1/5]", fontsize=8.6, color=t["fg"])
    a2.text(1.2, 0.6, "b=(1,1)", fontsize=8.6, color=t["fg"])
    a2.set_xlim(0, 13); a2.set_ylim(0, 9.5)
    a2.set_title("Chưa chuẩn hóa: cùng góc, khác dot")
    a2.text(5.4, 2.2, "cos(a,b) = cos(c,b) ≈ 0.990\naᵀb = 7   ·   cᵀb = 70", fontsize=9, color=t["fg"])
    a2.set_xticks([]); a2.set_yticks([])


# ---------------------------------------------------------------- 7.2 Matryoshka


# ---------------------------------------------------------------- 6.1 anisotropy
def _pair_cos(E, n=20000, rng=None):
    i, j = rng.integers(0, len(E), (2, n)); k = i != j
    return np.sum(E[i[k]] * E[j[k]], 1)


@figure("anisotropy", size=(8.8, 3.2))
def _(fig, t):
    rng = np.random.default_rng(0)
    a1, a2 = fig.subplots(1, 2, gridspec_kw=dict(width_ratios=[1, 1.4], wspace=0.28))
    X2 = 2.0 + 0.3 * rng.normal(size=(250, 2)) * np.array([1.0, 1.0])
    X2 = X2 + rng.normal(0, 0.25, (250, 2))
    Z = X2 / np.linalg.norm(X2, axis=1, keepdims=True)
    C = X2 - X2.mean(0); Zc = C / np.linalg.norm(C, axis=1, keepdims=True)
    a1.set_aspect("equal")
    th = np.linspace(0, 2 * np.pi, 200)
    a1.plot(np.cos(th), np.sin(th), color=t["line"], lw=1)
    a1.scatter(Zc[:, 0], Zc[:, 1], s=8, color=t["c"][AQUA], alpha=.7, label="sau centering")
    a1.scatter(Z[:, 0], Z[:, 1], s=8, color=t["c"][ORANGE], alpha=.8, label="ban đầu (hình nón hẹp)")
    a1.set_xlim(-1.25, 1.25); a1.set_ylim(-1.25, 1.55); a1.set_xticks([]); a1.set_yticks([])
    for s in a1.spines.values():
        s.set_visible(False)
    a1.legend(loc="upper center", fontsize=8.2, ncol=1)
    a1.set_title("2 chiều: vector đã chuẩn hóa", fontsize=9.8)

    X = 2.0 + 0.3 * rng.normal(size=(2000, 64))
    E = X / np.linalg.norm(X, axis=1, keepdims=True)
    Xc = X - X.mean(0); Ec = Xc / np.linalg.norm(Xc, axis=1, keepdims=True)
    c0, c1 = _pair_cos(E, rng=rng), _pair_cos(Ec, rng=rng)
    bins = np.linspace(-0.6, 1.0, 81)
    a2.hist(c1, bins=bins, color=t["c"][AQUA], label=f"sau centering: TB ≈ {abs(c1.mean()):.3f}")
    a2.hist(c0, bins=bins, color=t["c"][ORANGE], label=f"ban đầu: TB = {c0.mean():.3f}")
    a2.set_yscale("log")
    a2.set_xlabel("cos giữa hai vector ngẫu nhiên khác nhau"); a2.set_ylabel("số cặp (log)")
    a2.set_title("Mô phỏng 2.000 vector 64 chiều, mỗi tọa độ 2.0 + 0.3·N(0,1)", fontsize=9.8)
    a2.legend(loc="upper left", fontsize=8.3)
    ygrid(a2, t)


# ---------------------------------------------------------------- 6.2 hubness
def _nk(X, k=10):
    E = X / np.linalg.norm(X, axis=1, keepdims=True)
    S = E @ E.T
    np.fill_diagonal(S, -np.inf)
    top = np.argpartition(-S, k, axis=1)[:, :k]
    return np.bincount(top.ravel(), minlength=len(X))


@figure("hubness", size=(8.8, 3.2))
def _(fig, t):
    rng = np.random.default_rng(1)
    a1, a2 = fig.subplots(1, 2, gridspec_kw=dict(wspace=0.3, width_ratios=[1.35, 1]))
    n = 2000
    X3 = rng.uniform(size=(n, 3)); X100 = rng.uniform(size=(n, 100))
    cfg = [("3 chiều", X3, BLUE),
           ("100 chiều", X100, ORANGE),
           ("100 chiều, sau centering", X100 - X100.mean(0), AQUA)]
    bins = np.arange(0, 210, 5)
    for name, X, c in cfg:
        N = _nk(X)
        a1.hist(N, bins=bins, histtype="step", lw=2, color=t["c"][c],
                label=f"{name}: skew {skew(N):.1f}, max {N.max()}")
    a1.set_yscale("log")
    a1.axvline(10, color=t["muted"], ls="--", lw=1)
    a1.set_ylim(0.7, 3e4)
    a1.text(13, 1.2, "E[N₁₀] = 10", fontsize=8.4, color=t["fg2"])
    a1.set_xlabel("N₁₀(x): số lần x lọt top-10 (cosine) của điểm khác")
    a1.set_ylabel("số điểm (log)")
    a1.set_title("Mô phỏng 2.000 điểm đều trong [0,1]ᵈ", fontsize=9.6)
    a1.legend(fontsize=8.0, loc="upper right")
    ygrid(a1, t)

    names = ["d_A: «liên hệ CS»\n(hub)", "d_B: «lỗi SSO\nAzure AD»"]
    cosv = np.array([0.71, 0.69]); r = np.array([0.68, 0.40])
    csls = 2 * cosv - r
    x = np.arange(2); w = 0.36
    a2.bar(x - w / 2, cosv, w * 0.92, color=t["c"][BLUE], label="cosine")
    a2.bar(x + w / 2, csls, w * 0.92, color=t["c"][AQUA], label="CSLS (bỏ r_D(q))")
    for i in range(2):
        a2.text(i - w / 2, cosv[i] + 0.02, f"{cosv[i]:.2f}", ha="center", fontsize=8.6, color=t["fg"])
        a2.text(i + w / 2, csls[i] + 0.02, f"{csls[i]:.2f}", ha="center", fontsize=8.6, color=t["fg"])
    a2.set_xticks(x, names, fontsize=8.6); a2.set_ylim(0, 1.15)
    a2.set_title("CSLS phạt hub\n(r_Q(d_A)=0.68, r_Q(d_B)=0.40)", fontsize=9.6)
    a2.legend(fontsize=8.2, loc="upper left")
    ygrid(a2, t)


# ---------------------------------------------------------------- 7.1 bộ nhớ
@figure("index-memory", size=(8.2, 3.3))
def _(fig, t):
    ax = fig.subplots()
    dims = [4096, 1024, 512, 256]
    prec = [("float32", 4, BLUE), ("int8", 1, AQUA), ("binary", 1 / 8, ORANGE)]
    y = np.arange(len(dims)); h = 0.26
    for k, (name, byt, c) in enumerate(prec):
        mib = np.array([300_000 * d * byt / 2 ** 20 for d in dims])
        ax.barh(y + (k - 1) * h, mib, height=h * 0.9, color=t["c"][c], label=name)
        for yi, v in zip(y, mib):
            s = f"{v / 1024:.2f} GiB" if v >= 1024 else f"{v:.0f} MiB" if v >= 10 else f"{v:.1f} MiB"
            ax.text(v * 1.12, yi + (k - 1) * h, s, va="center", fontsize=8, color=t["fg2"])
    ax.set_xscale("log"); ax.set_xlim(5, 30000)
    ax.set_yticks(y, [f"d = {d}" for d in dims]); ax.invert_yaxis()
    ax.set_xlabel("bộ nhớ vector thô cho ~300.000 chunk (log, chưa tính overhead HNSW)")
    ax.set_title("Bộ nhớ index theo số chiều và độ chính xác số")
    ax.legend(loc="lower right", fontsize=8.5)
    xgrid(ax, t)


# ---------------------------------------------------------------- 7.2 Matryoshka
@figure("matryoshka", size=(8.8, 3.1))
def _(fig, t):
    from matplotlib.patches import Rectangle
    a1 = fig.add_axes([0.0, 0.05, 0.58, 0.85]); a2 = fig.add_axes([0.68, 0.16, 0.30, 0.66])
    a1.set_xlim(0, 1500); a1.set_ylim(0, 7); a1.axis("off")
    ms = [64, 128, 256, 512, 1024]
    cols = [t["c"][i] for i in (BLUE, AQUA, YELLOW, ORANGE, VIOLET)]
    for k, (m, c) in enumerate(zip(ms, cols)):
        y = 5.6 - k * 1.1
        a1.add_patch(Rectangle((20, y), m, 0.75, color=c, lw=0))
        a1.text(20 + m + 15, y + 0.37, f"z₁:{m} → L", va="center", fontsize=8.6, color=t["fg"])
    a1.text(20, 6.6, "Tiền tố lồng nhau, lát nào cũng phải truy hồi tốt:  L_MRL = Σₘ cₘ · L(z₁:ₘ)",
            fontsize=9.2, color=t["fg"], weight="bold")
    a1.text(20, 0.0, "→ thông tin quan trọng dồn về các chiều đầu; cắt 1024 → 256 vẫn dùng được",
            fontsize=8.6, color=t["fg2"])

    u = np.array([0.7, 0.5, 0.4, 0.3]); v = np.array([0.6, 0.6, -0.3, 0.4])
    c4 = u @ v / np.linalg.norm(u) / np.linalg.norm(v)
    c2 = u[:2] @ v[:2] / np.linalg.norm(u[:2]) / np.linalg.norm(v[:2])
    b = a2.bar(["đủ 4 chiều", "cắt còn 2"], [c4, c2], color=[t["c"][BLUE], t["c"][RED]], width=0.55)
    for bb, val in zip(b, [c4, c2]):
        a2.text(bb.get_x() + bb.get_width() / 2, val + 0.03, f"{val:.3f}", ha="center", fontsize=9, color=t["fg"])
    a2.set_ylim(0, 1.15); a2.set_ylabel("cos(u, v)")
    a2.set_title("Model KHÔNG có MRL:\ncắt chiều làm sai lệch", fontsize=9.3)
    ygrid(a2, t)


# ---------------------------------------------------------------- 8.2 thiên lệch ngôn ngữ


# ---------------------------------------------------------------- 7.3 int8
@figure("int8-quantization", size=(6.6, 3.3))
def _(fig, t):
    ax = fig.subplots()
    lo, hi = -0.30, 0.30; d = (hi - lo) / 255
    x = np.linspace(-0.42, 0.42, 2000)
    q = np.clip(np.round((x - lo) / d) - 128, -128, 127)
    ax.plot(x, q, color=t["c"][BLUE])
    ax.axvspan(-0.42, lo, color=t["c"][RED], alpha=0.12, lw=0)
    ax.axvspan(hi, 0.42, color=t["c"][RED], alpha=0.12, lw=0)
    ax.text(0.36, -40, "ngoài dải\nhiệu chuẩn\n→ bị clip", ha="center", fontsize=8.4, color=t["fg2"])
    for xv in [0.13, -0.05, 0.34, -0.21]:
        qv = int(np.clip(round((xv - lo) / d) - 128, -128, 127))
        ax.scatter([xv], [qv], color=t["c"][ORANGE], zorder=4, s=36)
        ax.text(xv + 0.012, qv - 16 if xv != 0.34 else qv + 8, f"{xv:+.2f} → {qv}", fontsize=8.5, color=t["fg"])
    ax.set_xlabel("giá trị float32 xᵢ (dải hiệu chuẩn [−0.30, 0.30])")
    ax.set_ylabel("mã int8 x̂ᵢ")
    ax.set_yticks([-128, -64, 0, 64, 127])
    ax.set_title("Scalar quantization: 256 mức, Δ = 0.6/255 ≈ 0.00235")
    ygrid(ax, t)


# ---------------------------------------------------------------- 7.4 binary / Hamming
@figure("binary-hamming", size=(8.8, 3.4))
def _(fig, t):
    a1, a2 = fig.subplots(1, 2, gridspec_kw=dict(wspace=0.3, width_ratios=[1, 1.25]))
    a1.set_aspect("equal"); a1.axis("off")
    from matplotlib.patches import Wedge
    ax_ang, ay_ang = 20, 70
    R = 1.0
    # Vùng góc của pháp tuyến r khiến siêu phẳng tách x, y: [x+90, y+90] và đối đỉnh
    for s in (ax_ang + 90, ax_ang + 270):
        a1.add_patch(Wedge((0, 0), R * 0.95, s, s + (ay_ang - ax_ang), color=t["c"][ORANGE], alpha=0.25, lw=0))
    th = np.linspace(0, 2 * np.pi, 200)
    a1.plot(np.cos(th), np.sin(th), color=t["line"], lw=1)
    for a, lab in [(ax_ang, "x"), (ay_ang, "y")]:
        v = np.array([np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))])
        a1.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=t["c"][BLUE], lw=2.2))
        a1.text(*(v * 1.12), lab, fontsize=11, color=t["fg"], ha="center", va="center", weight="bold")
    a1.text(0.32, 0.22, "θ", fontsize=11, color=t["fg"])
    a1.text(0, -1.3, "Vùng cam: hướng pháp tuyến r làm siêu phẳng\ntách x và y — tổng cung 2θ / 2π ⇒ P = θ/π",
            ha="center", fontsize=8.5, color=t["fg2"])
    a1.set_xlim(-1.25, 1.25); a1.set_ylim(-1.55, 1.25)
    a1.set_title("Siêu phẳng ngẫu nhiên (SimHash)", fontsize=9.8)

    rng = np.random.default_rng(7)
    for d, c, mk in [(8, ORANGE, "o"), (1024, BLUE, "s")]:
        true, est = [], []
        for _ in range(350):
            x = rng.normal(size=48); y = x + rng.normal(size=48) * rng.uniform(0.1, 2.0)
            R_ = rng.normal(size=(d, 48))
            H = np.sum(np.sign(R_ @ x) != np.sign(R_ @ y))
            true.append(x @ y / np.linalg.norm(x) / np.linalg.norm(y)); est.append(np.cos(np.pi * H / d))
        a2.scatter(true, est, s=10, color=t["c"][c], alpha=.6, marker=mk, label=f"d = {d} bit", lw=0)
    a2.plot([0, 1], [0, 1], color=t["muted"], lw=1, ls="--")
    a2.set_xlabel("cosine thật"); a2.set_ylabel("ước lượng cos(π·H/d)")
    a2.set_xlim(0, 1); a2.set_ylim(-0.75, 1.02)
    a2.set_title("Ước lượng từ khoảng cách Hamming (mô phỏng)", fontsize=9.8)
    a2.legend(fontsize=8.4, loc="lower right")
    ygrid(a2, t)


# ---------------------------------------------------------------- 7.6 trade-off nén
@figure("compression-tradeoff", size=(7.2, 2.8))
def _(fig, t):
    ax = fig.subplots()
    base = 300_000 * 1024 * 4 / 2 ** 20
    opts = [("float32, 1024 chiều", 1, "mốc", BLUE), ("MRL 1024→256", 4, "rủi ro thấp nếu model có MRL", AQUA),
            ("int8", 4, "rủi ro thấp, cần tập hiệu chuẩn đại diện", AQUA),
            ("binary", 32, "cần rescoring; tránh khi d nhỏ", ORANGE),
            ("MRL-256 + binary", 128, "chỉ đáng khi kho rất lớn", RED)]
    y = np.arange(len(opts))
    vals = [base / f for _, f, _, _ in opts]
    ax.barh(y, vals, color=[t["c"][c] for *_, c in opts], height=0.6)
    for yi, (name, f, cm, _), v in zip(y, opts, vals):
        ax.text(v * 1.15, yi, f"{v:,.0f} MiB · {f}× · {cm}" if v > 20 else f"{v:.1f} MiB · {f}× · {cm}",
                va="center", fontsize=8.3, color=t["fg2"])
    ax.set_yticks(y, [o[0] for o in opts]); ax.invert_yaxis()
    ax.set_xscale("log"); ax.set_xlim(5, 1e5)
    ax.set_xlabel("bộ nhớ cho ~300K chunk (log)")
    ax.set_title("Mức nén và cái giá đi kèm")
    xgrid(ax, t)


# ---------------------------------------------------------------- 8.2 thiên lệch ngôn ngữ
@figure("language-bias", size=(6.8, 2.6))
def _(fig, t):
    ax = fig.subplots()
    names = ["d_vi: «đổi email đăng nhập»\n(sai chủ đề, cùng ngôn ngữ)", "d_en: «Export invoices as PDF»\n(đúng chủ đề, khác ngôn ngữ)"]
    sem = [0.57, 0.60]; bonus = [0.05, 0.0]
    ax.barh(names, sem, color=t["c"][BLUE], height=0.5, label="phần ngữ nghĩa sᵀs′")
    ax.barh(names, bonus, left=sem, color=t["c"][ORANGE], height=0.5, label="«thưởng» cùng ngôn ngữ ‖l‖²")
    for i in range(2):
        ax.text(sem[i] + bonus[i] + 0.004, i, f"cos = {sem[i] + bonus[i]:.2f}", va="center", fontsize=9, color=t["fg"])
    ax.invert_yaxis()
    ax.set_xlim(0.45, 0.80); ax.set_xlabel("cos với email tiếng Việt hỏi cách xuất hóa đơn (số giả định)")
    ax.set_title("Thiên lệch ngôn ngữ đẩy tài liệu sai lên trên")
    ax.legend(fontsize=8.2, loc="center right")
    xgrid(ax, t)


# ---------------------------------------------------------------- 10. SPLADE


# ---------------------------------------------------------------- 9. instruction
@figure("instruction-prefix", size=(8.8, 3.2))
def _(fig, t):
    ax = canvas(fig, (0, 120), (0, 47))
    box(ax, 2, 18, 22, 8, "«Tôi muốn hủy\ngói Pro»", t, color=t["fg2"], fs=9.5, weight="bold")
    rows = [(34, "kb_search", "→ bài HC «Cách hủy đăng ký»", BLUE),
            (19, "intent", "→ các email «muốn hủy» khác", AQUA),
            (4, "dedup", "→ câu gần y hệt", ORANGE)]
    for y, task, res, c in rows:
        box(ax, 34, y, 30, 7, f"Instruct: {task}\nQuery: …", t, color=t["c"][c], fill=tint(t["c"][c], t, .14), fs=8.8)
        arrow(ax, 24, 22, 34, y + 3.5, t, color=t["c"][c])
        box(ax, 70, y, 12, 7, "Encoder", t, fs=8.8)
        arrow(ax, 64, y + 3.5, 70, y + 3.5, t)
        arrow(ax, 82, y + 3.5, 89, y + 3.5, t, color=t["c"][c])
        ax.text(90, y + 3.5, res, va="center", fontsize=8.8, color=t["fg"])
    ax.text(2, 45, "Cùng văn bản, khác tác vụ → khác vector; tài liệu embed một lần, không cần prefix (Qwen3/Harrier)",
            fontsize=8.8, color=t["fg2"])


# ---------------------------------------------------------------- 10. SPLADE
@figure("splade-expansion", size=(8.0, 3.0))
def _(fig, t):
    from matplotlib.patches import Patch
    ax = fig.subplots()
    terms = ["đăng nhập", "không", "được", "login", "sign", "mật khẩu", "tài khoản", "hóa đơn"]
    w = [2.1, 0.9, 0.3, 1.6, 1.2, 0.9, 0.7, 0.0]
    inq = [1, 1, 1, 0, 0, 0, 0, 0]
    cols = [t["c"][BLUE] if i else t["c"][ORANGE] for i in inq]
    ax.bar(terms, w, color=cols, width=0.62)
    ax.set_ylabel("trọng số wⱼ(x)")
    ax.set_title("SPLADE: vector thưa trên từ vựng cho «không đăng nhập được» (số giả định)")
    ax.legend(handles=[Patch(color=t["c"][BLUE], label="từ có trong văn bản"),
                       Patch(color=t["c"][ORANGE], label="mở rộng (expansion)")], fontsize=8.5)
    ax.text(7, 0.08, "0 (thưa)", ha="center", fontsize=8.3, color=t["fg2"])
    ax.tick_params(axis="x", labelsize=8.6)
    ygrid(ax, t)


# ---------------------------------------------------------------- 11.2 Recall / MRR


# ---------------------------------------------------------------- 11.2 Recall / MRR
@figure("recall-mrr", size=(8.2, 2.2))
def _(fig, t):
    ax = canvas(fig, (0, 120), (0, 26))
    ranked = ["d₄", "d₃", "d₉", "d₁", "d₂", "d₇", "d₅"]
    rel = {"d₃", "d₇"}
    for i, d in enumerate(ranked):
        x = 4 + i * 11
        is_rel = d in rel
        c = t["c"][GREEN] if is_rel else t["line"]
        box(ax, x, 9, 9, 8, d, t, color=c, fill=tint(t["c"][GREEN], t, .2) if is_rel else t["panel"], fs=10,
            weight="bold" if is_rel else "normal")
        ax.text(x + 4.5, 19, f"#{i + 1}", ha="center", fontsize=8.6, color=t["fg2"])
    ax.plot([3, 3 + 5 * 11 - 0.5], [6.5, 6.5], color=t["c"][BLUE], lw=2.5)
    ax.text(3, 3.2, "top-5", fontsize=8.6, color=t["c"][BLUE])
    ax.text(84, 17, "R = {d₃, d₇}", fontsize=9.5, color=t["fg"], weight="bold")
    ax.text(84, 11.5, "Recall@5 = |{d₃}| / |R| = 1/2", fontsize=9, color=t["fg"])
    ax.text(84, 7, "MRR: liên quan đầu tiên ở #2 → 1/2", fontsize=9, color=t["fg"])
    ax.text(84, 2.5, "Recall@7 = 2/2 = 1", fontsize=9, color=t["fg2"])


@figure("word2vec-skipgram", size=(8.8, 2.7))
def _(fig, t):
    ax = canvas(fig, (0, 120), (4, 40))
    toks = ["tài khoản", "bị", "khóa", "sau", "khi", "đổi", "mật khẩu"]
    ws = [12, 6, 8, 7, 6, 7, 12]
    x = 4; pos = []
    for i, (s, w) in enumerate(zip(toks, ws)):
        if i == 2:
            c, f = t["c"][BLUE], tint(t["c"][BLUE], t, .25)
        elif i in (0, 1, 3, 4):
            c, f = t["c"][AQUA], tint(t["c"][AQUA], t, .16)
        else:
            c, f = t["line"], t["panel"]
        box(ax, x, 27, w, 6, s, t, color=c, fill=f, fs=9, weight="bold" if i == 2 else "normal")
        pos.append(x + w / 2); x += w + 1.2
    ax.plot([4, pos[4] + 3], [24.5, 24.5], color=t["c"][AQUA], lw=2)
    ax.text(4, 21.5, "cửa sổ ±2 quanh từ trung tâm «khóa»", fontsize=8.5, color=t["fg2"])
    # cặp dương / âm
    ax.text(72, 35.5, "Cặp (w, c) huấn luyện", fontsize=9.5, color=t["fg"], weight="bold")
    rows = [("(khóa, bị), (khóa, sau), …", "kéo lại: −log σ(u_wᵀv_c)", GREEN),
            ("(khóa, hóa đơn), (khóa, PDF), …", "đẩy ra: −log σ(−u_wᵀv_c′)", RED)]
    for k, (pairs, act, c) in enumerate(rows):
        y = 27 - k * 9
        box(ax, 72, y, 46, 7, "", t, color=t["c"][c], fill=tint(t["c"][c], t, .12))
        ax.text(74, y + 4.8, pairs, fontsize=8.8, color=t["fg"])
        ax.text(74, y + 1.6, act, fontsize=8.6, color=t["fg2"])
    ax.text(72, 14.5, "K từ nhiễu c′ lấy theo tần suất^(3/4)", fontsize=8.4, color=t["fg2"])
    ax.text(4, 12, "Từ đi cùng ngữ cảnh giống nhau («mật khẩu», «tài khoản») → vector gần nhau:", fontsize=8.8, color=t["fg"])
    ax.text(4, 7.5, "«login» ≈ «đăng nhập».   Nhưng mỗi từ chỉ MỘT vector: «khóa» (lock / khóa học / API key) bị trộn.",
            fontsize=8.8, color=t["fg2"])


# ---------------------------------------------------------------- 4. pooling


if __name__ == "__main__":
    run("03")
