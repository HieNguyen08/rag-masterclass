"""Hình minh họa cho Module 01 — Nền tảng LLM."""
import numpy as np
from matplotlib.patches import FancyBboxPatch

from figkit import (AQUA, BLUE, GREEN, MAGENTA, ORANGE, RED, VIOLET, YELLOW, arrow, box,
                    canvas, figure, run, tint, xgrid, ygrid)


def softmax(z):
    e = np.exp(np.asarray(z, float) - np.max(z))
    return e / e.sum()


def chips(ax, x, y, parts, t, color, h=5.2, fs=8.4, cw=1.35, pad=1.6, gap=0.8):
    """Vẽ dãy ký hiệu dạng «chip»; trả về x kết thúc."""
    for p in parts:
        w = pad + cw * len(p)
        box(ax, x, y, w, h, p, t, color=color, fill=tint(color, t, .16), fs=fs, radius=0.8)
        x += w + gap
    return x


# ---------------------------------------------------------------- 1.2 BPE
@figure("bpe-merges", size=(8.8, 3.9))
def _(fig, t):
    ax = canvas(fig, (0, 124), (0, 55))
    ax.text(2, 51, "5 phép gộp (tần suất cặp)", fontsize=9, weight="bold", color=t["fg"])
    merges = [("n + </w>", "n</w>", 15), ("o + à", "oà", 12), ("oà + n</w>", "oàn</w>", 9),
              ("t + i", "ti", 8), ("ti + ề", "tiề", 6)]
    for i, (a, b, c) in enumerate(merges):
        y = 42 - i * 8
        box(ax, 2, y, 5, 5.2, str(i + 1), t, color=t["c"][VIOLET], fill=tint(t["c"][VIOLET], t, .2), fs=8.4, weight="bold")
        ax.text(9, y + 2.6, a, va="center", fontsize=8.4, color=t["fg"])
        ax.text(22, y + 2.6, "→", va="center", fontsize=8.4, color=t["fg2"])
        ax.text(25.5, y + 2.6, f"{b}  ({c})", va="center", fontsize=8.4, color=t["fg"], weight="bold")
    ax.plot([45, 45], [2, 49], color=t["line"], lw=0.8)
    ax.text(48, 51, "Từ (tần suất)", fontsize=8.6, weight="bold", color=t["fg"])
    ax.text(68, 51, "Sau 5 phép gộp", fontsize=8.6, weight="bold", color=t["fg"])
    ax.text(121, 51, "số ký hiệu", fontsize=8.2, color=t["fg2"], ha="right")
    words = [("hoàn", 5, ["h", "oàn</w>"], 5), ("hoàng", 3, ["h", "oà", "n", "g", "</w>"], 6),
             ("toàn", 4, ["t", "oàn</w>"], 5), ("tiền", 6, ["tiề", "n</w>"], 5),
             ("tiếng", 2, ["ti", "ế", "n", "g", "</w>"], 6)]
    for i, (w, f, segs, n0) in enumerate(words):
        y = 42 - i * 8
        ax.text(48, y + 2.6, f"{w} ({f})", va="center", fontsize=8.6, color=t["fg"])
        frequent = len(segs) == 2
        col = t["c"][GREEN] if frequent else t["c"][ORANGE]
        chips(ax, 68, y, segs, t, col)
        ax.text(121, y + 2.6, f"{n0} → {len(segs)}", va="center", ha="right", fontsize=8.6, color=t["fg"],
                weight="bold" if frequent else "normal")
    ax.text(48, 1.5, "Từ phổ biến được nén mạnh; từ hiếm hơn (hoàng, tiếng) vẫn bị cắt vụn.",
            fontsize=8.2, color=t["fg2"])


# ---------------------------------------------------------------- 1.3 Unigram / Viterbi
@figure("unigram-viterbi", size=(8.4, 2.7))
def _(fig, t):
    ax = canvas(fig, (0, 118), (0, 38))
    xs = [12, 52, 92]
    for x, lab in zip(xs, ["đầu", "giữa", "cuối"]):
        ax.add_patch(FancyBboxPatch((x - 2.5, 13.5), 5, 5, boxstyle="circle,pad=0", fc=t["panel"], ec=t["fg2"], lw=1.2))
        ax.text(x, 9.5, lab, ha="center", fontsize=8, color=t["fg2"])
    bad = t["c"][ORANGE]; good = t["c"][GREEN]
    for (x1, x2), lab in zip([(xs[0], xs[1]), (xs[1], xs[2])], ["đăng\nln 0,02 = −3,91", "nhập\nln 0,02 = −3,91"]):
        arrow(ax, x1 + 3, 16, x2 - 3, 16, t, color=bad, lw=1.6)
        ax.text((x1 + x2) / 2, 5.5, lab, ha="center", va="top", fontsize=8.2, color=t["fg"], linespacing=1.3)
    arrow(ax, xs[0] + 2, 19, xs[2] - 2, 19, t, color=good, lw=2.4, rad=-0.32)
    ax.text(52, 33.5, "▁đăngnhập   ln 0,001 = −6,91", ha="center", fontsize=8.6, color=t["fg"], weight="bold")
    ax.text(100, 24, "[đăng, nhập]: −7,82\n[▁đăngnhập]: −6,91  ✓", fontsize=8.6, color=t["fg"], va="center", linespacing=1.5)
    ax.text(100, 13, "Viterbi chọn đường\ncó log-prob cao nhất", fontsize=8, color=t["fg2"], va="center")


# ---------------------------------------------------------------- 1.4 tokenization premium
@figure("token-premium", size=(8.8, 3.0))
def _(fig, t):
    a1, a2 = fig.subplots(1, 2, gridspec_kw=dict(wspace=0.75, width_ratios=[1.1, 1]))
    names = ["cl100k_base\n(GPT-3.5/4)", "DeepSeek-V3.2", "o200k_base\n(GPT-4o+)", "Qwen2.5/Qwen3"]
    vals = [2.14, 1.88, 1.34, 1.26]
    cols = [t["c"][RED], t["c"][ORANGE], t["c"][BLUE], t["c"][AQUA]]
    y = np.arange(4)[::-1]
    a1.barh(y, vals, color=cols, height=0.6)
    a1.axvline(1, color=t["fg2"], ls="--", lw=1)
    a1.text(1.03, 3.55, "= tiếng Anh", fontsize=7.8, color=t["fg2"])
    for yi, v in zip(y, vals):
        a1.text(v + 0.04, yi, f"{v:.2f}×", va="center", fontsize=8.4, color=t["fg"])
    a1.set_yticks(y, names, fontsize=8)
    a1.set_xlim(0, 2.6)
    a1.set_ylim(-0.6, 3.9)
    a1.set_xlabel("số token tiếng Việt / tiếng Anh")
    a1.set_title("Tokenization premium của tiếng Việt", fontsize=9.2)
    xgrid(a1, t)
    labs = ["tiếng Anh\no200k", "tiếng Việt\no200k", "tiếng Việt\ncl100k"]
    tok = [330, 440, 700]
    a2.bar(range(3), tok, color=[t["muted"], t["c"][BLUE], t["c"][RED]], width=0.6)
    for i, v in enumerate(tok):
        a2.text(i, v + 15, f"~{v}", ha="center", fontsize=8.4, color=t["fg"])
    a2.set_xticks(range(3), labs, fontsize=8)
    a2.set_ylim(0, 820)
    a2.set_ylabel("token")
    a2.set_title("Một email ~250 từ (giả định)", fontsize=9.2)
    ygrid(a2, t)


# ---------------------------------------------------------------- 2 embedding + LM head
@figure("embedding-lmhead", size=(8.8, 3.1))
def _(fig, t):
    ax = canvas(fig, (0, 124), (0, 47))
    # ma trận E
    ax.text(2, 39.5, "token ID t", fontsize=8.4, color=t["fg"], weight="bold")
    ex, ey, ew, eh = 2, 6, 16, 28
    ax.add_patch(FancyBboxPatch((ex, ey), ew, eh, boxstyle="square,pad=0", fc=t["panel"], ec=t["fg2"], lw=1))
    for k in range(1, 9):
        ax.plot([ex, ex + ew], [ey + k * eh / 9] * 2, color=t["grid"], lw=0.6)
    ax.add_patch(FancyBboxPatch((ex, ey + 5 * eh / 9), ew, eh / 9, boxstyle="square,pad=0",
                                fc=tint(t["c"][BLUE], t, .45), ec=t["c"][BLUE], lw=1.4))
    ax.text(ex + ew / 2, 2.5, "E  (V × d)", ha="center", fontsize=8.4, color=t["fg"])
    ax.text(ex + ew + 1, ey + 5.5 * eh / 9, "← hàng t", va="center", fontsize=7.8, color=t["c"][BLUE] if t["name"] == "light" else t["fg"])
    arrow(ax, 10, 38.5, 10, 35, t, lw=1.2)
    # x
    box(ax, 32, 20, 14, 6, "x ∈ ℝᵈ", t, color=t["c"][BLUE], fill=tint(t["c"][BLUE], t, .16), fs=8.6)
    arrow(ax, 26.5, 23, 31.7, 23, t)
    box(ax, 50, 15, 22, 16, "N khối decoder\n(attention + FFN)", t, color=t["c"][VIOLET], fill=tint(t["c"][VIOLET], t, .14), fs=8.4)
    arrow(ax, 46.3, 23, 49.7, 23, t)
    box(ax, 76, 20, 12, 6, "hₙ ∈ ℝᵈ", t, color=t["c"][BLUE], fill=tint(t["c"][BLUE], t, .16), fs=8.6)
    arrow(ax, 72.3, 23, 75.7, 23, t)
    box(ax, 92, 18, 13, 10, "z = W_out hₙ\nlogits ∈ ℝⱽ", t, color=t["c"][ORANGE], fill=tint(t["c"][ORANGE], t, .16), fs=8)
    arrow(ax, 88.3, 23, 91.7, 23, t)
    box(ax, 109, 18, 13, 10, "softmax\np(t | ngữ cảnh)", t, color=t["c"][GREEN], fill=tint(t["c"][GREEN], t, .16), fs=8)
    arrow(ax, 105.3, 23, 108.7, 23, t)
    # weight tying
    arrow(ax, 18, 36, 98, 28.5, t, color=t["c"][MAGENTA], ls="--", lw=1.3, rad=-0.12)
    ax.text(62, 44, "weight tying: W_out = E  → zₜ = E[t,:] · hₙ", ha="center", fontsize=8.4,
            color=t["c"][MAGENTA] if t["name"] == "light" else t["fg"], weight="bold")
    ax.text(50, 7, "Qwen: V = 151.936, d = 1024 → tie tiết kiệm V·d ≈ 155 triệu tham số", fontsize=8, color=t["fg2"])
    ax.text(50, 2.5, "logit = tích vô hướng giữa trạng thái ẩn và embedding của token", fontsize=8, color=t["fg2"])


# ---------------------------------------------------------------- 3.1 attention ví dụ tay
@figure("attention-example", size=(8.6, 3.2))
def _(fig, t):
    a1, a2 = fig.subplots(1, 2, gridspec_kw=dict(wspace=0.35, width_ratios=[1, 1.05]))
    W = np.array([[1, 0, 0], [0.330, 0.670, 0], [0.248, 0.248, 0.503]])
    mask = np.triu(np.ones((3, 3)), 1).astype(bool)
    cmap_c = t["c"][BLUE]
    for i in range(3):
        for j in range(3):
            if mask[i, j]:
                a1.add_patch(FancyBboxPatch((j - .45, i - .45), .9, .9, boxstyle="square,pad=0", fc=t["panel2"], ec=t["line"], hatch="///", lw=0.6))
                a1.text(j, i, "−∞", ha="center", va="center", fontsize=8.4, color=t["fg2"])
            else:
                v = W[i, j]
                a1.add_patch(FancyBboxPatch((j - .45, i - .45), .9, .9, boxstyle="square,pad=0",
                                            fc=tint(cmap_c, t, 0.12 + 0.8 * v), ec=t["line"], lw=0.6))
                a1.text(j, i, f"{v:.3f}".rstrip("0").rstrip(".") if v in (0, 1) else f"{v:.3f}", ha="center", va="center",
                        fontsize=9, color=t["fg"], weight="bold")
    a1.set_xlim(-.6, 2.6); a1.set_ylim(2.6, -.6)
    a1.set_xticks(range(3), ["key 1", "key 2", "key 3"])
    a1.set_yticks(range(3), ["query 1", "query 2", "query 3"])
    a1.set_title("Trọng số α (sau causal mask + softmax)", fontsize=9)
    for s in a1.spines.values():
        s.set_visible(False)
    a1.tick_params(length=0)
    # outputs
    V = np.array([[1, 0], [0, 2], [1, 1]])
    O = W @ V
    a2.fill(V[:, 0], V[:, 1], color=tint(t["c"][BLUE], t, .12), ec=t["line"], lw=0.8)
    for k, (x, y) in enumerate(V):
        a2.plot(x, y, "s", color=t["muted"], ms=7)
        a2.text(x + 0.06, y + 0.07, f"v{k + 1} = ({x}, {y})", fontsize=8, color=t["fg2"])
    cols = [t["c"][ORANGE], t["c"][VIOLET], t["c"][GREEN]]
    offs = [(0.05, -0.2), (0.07, 0.0), (-0.62, -0.25)]
    for k, ((x, y), c) in enumerate(zip(O, cols)):
        a2.plot(x, y, "o", color=c, ms=8)
        a2.text(x + offs[k][0], y + offs[k][1], f"o{k + 1} = ({x:.2f}, {y:.2f})", fontsize=8.2, color=t["fg"], weight="bold")
    a2.set_xlim(-0.25, 1.75); a2.set_ylim(-0.35, 2.35)
    a2.set_aspect("equal")
    a2.set_title("Đầu ra oᵢ = tổ hợp lồi của các value", fontsize=9)
    ygrid(a2, t); xgrid(a2, t)


# ---------------------------------------------------------------- 3.2 sqrt(d_k)
@figure("sqrt-dk", size=(8.8, 3.0))
def _(fig, t):
    a1, a2 = fig.subplots(1, 2, gridspec_kw=dict(wspace=0.3, width_ratios=[1.25, 1]))
    rng = np.random.default_rng(0)
    q = rng.standard_normal((100000, 128)); k = rng.standard_normal((100000, 128))
    s = (q * k).sum(1)
    bins = np.linspace(-40, 40, 121)
    a1.hist(s, bins=bins, color=t["c"][RED], alpha=0.75, density=True, label=f"q·k, d_k = 128 (độ lệch chuẩn ≈ {s.std():.1f})")
    a1.hist(s / np.sqrt(128), bins=bins, color=t["c"][BLUE], alpha=0.75, density=True, label="q·k / √d_k (độ lệch chuẩn ≈ 1)")
    a1.set_xlim(-40, 40)
    a1.set_ylim(0, 0.5)
    a1.set_xlabel("điểm attention thô")
    a1.set_ylabel("mật độ")
    a1.legend(fontsize=7.8, loc="upper left")
    a1.set_title("Mô phỏng 100.000 cặp vector Gauss", fontsize=9)
    ygrid(a1, t)
    x = np.arange(3); w = 0.36
    p8 = softmax([8, 0, 0]); p1 = softmax([1, 0, 0])
    a2.bar(x - w / 2, p8, w * .92, color=t["c"][RED], label="softmax(8, 0, 0)")
    a2.bar(x + w / 2, p1, w * .92, color=t["c"][BLUE], label="softmax(1, 0, 0)")
    for xi, a, b in zip(x, p8, p1):
        a2.text(xi - w / 2, a + 0.02, f"{a:.4f}" if a > .5 else f"{a:.4f}", ha="center", fontsize=7.2, color=t["fg"], rotation=0 if a > .5 else 90, va="bottom")
        a2.text(xi + w / 2, b + 0.02, f"{b:.3f}", ha="center", fontsize=7.6, color=t["fg"])
    a2.set_xticks(x, ["token 1", "token 2", "token 3"])
    a2.set_ylim(0, 1.25)
    a2.set_yticks([0, .25, .5, .75, 1])
    a2.legend(fontsize=7.8, loc="upper right")
    a2.set_title("Điểm chênh lớn → softmax gần one-hot", fontsize=9)
    ygrid(a2, t)


# ---------------------------------------------------------------- 3.4 MHA / GQA / MQA
@figure("gqa", size=(8.8, 2.9))
def _(fig, t):
    ax = canvas(fig, (0, 124), (0, 40))
    cfg = [("MHA", 8, "8 đầu query · 8 bộ K,V"), ("GQA", 2, "8 đầu query · 2 bộ K,V"), ("MQA", 1, "8 đầu query · 1 bộ K,V")]
    qc = t["c"][BLUE]; kc = t["c"][ORANGE]
    for p, (name, g, sub) in enumerate(cfg):
        x0 = 2 + p * 42
        ax.text(x0 + 18, 36, name, ha="center", fontsize=9.6, weight="bold", color=t["fg"])
        ax.text(x0 + 18, 4, sub, ha="center", fontsize=8, color=t["fg2"])
        qx = [x0 + 1 + i * 4.4 for i in range(8)]
        for x in qx:
            box(ax, x, 25, 3.4, 5, "", t, color=qc, fill=tint(qc, t, .3), radius=0.5)
        span = 8 // g
        for j in range(g):
            members = qx[j * span:(j + 1) * span]
            cx = (members[0] + members[-1]) / 2 + 1.7
            kw = max(3.4, min(4.4 * span - 1, 10))
            box(ax, cx - kw / 2, 10, kw, 5, "K,V" if kw > 6 else "", t, color=kc, fill=tint(kc, t, .3), fs=7.6, radius=0.5)
            for x in members:
                ax.plot([x + 1.7, cx], [24.8, 15.2], color=t["line"], lw=0.8)
    ax.text(2, 31.5, "Q", fontsize=8, color=qc if t["name"] == "light" else t["fg"], weight="bold")
    ax.text(124, 31.5, "KV cache tỉ lệ với số bộ K,V", ha="right", fontsize=8, color=t["fg2"])


# ---------------------------------------------------------------- 3.6 LayerNorm vs RMSNorm
@figure("rmsnorm", size=(7.6, 2.9))
def _(fig, t):
    ax = fig.subplots()
    x = np.array([2, -1, 3, 0.])
    rms = x / np.sqrt((x ** 2).mean())
    ln = (x - x.mean()) / x.std()
    i = np.arange(4); w = 0.26
    for k, (v, c, lab) in enumerate([(x, t["muted"], "x = (2, −1, 3, 0)"),
                                     (rms, t["c"][BLUE], "RMSNorm: x / RMS(x), RMS = √3,5 ≈ 1,871"),
                                     (ln, t["c"][ORANGE], "LayerNorm: (x − 1) / 1,581")]):
        ax.bar(i + (k - 1) * w, v, w * .92, color=c, label=lab)
        if k:
            for xi, vi in zip(i, v):
                ax.text(xi + (k - 1) * w, vi + (0.08 if vi >= 0 else -0.1), f"{vi:.3f}".replace("-", "−"), ha="center",
                        va="bottom" if vi >= 0 else "top", fontsize=6.8, color=t["fg"])
    ax.axhline(0, color=t["fg2"], lw=0.8)
    ax.set_xticks(i, [f"chiều {k + 1}" for k in i])
    ax.set_ylim(-2, 4.3)
    ax.legend(fontsize=7.8, loc="upper left", ncol=1)
    ax.set_title("RMSNorm giữ dấu và tỉ lệ, không dời trung bình; LayerNorm trừ trung bình trước", fontsize=9)
    ygrid(ax, t)


# ---------------------------------------------------------------- 3.7 đếm tham số
@figure("param-count", size=(8.8, 3.0))
def _(fig, t):
    a1, a2 = fig.subplots(1, 2, gridspec_kw=dict(wspace=0.45, width_ratios=[1.5, 1]))
    parts = [("Attention 32 × 67,1M", 32 * 67.1e6, t["c"][BLUE]),
             ("FFN SwiGLU 32 × 135,3M", 32 * 135.3e6, t["c"][ORANGE]),
             ("Embedding + LM head", 262.1e6, t["c"][AQUA])]
    left = 0
    for name, v, c in parts:
        a1.barh(0, v / 1e9, left=left, color=c, height=0.5, label=f"{name} = {v / 1e9:.2f} tỷ".replace(".", ","))
        left += v / 1e9
    a1.text(left + 0.08, 0, "≈ 6,74 tỷ", va="center", fontsize=8.6, color=t["fg"], weight="bold")
    a1.set_xlim(0, 8.2); a1.set_ylim(-0.6, 1.6)
    a1.set_yticks([])
    a1.set_xlabel("tỷ tham số")
    a1.legend(fontsize=7.8, loc="upper left", ncol=1)
    a1.set_title("Kiến trúc kiểu Llama-2-7B: FFN chiếm ~2/3", fontsize=9)
    xgrid(a1, t)
    labs = ["FP16/BF16", "4-bit"]
    gb = [13.5, 4.0]
    a2.bar(range(2), gb, color=[t["c"][RED], t["c"][GREEN]], width=0.55)
    a2.axhline(6, color=t["fg2"], ls="--", lw=1.1)
    a2.text(1.45, 6.3, "GPU 6 GB", ha="right", fontsize=8, color=t["fg2"])
    for k, v in enumerate(gb):
        a2.text(k, v + 0.3, f"~{v:g} GB".replace(".", ","), ha="center", fontsize=8.4, color=t["fg"])
    a2.set_xticks(range(2), labs)
    a2.set_ylim(0, 16)
    a2.set_ylabel("GB (chỉ trọng số)")
    a2.set_title("Bộ nhớ trọng số", fontsize=9)
    ygrid(a2, t)


# ---------------------------------------------------------------- 4.2 sinusoidal
@figure("sinusoidal", size=(8.6, 3.0))
def _(fig, t):
    a1, a2 = fig.subplots(1, 2, gridspec_kw=dict(wspace=0.3, width_ratios=[1.2, 1]))
    d, n = 64, 100
    m = np.arange(n)[:, None]; i = np.arange(d // 2)[None, :]
    th = 10000 ** (-2 * i / d)
    P = np.zeros((n, d)); P[:, 0::2] = np.sin(m * th); P[:, 1::2] = np.cos(m * th)
    cmap = "RdBu_r"
    im = a1.imshow(P, aspect="auto", cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")
    a1.set_xlabel("chiều (cặp đầu → cặp cuối)")
    a1.set_ylabel("vị trí m")
    a1.set_title("Vector vị trí pₘ, d = 64", fontsize=9)
    cb = fig.colorbar(im, ax=a1, fraction=0.05, pad=0.02)
    cb.outline.set_visible(False)
    mm = np.linspace(0, n, 800)
    for k, c in zip([2, 8, 16], [t["c"][RED], t["c"][ORANGE], t["c"][BLUE]]):
        a2.plot(mm, np.sin(mm * 10000 ** (-2 * k / d)), color=c, lw=1.4, label=f"cặp i = {k}")
    a2.set_xlabel("vị trí m")
    a2.set_ylim(-1.3, 1.9)
    a2.legend(fontsize=7.8, loc="upper right", ncol=3, handlelength=1.2, columnspacing=0.8)
    a2.set_title("sin(m·θᵢ) của ba cặp: nhanh → chậm", fontsize=9)
    ygrid(a2, t)


# ---------------------------------------------------------------- 4.3 RoPE
@figure("rope", size=(8.8, 3.3))
def _(fig, t):
    a1, a2 = fig.subplots(1, 2, gridspec_kw=dict(wspace=0.3, width_ratios=[1, 1.25]))
    th = 0.5
    q = np.array([1, 0.]); k = np.array([0.6, 0.8])

    def R(p):
        return np.array([[np.cos(p), -np.sin(p)], [np.sin(p), np.cos(p)]])
    circ = np.linspace(0, 2 * np.pi, 200)
    a1.plot(np.cos(circ), np.sin(circ), color=t["grid"], lw=1)
    for (m, n), c in zip([(3, 1), (10, 8)], [t["c"][BLUE], t["c"][ORANGE]]):
        qm = R(m * th) @ q; kn = R(n * th) @ k
        for v, lab, ls in [(qm, f"q̃_{m}", "-"), (kn, f"k̃_{n}", "--")]:
            a1.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=c, lw=1.6, ls=ls))
        a1.text(0.3 if m == 3 else 0.42, 0.9 if m == 3 else -0.8, f"q̃{'₃' if m == 3 else '₁₀'} (liền), k̃{'₁' if m == 3 else '₈'} (đứt)",
                ha="left" if m == 3 else "left", va="center", fontsize=7.8, color=c, weight="bold")
    a1.text(0.45, -1.5, "hai cặp quay tới chỗ khác nhau,\nnhưng góc giữa q̃ và k̃ như nhau", ha="center", fontsize=7.8, color=t["fg2"])
    a1.set_xlim(-1.3, 2.2); a1.set_ylim(-1.75, 1.4)
    a1.set_aspect("equal"); a1.axis("off")
    a1.set_title("Quay q theo m, k theo n", fontsize=9)
    dd = np.linspace(-12, 12, 400)
    a2.plot(dd, 0.6 * np.cos(dd * th) + 0.8 * np.sin(dd * th), color=t["c"][VIOLET], lw=1.6)
    a2.axvline(2, color=t["line"], ls=":", lw=1)
    for (m, n), mk in zip([(3, 1), (5, 3), (10, 8)], ["o", "s", "^"]):
        v = (R(m * th) @ q) @ (R(n * th) @ k)
        a2.plot(2, v, mk, color=t["c"][GREEN], ms=7, mfc="none", mew=1.5, label=f"(m, n) = ({m}, {n}): {v:.4f}")
    a2.set_xlabel("khoảng cách m − n")
    a2.set_ylabel("q̃ₘ · k̃ₙ")
    a2.set_ylim(-1.15, 1.6)
    a2.legend(fontsize=7.6, loc="upper left")
    a2.set_title("Điểm chỉ phụ thuộc m − n (θ = 0,5)", fontsize=9)
    ygrid(a2, t)


# ---------------------------------------------------------------- 4.4 PI / NTK / YaRN
@figure("rope-scaling", size=(8.4, 3.0))
def _(fig, t):
    ax = fig.subplots()
    d, s, b, L = 128, 4, 10000, 4096
    i = np.arange(d // 2)
    th = b ** (-2 * i / d)
    ntk = s ** (-2 * i / (d - 2))
    lam = 2 * np.pi / th
    r = L / lam
    alpha, beta = 1, 32
    gamma = np.clip((r - alpha) / (beta - alpha), 0, 1)
    yarn = gamma * 1 + (1 - gamma) / s
    ax.plot(i, np.ones_like(i), color=t["muted"], lw=1.2, ls=":", label="ngoại suy thô (không đổi)")
    ax.plot(i, np.full(len(i), 1 / s), color=t["c"][RED], lw=1.8, label="PI: nén đều mọi tần số (1/s)")
    ax.plot(i, ntk, color=t["c"][BLUE], lw=1.8, label=f"NTK-aware: b′ ≈ {b * s ** (d / (d - 2)):,.0f}".replace(",", "."))
    ax.plot(i, yarn, color=t["c"][GREEN], lw=1.8, ls="--", label="YaRN by-parts (minh họa α = 1, β = 32, L = 4K)")
    ax.set_xlabel("chỉ số cặp i  (trái: tần số cao, chu kỳ ≈ 6,3 token · phải: tần số thấp, chu kỳ ≈ 54.000 token)", fontsize=8.2)
    ax.set_ylabel("θ′ᵢ / θᵢ")
    ax.set_ylim(0, 1.25)
    ax.set_xlim(0, 63)
    ax.legend(fontsize=7.8, loc="upper right", ncol=2)
    ax.set_title("Mở rộng context ×4 (d = 128): mỗi cách chỉnh tần số quay khác nhau", fontsize=9)
    ygrid(ax, t)


# ---------------------------------------------------------------- 5.1 gradient cross-entropy
@figure("ce-gradient", size=(7.8, 2.9))
def _(fig, t):
    ax = fig.subplots()
    p = softmax([2, 1, 0]); y = np.array([0, 1, 0.]); g = p - y
    i = np.arange(3); w = 0.26
    for k, (v, c, lab) in enumerate([(p, t["c"][BLUE], "dự đoán p = softmax(2, 1, 0)"),
                                     (y, t["muted"], "sự thật y (one-hot, token đúng ở giữa)"),
                                     (g, t["c"][ORANGE], "gradient ∂ℓ/∂z = p − y")]):
        ax.bar(i + (k - 1) * w, v, w * .92, color=c, label=lab)
        for xi, vi in zip(i, v):
            if k == 1 and vi == 0:
                continue
            ax.text(xi + (k - 1) * w, vi + (0.03 if vi >= 0 else -0.04), f"{vi:.3f}".replace("-", "−").replace("1.000", "1"),
                    ha="center", va="bottom" if vi >= 0 else "top", fontsize=7.4, color=t["fg"])
    ax.axhline(0, color=t["fg2"], lw=0.8)
    ax.set_xticks(i, ["token A (z = 2)", "token B (z = 1) — đúng", "token C (z = 0)"])
    ax.set_ylim(-1.0, 1.55)
    ax.legend(fontsize=7.8, loc="upper right")
    ax.set_title("ℓ = −ln 0,245 = 1,41; gradient âm ở token đúng → logit đó được đẩy lên", fontsize=9)
    ygrid(ax, t)


# ---------------------------------------------------------------- 5.2 perplexity
@figure("perplexity", size=(7.6, 2.8))
def _(fig, t):
    ax = fig.subplots()
    p = np.array([0.5, 0.25, 0.8, 0.1]); nl = -np.log(p)
    ax.bar(range(4), nl, color=t["c"][VIOLET], width=0.55)
    for k, (pi, v) in enumerate(zip(p, nl)):
        ax.text(k, v + 0.05, f"p = {pi:g}\n−ln p = {v:.3f}".replace(".", ","), ha="center", fontsize=7.8, color=t["fg"])
    L = nl.mean()
    ax.axhline(L, color=t["c"][ORANGE], ls="--", lw=1.4)
    ax.text(3.45, L + 0.08, f"trung bình ℒ = {L:.3f} nat/token\nPPL = e^ℒ = {np.exp(L):.2f}".replace(".", ","),
            ha="left", fontsize=8.2, color=t["fg"], weight="bold")
    ax.set_xticks(range(4), [f"token {k + 1}" for k in range(4)])
    ax.set_ylim(0, 3.2)
    ax.set_xlim(-0.5, 5.1)
    ax.set_ylabel("−ln p (nat)")
    ax.set_title("Perplexity = mũ của loss trung bình: model phân vân như giữa ~3,16 lựa chọn", fontsize=9)
    ygrid(ax, t)


# ---------------------------------------------------------------- 5.3 Chinchilla
@figure("chinchilla", size=(8.2, 3.0))
def _(fig, t):
    ax = fig.subplots()
    E, A, B, a, bb = 1.69, 406.4, 410.7, 0.34, 0.28
    N = 7e9
    D = np.logspace(10, 13.4, 300)
    Lh = E + A / N ** a + B / D ** bb
    ax.plot(D, Lh, color=t["c"][BLUE], lw=1.8, label="L̂(N = 7B, D) = E + A/N^0,34 + B/D^0,28")
    ax.axhline(E + A / N ** a, color=t["c"][ORANGE], ls="--", lw=1.1, label="giới hạn khi D → ∞ (E + A/N^α ≈ 1,87)")
    ax.axhline(E, color=t["muted"], ls=":", lw=1.1, label="E = 1,69 (entropy không giảm được)")
    pts = [(1.4e11, "Chinchilla-tối ưu: 140B token"), (2e12, "2T token"), (1.5e13, "15T token (cỡ Llama 3)")]
    for Dk, lab in pts:
        v = E + A / N ** a + B / Dk ** bb
        ax.plot(Dk, v, "o", color=t["c"][GREEN], ms=6)
        ax.text(Dk * 1.15, v + 0.025, f"{lab}\nL̂ ≈ {v:.2f}".replace(".", ","), fontsize=7.8, color=t["fg"])
    ax.set_xscale("log")
    ax.set_xlabel("số token huấn luyện D")
    ax.set_ylabel("loss dự đoán L̂")
    ax.set_ylim(1.6, 2.75)
    ax.legend(fontsize=7.6, loc="upper right")
    ax.set_title("Chinchilla: cùng model 7B, huấn luyện lâu hơn vẫn giảm loss nhưng chậm dần", fontsize=9)
    ygrid(ax, t)


# ---------------------------------------------------------------- 6.2 temperature
@figure("temperature", size=(7.8, 2.9))
def _(fig, t):
    ax = fig.subplots()
    z = np.array([2, 1, 0.])
    i = np.arange(3); w = 0.26
    for k, (T, c) in enumerate([(0.5, t["c"][RED]), (1, t["c"][BLUE]), (2, t["c"][AQUA])]):
        p = softmax(z / T); H = -(p * np.log(p)).sum()
        ax.bar(i + (k - 1) * w, p, w * .92, color=c, label=f"T = {T:g}   (entropy {H:.2f} nat)".replace(".", ","))
        for xi, v in zip(i, p):
            ax.text(xi + (k - 1) * w, v + 0.015, f"{v:.3f}".replace(".", ","), ha="center", fontsize=7.2, color=t["fg"])
    ax.set_xticks(i, ["token A (z = 2)", "token B (z = 1)", "token C (z = 0)"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("p_T")
    ax.legend(fontsize=7.8, loc="upper right")
    ax.set_title("softmax(z / T): T nhỏ → nhọn, T lớn → phẳng; thứ hạng không đổi", fontsize=9)
    ygrid(ax, t)


# ---------------------------------------------------------------- 6.3 top-p
@figure("top-p", size=(8.8, 3.0))
def _(fig, t):
    a1, a2 = fig.subplots(1, 2, gridspec_kw=dict(wspace=0.28))
    p = np.array([0.5, 0.2, 0.15, 0.1, 0.05]); cs = np.cumsum(p)
    i = np.arange(5)
    cols = [t["c"][BLUE] if k < 4 else t["muted"] for k in i]
    a1.bar(i, p, color=cols, width=0.6, label="p(t)")
    a1.plot(i, cs, "o-", color=t["c"][ORANGE], lw=1.4, ms=4, label="tổng tích lũy")
    for k, v in enumerate(cs):
        a1.text(k, v + 0.04, f"{v:.2f}".replace(".", ","), ha="center", fontsize=7.6, color=t["fg"])
    for th, lab in [(0.9, "p = 0,9"), (0.7, "p = 0,7")]:
        a1.axhline(th, color=t["fg2"], ls="--", lw=0.9)
        a1.text(4.45, th - 0.075, lab, ha="right", fontsize=7.6, color=t["fg2"])
    a1.set_xticks(i, [f"t{k + 1}" for k in i])
    a1.set_ylim(0, 1.18)
    a1.legend(fontsize=7.8, loc="upper left")
    a1.set_title("Phân phối gốc (đã sắp xếp)", fontsize=9)
    ygrid(a1, t)
    w = 0.36
    p9 = np.where(i < 4, p, 0) / 0.95; p7 = np.where(i < 2, p, 0) / 0.7
    a2.bar(i - w / 2, p9, w * .92, color=t["c"][BLUE], label="top-p 0,9: giữ 4 token, chia 0,95")
    a2.bar(i + w / 2, p7, w * .92, color=t["c"][VIOLET], label="top-p 0,7: giữ 2 token, chia 0,70")
    for k in i:
        if p9[k]:
            a2.text(k - w / 2, p9[k] + 0.015, f"{p9[k]:.3f}".replace(".", ","), ha="center", fontsize=6.8, color=t["fg"])
        if p7[k]:
            a2.text(k + w / 2, p7[k] + 0.015, f"{p7[k]:.3f}".replace(".", ","), ha="center", fontsize=6.8, color=t["fg"])
    a2.set_xticks(i, [f"t{k + 1}" for k in i])
    a2.set_ylim(0, 1.0)
    a2.legend(fontsize=7.8, loc="upper right")
    a2.set_title("Sau khi cắt và chuẩn hóa lại", fontsize=9)
    ygrid(a2, t)


# ---------------------------------------------------------------- 7.2–7.3 Bradley–Terry / DPO
@figure("preference-loss", size=(8.0, 3.0))
def _(fig, t):
    ax = fig.subplots()
    x = np.linspace(-4, 4, 400)
    sig = 1 / (1 + np.exp(-x))
    ax.plot(x, sig, color=t["c"][BLUE], lw=1.7, label="σ(Δ) = P(y_w ≻ y_l)")
    ax.plot(x, -np.log(sig), color=t["c"][ORANGE], lw=1.7, label="loss = −ln σ(Δ)")
    pts = [(1.2, "Reward model: Δ = 1,5 − 0,3 = 1,2\nP = 0,769 · loss = 0,263", (1.45, 1.25)),
           (0.3, "DPO: Δ = 0,1 · (2 − (−1)) = 0,3\nloss = 0,554", (1.45, 2.0))]
    for d, lab, pos in pts:
        l = -np.log(1 / (1 + np.exp(-d)))
        ax.plot(d, l, "o", color=t["c"][GREEN], ms=6)
        ax.plot(d, 1 / (1 + np.exp(-d)), "o", color=t["c"][GREEN], ms=6, mfc="none")
        ax.axvline(d, color=t["line"], ls=":", lw=0.9)
        ax.text(*pos, lab, fontsize=7.8, color=t["fg"])
    ax.set_xlabel("Δ = hiệu điểm (reward, hoặc reward ẩn β·log π/π_ref của DPO)")
    ax.set_ylim(0, 3.2)
    ax.set_xlim(-4, 4)
    ax.legend(fontsize=7.8, loc="upper right")
    ax.set_title("Cùng một loss Bradley–Terry: loss chỉ lớn khi cặp đang bị xếp sai thứ tự (Δ < 0)", fontsize=9)
    ygrid(ax, t)


# ---------------------------------------------------------------- 8 ICL
@figure("icl-prompt", size=(8.8, 3.2))
def _(fig, t):
    ax = canvas(fig, (0, 124), (0, 46))
    rows = [("System prompt", "vai trò, quy tắc, định dạng đầu ra", t["muted"]),
            ("Demonstrations", "2–3 macro + ticket cũ CSAT cao, giống email này → văn phong", t["c"][VIOLET]),
            ("Tài liệu tham khảo", "chunk tri thức từ retrieval → căn cứ sự thật (RAG)", t["c"][AQUA]),
            ("Email hiện tại", "x — câu hỏi cần trả lời", t["c"][BLUE])]
    for k, (a, b, c) in enumerate(rows):
        y = 34 - k * 9.5
        box(ax, 2, y, 84, 8, "", t, color=c, fill=tint(c, t, .14))
        ax.text(4, y + 4, a, va="center", fontsize=8.8, weight="bold", color=t["fg"])
        ax.text(29, y + 4, b, va="center", fontsize=7.8, color=t["fg"])
    ax.text(2, 43.5, "Một prompt = một lần in-context learning (không cập nhật trọng số)", fontsize=9, weight="bold", color=t["fg"])
    box(ax, 92, 18, 30, 12, "LLM\nŷ = argmax p(y | demos, docs, x)", t, color=t["c"][GREEN], fill=tint(t["c"][GREEN], t, .16), fs=7.6)
    arrow(ax, 86.5, 24, 91.7, 24, t, lw=1.6)
    ax.text(92, 13, "Giữ tách biệt: demos dạy cách viết,\ntài liệu mới là căn cứ sự thật.\nVí dụ cũ có thể chứa PII hoặc\nchính sách lỗi thời.",
            fontsize=7.6, color=t["fg2"], va="top", linespacing=1.4)


# ---------------------------------------------------------------- 9 KV cache
@figure("kv-cache", size=(8.8, 3.1))
def _(fig, t):
    ax = canvas(fig, (0, 124), (0, 44))
    pc, nc, kc = t["c"][BLUE], t["c"][ORANGE], t["c"][VIOLET]
    ax.text(2, 40.5, "Prefill: cả prompt một lần (song song, nặng tính toán)", fontsize=8.8, weight="bold", color=t["fg"])
    for k in range(8):
        box(ax, 2 + k * 5.4, 28, 4.6, 6, "", t, color=pc, fill=tint(pc, t, .3), radius=0.5)
    ax.text(2, 24.5, "token 1 … n của prompt", fontsize=7.8, color=t["fg2"])
    arrow(ax, 46, 31, 54.5, 31, t, lw=1.5)
    ax.text(50.2, 33, "ghi K, V", ha="center", fontsize=7.6, color=t["fg2"])
    ax.text(68, 40.5, "Decode: mỗi bước 1 token (nặng băng thông bộ nhớ)", fontsize=8.8, weight="bold", color=t["fg"])
    # cache
    box(ax, 55, 6, 30, 30, "", t, color=kc, fill=tint(kc, t, .08), ls="--")
    ax.text(70, 7.5, "KV cache", ha="center", fontsize=8.2, color=t["fg"], weight="bold")
    for k in range(8):
        box(ax, 57.5 + (k % 4) * 6.5, 27 - (k // 4) * 7, 5.5, 5, "", t, color=pc, fill=tint(pc, t, .3), radius=0.5)
    for k in range(3):
        box(ax, 57.5 + (k % 4) * 6.5, 13, 5.5, 5, "", t, color=nc, fill=tint(nc, t, .35), radius=0.5)
    ax.text(87, 9.5, "+1 mỗi bước", fontsize=7.6, color=t["fg2"], va="center")
    box(ax, 104, 22, 18, 9, "token mới:\ntính q, k, v", t, color=nc, fill=tint(nc, t, .2), fs=7.8)
    arrow(ax, 103.7, 26.5, 85.5, 26.5, t, lw=1.3)
    ax.text(94.5, 28, "attend", ha="center", fontsize=7.6, color=t["fg2"])
    arrow(ax, 113, 21.7, 79.5, 15.5, t, lw=1.1, rad=-0.25, ls="--")
    ax.text(2, 13, "Bộ nhớ cache ∝ số lớp × số đầu KV\n× độ dài chuỗi × số request đồng thời\n(chi tiết ở Module 11)",
            fontsize=7.8, color=t["fg2"], va="center", linespacing=1.4)


# ---------------------------------------------------------------- 10 MoE
@figure("moe", size=(8.8, 3.1))
def _(fig, t):
    ax = canvas(fig, (0, 124), (0, 44))
    box(ax, 2, 18, 12, 8, "token h", t, color=t["c"][BLUE], fill=tint(t["c"][BLUE], t, .16), fs=8.4)
    box(ax, 20, 16, 14, 12, "router\nsoftmax(TopK)", t, color=t["c"][VIOLET], fill=tint(t["c"][VIOLET], t, .16), fs=8)
    arrow(ax, 14.3, 22, 19.7, 22, t)
    on = {2: 0.62, 5: 0.38}
    for k in range(8):
        y = 38 - k * 5
        active = k in on
        c = t["c"][ORANGE] if active else t["line"]
        box(ax, 46, y, 20, 4, f"FFN chuyên gia {k + 1}", t, color=c,
            fill=tint(t["c"][ORANGE], t, .25) if active else t["panel"], fs=7.6, weight="bold" if active else "normal",
            tc=t["fg"] if active else t["muted"])
        ax.plot([34.3, 45.7], [22, y + 2], color=c if active else t["grid"], lw=1.5 if active else 0.7)
        if active:
            ax.text(35.5, 27.5 if k < 4 else 14, f"g = {on[k]:.2f}".replace(".", ","), fontsize=7.4, color=t["fg"])
            arrow(ax, 66.3, y + 2, 75.7, 22, t, color=c, lw=1.4)
    box(ax, 76, 17, 14, 10, "Σ gᵢ · FFNᵢ(h)", t, color=t["c"][GREEN], fill=tint(t["c"][GREEN], t, .16), fs=8)
    ax.text(46, 1.5, "chỉ 2/8 chuyên gia chạy cho token này", fontsize=7.8, color=t["fg2"])
    ax.text(95, 37, "tham số tổng vs kích hoạt", fontsize=8.4, weight="bold", color=t["fg"])
    for k, (name, tot, act) in enumerate([("Mixtral 8x7B", 47, 13), ("DeepSeek-V3", 671, 37)]):
        y = 26 - k * 13
        ax.text(95, y + 4, name, fontsize=8.2, color=t["fg"])
        ax.text(95, y - 1, f"~{tot}B tổng · ~{act}B / token\n({act / tot:.0%} tham số kích hoạt)", fontsize=7.8, color=t["fg2"], va="center")
    ax.text(95, 3.5, "bộ nhớ ∝ tổng; tính toán ∝ kích hoạt", fontsize=7.6, color=t["fg2"])


if __name__ == "__main__":
    run("01")
