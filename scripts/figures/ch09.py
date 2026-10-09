"""Hình minh họa cho Module 09 — Huấn luyện / fine-tune cho RAG."""
import numpy as np
from matplotlib.patches import Patch, Rectangle

from figkit import (AQUA, BLUE, GREEN, MAGENTA, ORANGE, RED, VIOLET, YELLOW, arrow, box,
                    canvas, figure, note, run, tint, xgrid, ygrid)


# ---------------------------------------------------------------- 1 chỗ hỏng → công cụ
@figure("where-to-finetune", size=(8.8, 3.3))
def _(fig, t):
    ax = canvas(fig, (0, 124), (-2, 46))
    stages = [("(a) Retrieval", "không lấy được\ntài liệu đúng", "hybrid → fine-tune\nembedding", BLUE),
              ("(b) Ranking", "có nhưng xếp\nhạng thấp", "reranker mạnh hơn →\nfine-tune reranker", AQUA),
              ("(c) Generation", "đọc sai, bịa,\nlẫn distractor", "prompt + verifier →\nSFT/RAFT generator", VIOLET),
              ("(d) Quyết định", "trả lời khi lẽ\nra phải chuyển", "ngưỡng + rule →\nhọc abstention", ORANGE)]
    for i, (a, b, c, col) in enumerate(stages):
        x = 2 + i * 31
        box(ax, x, 30, 28, 9, a, t, color=t["c"][col], fill=tint(t["c"][col], t, .16), fs=9, weight="bold")
        ax.text(x + 14, 26, b, ha="center", va="top", fontsize=8, color=t["fg2"])
        box(ax, x, 4, 28, 10, c, t, fs=8)
        arrow(ax, x + 14, 18.5, x + 14, 14.4, t)
        if i < 3:
            arrow(ax, x + 28.3, 34.5, x + 30.7, 34.5, t)
    ax.text(2, 43, "Fine-tune dạy hành vi và thước đo; tri thức biến động sống trong index", fontsize=9.2, weight="bold", color=t["fg"])
    ax.text(2, 0.5, "hàng dưới: thử cách rẻ trước, chỉ fine-tune khi số đo vẫn thiếu", fontsize=7.8, color=t["fg2"])


# ---------------------------------------------------------------- 2.1 cặp giám sát yếu
@figure("weak-pairs-funnel", size=(7.8, 2.6))
def _(fig, t):
    ax = fig.subplots()
    rows = [("ticket đã giải quyết", 200_000, t["muted"]), ("có link bài HC / macro rõ (~30%)", 60_000, t["c"][BLUE]),
            ("sau lọc CSAT, reopen (~50%)", 30_000, t["c"][GREEN])]
    y = np.arange(3)
    for i, (n, v, c) in enumerate(rows):
        ax.barh(i, v, color=c, height=0.55)
        ax.text(v + 3000, i, f"{v:,}".replace(",", ".") + f" — {n}", va="center", fontsize=8.6, color=t["fg"])
    ax.set_yticks([]); ax.invert_yaxis(); ax.set_xlim(0, 330_000)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("số ticket / cặp (query, passage) — giả định để học")
    ax.set_title("Từ 200.000 ticket đến ~30.000 cặp huấn luyện embedding", fontsize=9.6)
    xgrid(ax, t)


# ---------------------------------------------------------------- 2.3 chia không rò rỉ
@figure("leak-free-split", size=(8.8, 2.9))
def _(fig, t):
    ax = canvas(fig, (0, 124), (0, 40))
    ax.text(2, 37, "Theo thời gian", fontsize=9, weight="bold", color=t["fg"])
    box(ax, 2, 26, 62, 7, "train: ticket đến tháng 6", t, color=t["c"][BLUE], fill=tint(t["c"][BLUE], t, .16), fs=8.4)
    box(ax, 65, 26, 26, 7, "dev / test: tháng 7–8", t, color=t["c"][ORANGE], fill=tint(t["c"][ORANGE], t, .16), fs=8.4)
    box(ax, 93, 26, 29, 7, "golden set: KHÔNG BAO GIỜ train", t, color=t["c"][RED], fill=tint(t["c"][RED], t, .14), fs=7.8)
    ax.text(2, 20, "Theo nhóm tài liệu (group split) — mọi bản dịch, phiên bản của một bài nằm cùng một phía", fontsize=9,
            weight="bold", color=t["fg"])
    for i in range(10):
        test = i in (3, 7)
        c = t["c"][ORANGE] if test else t["c"][BLUE]
        box(ax, 2 + i * 12, 6, 11, 9, f"bài {i + 1}\nvi·en·ja", t, color=c, fill=tint(c, t, .14), fs=7.6, radius=0.6)
    ax.text(2, 1, "Khử trùng lặp gần (MinHash, Module 04) TRƯỚC khi chia.", fontsize=8.2, color=t["fg2"])


# ---------------------------------------------------------------- 3.2 MNRL
@figure("mnrl-cases", size=(8.8, 3.3))
def _(fig, t):
    a1, a2 = fig.subplots(1, 2, gridspec_kw=dict(wspace=0.3, width_ratios=[1.35, 1]))
    cases = [("negative dễ", [0.80, 0.60, 0.30, 0.10]), ("hard negative", [0.80, 0.78, 0.75, 0.30]),
             ("false negative", [0.80, 0.82, 0.30, 0.10])]
    cols = [t["c"][GREEN], t["c"][ORANGE], t["c"][YELLOW], t["c"][MAGENTA]]
    w = 0.2
    for ci, (name, s) in enumerate(cases):
        z = 20 * np.array(s); p = np.exp(z - z.max()); p /= p.sum()
        for k in range(4):
            a1.bar(ci + (k - 1.5) * w, p[k], w * 0.92, color=cols[k])
        a1.text(ci, -0.2, f"L = {-np.log(p[0]):.3f}", ha="center", fontsize=8.6, color=t["fg"], weight="bold")
    a1.set_xticks(range(3), [c[0] for c in cases]); a1.set_ylim(0, 1.15)
    a1.set_ylabel("π = softmax(γ·s), γ = 20")
    a1.legend(handles=[Patch(color=cols[0], label="positive"), Patch(color=cols[1], label="ứng viên 2"),
                       Patch(color=cols[2], label="ứng viên 3"), Patch(color=cols[3], label="ứng viên 4")],
              fontsize=7.8, ncol=4, loc="upper center")
    a1.set_title("Lực đẩy lên mỗi negative ∝ πᵢⱼ", fontsize=9.6)
    ygrid(a1, t)
    g = np.linspace(1, 40, 200)
    s = np.array([0.80, 0.60, 0.30, 0.10])
    L = [-(gg * s[0] - np.log(np.exp(gg * s).sum())) for gg in g]
    a2.plot(g, L, color=t["c"][BLUE])
    for gg in (1, 20):
        v = -(gg * s[0] - np.log(np.exp(gg * s).sum()))
        a2.scatter([gg], [v], color=t["c"][ORANGE], s=30, zorder=4)
        a2.text(gg + 1.5, v + 0.04, f"γ = {gg}: L = {v:.3f}", fontsize=8.3, color=t["fg"], va="bottom" if gg == 20 else "top")
    a2.set_xlabel("γ = 1/τ"); a2.set_ylabel("loss (trường hợp negative dễ)"); a2.set_ylim(0, 1.25)
    a2.set_title("Không scale, softmax quá «mềm»", fontsize=9.6)
    ygrid(a2, t)


# ---------------------------------------------------------------- 3.3 positive-aware mining
@figure("positive-aware-mining", size=(9.0, 3.2))
def _(fig, t):
    ax = fig.subplots()
    sp = 0.80
    cands = [("bản dịch EN\n(cùng họ)", 0.82, "family"), ("bản v2 cũ\n(cùng họ)", 0.79, "family"),
             ("«hoàn tiền\ngói năm»", 0.77, "high"), ("«hạ gói»", 0.72, "ok"), ("«bảng giá»", 0.61, "ok"),
             ("«đổi email»", 0.40, "ok")]
    for i, (n, s, k) in enumerate(cands):
        c = {"family": t["c"][RED], "high": t["c"][YELLOW], "ok": t["c"][GREEN]}[k]
        ax.bar(i, s, color=c, width=0.6)
        ax.text(i, s + 0.015, f"{s:.2f}", ha="center", fontsize=8.3, color=t["fg"])
    ax.axhline(sp, color=t["c"][BLUE], lw=1.4)
    ax.axhline(0.95 * sp, color=t["c"][ORANGE], ls="--", lw=1.4)
    ax.text(5.55, sp + 0.01, "s(q, p) = 0.80", ha="left", va="bottom", fontsize=8.2, color=t["fg"])
    ax.text(5.55, 0.72, "ngưỡng\n0.95·s(q, p)\n= 0.76", ha="left", va="top", fontsize=8.2, color=t["fg"])
    ax.set_xticks(range(len(cands)), [c[0] for c in cands], fontsize=7.6)
    ax.set_ylim(0, 0.95); ax.set_xlim(-0.5, 7.2); ax.set_ylabel("điểm cross-encoder s(q, n)")
    ax.set_title("Chọn hard negative (số minh họa): đỏ = loại theo họ tài liệu, vàng = loại vì quá gần positive", fontsize=9.2)
    ygrid(ax, t)


# ---------------------------------------------------------------- 3.6 VRAM embedding
@figure("embedding-vram", size=(6.8, 2.6))
def _(fig, t):
    ax = fig.subplots()
    rows = [("cỡ small ~118M", 118e6), ("cỡ base ~278M", 278e6), ("cỡ bge-m3 ~568M", 568e6)]
    gb = [n * 16 / 1e9 for _, n in rows]
    cols = [t["c"][GREEN] if g < 6 else t["c"][RED] for g in gb]
    ax.barh(range(3), gb, color=cols, height=0.55)
    for i, g in enumerate(gb):
        ax.text(g + 0.15, i, f"~{g:.1f} GB", va="center", fontsize=8.8, color=t["fg"])
    ax.axvline(6, color=t["muted"], ls="--", lw=1.2)
    ax.text(6.1, -0.42, "RTX 4050: 6 GB", fontsize=8, color=t["fg2"])
    ax.set_yticks(range(3), [r[0] for r in rows]); ax.set_ylim(2.5, -0.65)
    ax.set_xlim(0, 11); ax.set_xlabel("GB cho weight + grad + Adam (16 byte/tham số), chưa tính activation")
    ax.set_title("Full fine-tune embedding trên GPU 6 GB", fontsize=9.6)
    xgrid(ax, t)


# ---------------------------------------------------------------- 4.2 BCE reranker
@figure("reranker-bce", size=(6.8, 3.0))
def _(fig, t):
    ax = fig.subplots()
    z = np.linspace(-4, 4, 300)
    sig = 1 / (1 + np.exp(-z))
    ax.plot(z, -np.log(sig), color=t["c"][GREEN], label="y = 1 (positive): −log σ(z)")
    ax.plot(z, -np.log(1 - sig), color=t["c"][RED], label="y = 0 (negative): −log(1 − σ(z))")
    for zz, y, c in [(2.0, 1, GREEN), (1.5, 0, RED)]:
        s = 1 / (1 + np.exp(-zz)); L = -np.log(s) if y else -np.log(1 - s)
        ax.scatter([zz], [L], color=t["c"][c], s=34, zorder=4, edgecolor=t["bg"])
        ax.text(zz + 0.15, L + 0.15, f"logit {zz}: p̂ = {s:.3f}, L = {L:.3f}", fontsize=8.2, color=t["fg"])
    ax.set_xlabel("logit z = f_φ(q, d)"); ax.set_ylabel("BCE")
    ax.set_ylim(0, 4.5)
    ax.set_title("Hard negative có logit cao → loss lớn → gradient lớn", fontsize=9.6)
    ax.legend(fontsize=8, loc="upper center")
    ygrid(ax, t)


# ---------------------------------------------------------------- 5.1 SFT mask
@figure("sft-mask", size=(8.6, 2.6))
def _(fig, t):
    ax = canvas(fig, (0, 124), (0, 34))
    total = 2600; W = 118
    xw = W * 2400 / total
    box(ax, 2, 18, xw, 8, "prompt x: system + sources + email (2.400 token) — nhãn −100", t, color=t["muted"], fill=t["panel2"], fs=8.2)
    box(ax, 2 + xw + 0.5, 18, W - xw - 0.5, 8, "y (200)", t, color=t["c"][GREEN], fill=tint(t["c"][GREEN], t, .25), fs=8, weight="bold")
    ax.text(2, 30, "Một mẫu SFT trong RAG: x dài gấp 12 lần y", fontsize=9.2, weight="bold", color=t["fg"])
    ax.text(2, 12, "Không mask: loss ≈ 1.154, câu trả lời chỉ góp 4% tín hiệu (chủ yếu dạy chép tài liệu, email, PII).", fontsize=8.4, color=t["fg"])
    ax.text(2, 7, "Có mask (completion-only): loss = 0.6, 100% tín hiệu đến từ hành vi cần dạy.", fontsize=8.4, color=t["fg"], weight="bold")
    ax.text(2, 2, "Giả định: 1.2 nat/token prompt, 0.6 nat/token trả lời (ví dụ mục 5.1).", fontsize=7.8, color=t["fg2"])


# ---------------------------------------------------------------- 5.3 RAFT
@figure("raft-mix", size=(8.8, 3.2))
def _(fig, t):
    ax = canvas(fig, (0, 124), (0, 44))
    mix = [("A · oracle + distractor", 60, BLUE), ("B · chỉ distractor → abstain", 20, ORANGE),
           ("C · một phần", 10, AQUA), ("D · mâu thuẫn", 5, VIOLET), ("E · injection", 5, RED)]
    x = 2
    for name, p, c in mix:
        w = 1.2 * p
        ax.add_patch(Rectangle((x, 33), w - 0.4, 6, color=t["c"][c], lw=0))
        if p >= 10:
            ax.text(x + w / 2, 36, f"{p}%", ha="center", va="center", fontsize=8.6, color="#ffffff", weight="bold")
        x += w
    ax.text(2, 41.5, "Tỷ lệ loại mẫu RAFT cho Zendesk (điểm xuất phát)", fontsize=9, weight="bold", color=t["fg"])
    for i, (name, p, c) in enumerate(mix):
        ax.add_patch(Rectangle((2 + i * 24.4, 28), 2, 2, color=t["c"][c], lw=0))
        ax.text(5 + i * 24.4, 29, name, fontsize=7.5, color=t["fg"], va="center")
    ax.text(2, 22, "Ví dụ mẫu A: khách hỏi giới hạn API gói Business", fontsize=8.8, weight="bold", color=t["fg"])
    srcs = [("S1 Enterprise:\n1.000 req/phút", ORANGE), ("S2 Business:\n300 req/phút", GREEN), ("S3 distractor", t["muted"]),
            ("S4 distractor", t["muted"])]
    for i, (s, c) in enumerate(srcs):
        col = t["c"][c] if isinstance(c, int) else c
        box(ax, 2 + i * 17, 6, 15.5, 11, s, t, color=col, fill=tint(col, t, .15), fs=7.6, lw=2.2 if i == 1 else 1.2)
    arrow(ax, 71, 11.5, 76, 11.5, t, lw=1.6)
    box(ax, 77, 4, 45, 15, "claims: «Gói Business giới hạn 300\nrequest/phút.» → [S2]\nanswer_plan: «S1 là gói Enterprise,\nkhông áp dụng»", t,
        color=t["c"][BLUE], fill=tint(t["c"][BLUE], t, .1), fs=7.8, ha="left")
    ax.text(2, 1, "distractor lấy từ chính retriever production (xáo thứ tự)", fontsize=7.8, color=t["fg2"])


# ---------------------------------------------------------------- 5.4 abstention trade-off
@figure("abstention-tradeoff", size=(7.4, 3.3))
def _(fig, t):
    ax = fig.subplots()
    pts = [("Base + prompt", 0.97, 0.41, 307, t["muted"]), ("SFT chỉ loại A", 0.98, 0.22, 398, t["c"][RED]),
           ("SFT A+B+C", 0.94, 0.87, 89, t["c"][GREEN]), ("SFT B = 50%", 0.78, 0.96, 108, t["c"][ORANGE])]
    for name, ans, ab, cost, c in pts:
        ax.scatter([ab], [ans], s=90, color=c, zorder=4, edgecolor=t["bg"], lw=1.2)
        dx = -0.02 if ab > 0.8 else 0.02
        ax.text(ab + dx, ans + (-0.035 if name == "SFT B = 50%" else 0.012), f"{name}\nchi phí = {cost}", fontsize=8.2, color=t["fg"],
                ha="right" if ab > 0.8 else "left")
    ax.set_xlabel("AbsRate_U — abstain đúng khi context thiếu"); ax.set_ylabel("AnsRate_A — trả lời khi có đáp án")
    ax.set_xlim(0.1, 1.05); ax.set_ylim(0.72, 1.03)
    ax.text(1.03, 1.015, "góc lý tưởng", fontsize=8, color=t["fg2"], ha="right")
    ax.set_title("Bốn phiên bản trên dev (400 câu A, 100 câu U); chi phí với c_w = 5, c_e = 1", fontsize=9.4)
    ygrid(ax, t)


# ---------------------------------------------------------------- 6 DPO / ORPO
@figure("dpo-orpo", size=(8.8, 3.1))
def _(fig, t):
    a1, a2 = fig.subplots(1, 2, gridspec_kw=dict(wspace=0.3))
    d = np.linspace(-10, 20, 300); beta = 0.1
    a1.plot(d, -np.log(1 / (1 + np.exp(-beta * d))), color=t["c"][BLUE])
    for dv, lab in [(0, "bắt đầu: ln 2 = 0.693"), (3, "ví dụ: r̂_w − r̂_l = 3 → 0.554")]:
        v = -np.log(1 / (1 + np.exp(-beta * dv)))
        a1.scatter([dv], [v], color=t["c"][ORANGE], s=30, zorder=4)
        a1.text(dv + 0.8, v + (0.06 if dv == 0 else -0.12), lab, fontsize=8.2, color=t["fg"])
    a1.set_xlabel("r̂_w − r̂_l (phần thưởng ngầm)"); a1.set_ylabel("L_DPO, β = 0.1")
    a1.set_title("DPO", fontsize=9.6)
    ygrid(a1, t)
    pw = np.linspace(0.05, 0.95, 300); pl = 0.4
    odds = lambda p: p / (1 - p)
    pen = -np.log(1 / (1 + np.exp(-np.log(odds(pw) / odds(pl)))))
    a2.plot(pw, pen, color=t["c"][VIOLET])
    for pv in (0.4, 0.6):
        v = -np.log(1 / (1 + np.exp(-np.log(odds(pv) / odds(pl)))))
        a2.scatter([pv], [v], color=t["c"][ORANGE], s=30, zorder=4)
        a2.text(pv + 0.03, v + 0.05, f"P(y_w) = {pv}: {v:.3f}", fontsize=8.2, color=t["fg"])
    a2.set_xlabel("P_θ(y_w | x) (chuẩn hóa độ dài), P_θ(y_l | x) = 0.4"); a2.set_ylabel("số hạng phạt odds ratio")
    a2.set_title("ORPO", fontsize=9.6)
    ygrid(a2, t)


# ---------------------------------------------------------------- 7.1 LoRA
@figure("lora", size=(8.4, 3.1))
def _(fig, t):
    from matplotlib.patches import Circle
    ax = canvas(fig, (0, 120), (0, 44))
    box(ax, 2, 24, 8, 8, "x", t, fs=10)
    box(ax, 22, 22, 26, 12, "W₀  (d_out × d_in)\nđóng băng", t, color=t["muted"], fill=t["panel2"], fs=8.8)
    box(ax, 22, 4, 16, 8, "A: r × d_in", t, color=t["c"][ORANGE], fill=tint(t["c"][ORANGE], t, .2), fs=8.4)
    box(ax, 44, 4, 16, 8, "B: d_out × r", t, color=t["c"][BLUE], fill=tint(t["c"][BLUE], t, .2), fs=8.4)
    ax.add_patch(Circle((72, 28), 3, fill=False, ec=t["fg2"], lw=1.4))
    ax.text(72, 28, "+", ha="center", va="center", fontsize=12, color=t["fg"])
    box(ax, 84, 24, 10, 8, "h", t, fs=10)
    arrow(ax, 10.3, 28, 21.7, 28, t)
    arrow(ax, 48.3, 28, 68.7, 28, t)
    ax.plot([15, 15], [28, 8], color=t["fg2"], lw=1.5)
    arrow(ax, 15, 8, 21.7, 8, t)
    arrow(ax, 38.3, 8, 43.7, 8, t)
    ax.plot([60.3, 72], [8, 8], color=t["c"][BLUE], lw=1.5)
    arrow(ax, 72, 8, 72, 24.8, t, color=t["c"][BLUE])
    ax.text(73.5, 15, "× α/r", fontsize=8.4, color=t["fg2"])
    arrow(ax, 75.3, 28, 83.7, 28, t)
    ax.text(2, 40, "h = W₀x + (α/r)·B A x", fontsize=10, color=t["fg"], weight="bold")
    ax.text(84, 15, "khởi tạo: A ngẫu nhiên,\nB = 0 → ΔW = 0", fontsize=8, color=t["fg2"])
    ax.text(84, 6, "tham số: r(d_in + d_out)\nthay vì d_in·d_out", fontsize=8, color=t["fg2"])


# ---------------------------------------------------------------- 7.2 đếm tham số
@figure("lora-params", size=(8.4, 3.0))
def _(fig, t):
    ax = fig.subplots()
    mats = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    p17 = [65536, 49152, 49152, 65536, 131072, 131072, 131072]
    p4 = [106496, 57344, 57344, 106496, 196608, 196608, 196608]
    x = np.arange(len(mats)); w = 0.38
    ax.bar(x - w / 2, np.array(p17) / 1000, w * 0.92, color=t["c"][BLUE], label="Qwen3-1.7B: 622.592/lớp × 28 ≈ 17,4M (~1,0%)")
    ax.bar(x + w / 2, np.array(p4) / 1000, w * 0.92, color=t["c"][ORANGE], label="Qwen3-4B: 917.504/lớp × 36 ≈ 33,0M (~0,8%)")
    ax.set_xticks(x, mats, fontsize=8.4)
    ax.set_ylabel("nghìn tham số LoRA / lớp (r = 16)")
    ax.set_ylim(0, 260)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Tham số LoRA theo từng ma trận chiếu (bảng mục 7.2)", fontsize=9.6)
    ygrid(ax, t)


# ---------------------------------------------------------------- 7.3 NF4
NF4 = [-1.0, -0.6961928, -0.5250731, -0.3949175, -0.2844414, -0.1847734, -0.09105004, 0.0,
       0.0795803, 0.1609302, 0.2461123, 0.3379152, 0.4407098, 0.562617, 0.7229568, 1.0]


@figure("nf4-levels", size=(8.4, 3.0))
def _(fig, t):
    ax = fig.subplots()
    xs = np.linspace(-1.1, 1.1, 400)
    sd = 1 / 2.4
    ax.fill_between(xs, 0, np.exp(-xs ** 2 / (2 * sd ** 2)), color=t["c"][VIOLET], alpha=0.12, lw=0)
    for v in NF4:
        ax.plot([v, v], [0.55, 1.0], color=t["c"][BLUE], lw=1.6)
    for v in np.linspace(-1, 1, 16):
        ax.plot([v, v], [0.0, 0.42], color=t["c"][ORANGE], lw=1.6)
    ax.text(-1.08, 1.05, "NF4: 16 mức theo phân vị phân phối chuẩn (dày ở gần 0)", fontsize=8.4, color=t["fg"])
    ax.text(-1.08, 0.46, "INT4 đều: phí mức ở vùng đuôi thưa", fontsize=8.4, color=t["fg"])
    ax.annotate("w/c = 0.2625\n→ mức 0.2461\n→ 0.0197\n(sai số ≈ 6%)", xy=(0.2625, 0.78), xytext=(1.18, 0.62),
                fontsize=8, color=t["fg"], arrowprops=dict(arrowstyle="-|>", color=t["fg2"], lw=1))
    ax.scatter([0.2625], [0.78], color=t["c"][RED], s=26, zorder=5)
    ax.set_xlim(-1.12, 1.55); ax.set_ylim(0, 1.15); ax.set_yticks([]); ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("trọng số đã chuẩn hóa w / absmax của block (block 64)")
    ax.set_title("NormalFloat 4-bit so với INT4 đều", fontsize=9.6)


# ---------------------------------------------------------------- 7.4 VRAM QLoRA
@figure("qlora-vram", size=(8.8, 3.4))
def _(fig, t):
    ax = fig.subplots()
    names = ["Qwen3-1.7B, S = 3.072", "Qwen3-4B, S = 2.048"]
    comp = [("NF4 tuyến tính", [0.73, 1.87], BLUE), ("embedding bf16", [0.62, 0.78], AQUA), ("LoRA + Adam 8-bit", [0.17, 0.33], GREEN),
            ("activation", [0.35, 0.38], VIOLET), ("CUDA, tính lại lớp (giữa dải)", [0.9, 0.9], YELLOW),
            ("logits fp32 + gradient", [3.74, 2.48], RED)]
    for i in range(2):
        left = 0
        for name, v, c in comp:
            ax.barh(i, v[i], left=left, color=t["c"][c], height=0.5, label=name if i == 0 else None,
                    hatch="//" if c == RED else None, edgecolor=t["bg"] if c == RED else None)
            left += v[i]
        before = sum(v[i] for _, v, _ in comp[:-1])
        ax.text(0.05, i + 0.4, f"không tính logits ≈ {before:.1f} GB", fontsize=7.8, color=t["fg2"], va="center")
        ax.text(left + 0.08, i, f"≈ {left:.1f} GB nếu không fused CE", va="center", fontsize=8.2, color=t["fg"])
    ax.axvline(6, color=t["fg2"], ls="--", lw=1.3)
    ax.text(6.05, -0.42, "6 GB", fontsize=8.2, color=t["fg"])
    ax.set_yticks([0, 1], names); ax.set_ylim(1.6, -0.55)
    ax.set_xlim(0, 10.5); ax.set_xlabel("GB (ước lượng theo bảng mục 7.4)")
    ax.set_title("Thủ phạm OOM là logits S × V, không phải trọng số", fontsize=9.6)
    ax.legend(fontsize=7.4, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.3))
    xgrid(ax, t)


# ---------------------------------------------------------------- 8.2 forward vs reverse KL
@figure("forward-reverse-kl", size=(8.8, 3.1))
def _(fig, t):
    a1, a2 = fig.subplots(1, 2, gridspec_kw=dict(wspace=0.28, width_ratios=[1.3, 1]))
    x = np.linspace(-6, 6, 500)
    N = lambda m, s: np.exp(-(x - m) ** 2 / (2 * s ** 2)) / (s * np.sqrt(2 * np.pi))
    p = 0.5 * N(-2.2, 0.7) + 0.5 * N(2.2, 0.7)
    a1.fill_between(x, 0, p, color=t["muted"], alpha=0.3, lw=0, label="giáo viên p_T (hai mode)")
    a1.plot(x, N(0, 2.4), color=t["c"][BLUE], label="học trò theo forward KL: phủ cả hai")
    a1.plot(x, N(2.2, 0.75), color=t["c"][ORANGE], label="học trò theo reverse KL: chọn một")
    a1.set_yticks([]); a1.spines["left"].set_visible(False)
    a1.set_ylim(0, 0.62)
    a1.legend(fontsize=7.8, loc="upper left")
    a1.set_title("«Phủ» và «chọn» (minh họa)", fontsize=9.6)
    vals = [0.511, 0.368]
    b = a2.bar(["forward\nKL(p_T‖q)", "reverse\nKL(q‖p_T)"], vals, color=[t["c"][BLUE], t["c"][ORANGE]], width=0.55)
    for bb, v in zip(b, vals):
        a2.text(bb.get_x() + bb.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center", fontsize=9, color=t["fg"])
    a2.set_ylim(0, 0.65)
    a2.set_title("p_T = (0.5, 0.5), học trò B = (0.9, 0.1)", fontsize=9.4)
    ygrid(a2, t)


# ---------------------------------------------------------------- 9 cổng phát hành
@figure("release-gate", size=(8.8, 2.8))
def _(fig, t):
    ax = canvas(fig, (0, 124), (0, 38))
    box(ax, 2, 13, 18, 12, "adapter mới\n+ manifest", t, color=t["c"][VIOLET], fill=tint(t["c"][VIOLET], t, .15), fs=8.4)
    checks = ["không kém hơn ở mọi tầng intent × ngôn ngữ", "tốt hơn ở chỉ số mục tiêu", "JSON hợp lệ ≥ 99,5%",
              "không vi phạm policy (bộ đối kháng)", "canary / memorization probe sạch"]
    for i, c in enumerate(checks):
        box(ax, 28, 31 - i * 6.5, 62, 5.2, f"✓ {c}", t, color=t["c"][GREEN], fill=tint(t["c"][GREEN], t, .1), fs=8, ha="left", radius=0.6)
    arrow(ax, 20.3, 19, 27.7, 19, t, lw=1.6)
    arrow(ax, 90.3, 19, 97.7, 19, t, lw=1.6)
    box(ax, 98, 21, 24, 8, "shadow mode →\nproduction", t, color=t["c"][GREEN], fill=tint(t["c"][GREEN], t, .18), fs=8.2)
    box(ax, 98, 9, 24, 8, "một điều kiện trượt\n→ giữ bản hiện hành", t, color=t["c"][RED], fill=tint(t["c"][RED], t, .12), fs=7.8)
    ax.text(2, 1.5, "Kiểm định cặp theo Module 10; manifest ghi ID ticket để xử lý yêu cầu xóa dữ liệu.", fontsize=8.2, color=t["fg2"])


if __name__ == "__main__":
    run("09")
